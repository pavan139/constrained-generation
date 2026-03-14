"""Actual optional Hugging Face token-masked decoding from a local checkpoint."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from time import perf_counter

from .core import Grammar, SemanticGrammar, score, summarize, tasks, token_allowlist


def context_limit(config, tokenizer, fallback=4096):
    limits = [
        value
        for key in ("max_position_embeddings", "n_positions", "seq_length")
        if type(value := getattr(config, key, None)) is int and value > 0
    ]
    limit = getattr(tokenizer, "model_max_length", None)
    if type(limit) is int and 0 < limit <= 1_000_000:
        limits.append(limit)
    return min(limits) if limits else fallback


def run(model_path: str, count: int, token_budget: int) -> dict:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("--model must point to an existing local checkpoint directory")
    if type(count) is not int or type(token_budget) is not int or not 1 <= count <= 20 or not 1 <= token_budget <= 256:
        raise ValueError("count must be 1..20; token budget must be 1..256")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from transformers.generation.utils import GenerationMixin
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies: pip install -e '.[ml]'") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True)
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("A decoder-only causal model is required")
    vocab_size = model.get_output_embeddings().weight.shape[0]
    if type(tokenizer.eos_token_id) is not int or not 0 <= tokenizer.eos_token_id < vocab_size:
        raise ValueError("Tokenizer EOS is outside the model output vocabulary")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    all_rows = {"unconstrained_hf": [], "grammar_masked_hf": [], "semantic_masked_hf": []}
    for task in tasks()[:count]:
        prompt = (
            "Write ONLY a program. Grammar: zero to eight commands followed by STOP. "
            "Commands: MOVE D n; where D is N/E/S/W and n is 1/2/3; PICK a; or DROP a; "
            "(items a,b,c allowed). Literal spacing only; no newlines or prose. "
            "Grid is 5x5. Total movement budget is 6. "
            f"Initial state: {json.dumps(task.initial.__dict__)}. Instruction: {task.instruction}\nProgram:"
        )
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = encoded["input_ids"].shape[1]
        context = context_limit(model.config, tokenizer)
        if prompt_length + token_budget > context:
            raise ValueError("Prompt plus generation exceeds model context")
        for mode in all_rows:
            cache = {}
            grammar = SemanticGrammar(task.initial) if mode == "semantic_masked_hf" else Grammar()

            def allowed(_batch_id, input_ids):
                suffix = tuple(input_ids[prompt_length:].tolist())
                if suffix not in cache:
                    cache[suffix] = token_allowlist(tokenizer, list(suffix), grammar, vocab_size)
                return cache[suffix]

            kwargs = {"prefix_allowed_tokens_fn": allowed} if mode != "unconstrained_hf" else {}
            # A checkpoint's forced BOS/EOS, min length, beams or suppressions
            # must not override the advertised greedy constraint experiment.
            generation_config = GenerationConfig(
                do_sample=False,
                num_beams=1,
                max_new_tokens=token_budget,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            defaults = (
                {"use_model_defaults": False}
                if "use_model_defaults" in inspect.signature(GenerationMixin.generate).parameters
                else {}
            )
            if device == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            with torch.no_grad():
                output = model.generate(**encoded, generation_config=generation_config, **defaults, **kwargs)
            if device == "cuda":
                torch.cuda.synchronize()
            latency_ms = (perf_counter() - start) * 1000
            ids = output[0, prompt_length:].tolist()
            ended = bool(ids and ids[-1] == tokenizer.eos_token_id)
            text = tokenizer.decode(
                ids[:-1] if ended else ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            generation = {
                "text": text,
                "tokens": len(ids),
                "eos": ended,
                "budget_exhausted": len(ids) >= token_budget and not ended,
                "mask_interventions": None,
                "latency_ms": latency_ms,
            }
            all_rows[mode].append(score(task, generation))
    return {
        "experiment": "actual local HF greedy generation: unconstrained, grammar, semantic constraints",
        "learned_model_evaluated": True,
        "model_path": str(path),
        "token_budget": token_budget,
        "device": device,
        "context_limit": context_limit(model.config, tokenizer),
        "generation_settings": "fresh greedy config; no inherited forced tokens or length penalties",
        "approaches": {key: summarize(rows) for key, rows in all_rows.items()},
    }
