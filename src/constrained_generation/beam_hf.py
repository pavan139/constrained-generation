"""Optional actual local-model scorer for the compiled/reference beam engine."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from .compiled import beam_search, greedy_search, tokenizer_backend
from .core import score, summarize, tasks
from .hf import context_limit


class HFCausalScorer:
    """Full-prefix logits: transparent baseline, no claimed KV-cache speedup."""

    def __init__(self, model, prompt_ids):
        import torch

        self.torch, self.model, self.prompt_ids = torch, model, prompt_ids

    def __call__(self, generated_ids):
        torch = self.torch
        continuation = torch.tensor([generated_ids], dtype=torch.long, device=self.prompt_ids.device)
        inputs = torch.cat((self.prompt_ids, continuation), dim=1)
        with torch.inference_mode():
            logits = (
                self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
                .logits[0, -1]
                .float()
            )
            values = torch.log_softmax(logits, dim=-1).cpu().tolist()
        return dict(enumerate(values))


def run_beams(
    model_path: str, count: int = 20, token_budget: int = 128, beam_width: int = 4, force_reference: bool = False
) -> dict:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("Supply a local checkpoint directory; no downloads are performed")
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError("count must be in 1..20")
    if type(token_budget) is not int or not 1 <= token_budget <= 256:
        raise ValueError("token_budget must be in 1..256")
    if type(beam_width) is not int or not 1 <= beam_width <= 64:
        raise ValueError("beam_width must be in 1..64")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional ml extra") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True)
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("A decoder-only causal model is required")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    vocabulary_size = model.get_output_embeddings().weight.shape[0]
    input_vocabulary_size = model.get_input_embeddings().num_embeddings
    if vocabulary_size > input_vocabulary_size:
        raise ValueError("Output token IDs must fit the model's input embeddings")
    reports = {"greedy_unconstrained_hf": [], "grammar_beam_hf": [], "semantic_beam_hf": []}
    for task in tasks()[:count]:
        prompt = (
            "Write ONLY a program. Grammar: zero to eight commands followed by STOP. "
            "Commands: MOVE D n; where D is N/E/S/W and n is 1/2/3; PICK a; or DROP a; "
            "(items a,b,c allowed). Literal spacing only; no newlines or prose. "
            "Grid is 5x5. Total movement budget is 6. "
            f"Initial state: {json.dumps(task.initial.__dict__)}. Instruction: {task.instruction}\nProgram:"
        )
        encoded = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        if int(encoded.min()) < 0 or int(encoded.max()) >= input_vocabulary_size:
            raise ValueError("Prompt tokenizer IDs exceed model input embeddings")
        if encoded.shape[1] + token_budget > context_limit(model.config, tokenizer):
            raise ValueError("Prompt plus decoding budget exceeds the model/tokenizer context")
        for name in reports:
            backend = tokenizer_backend(
                tokenizer, vocabulary_size, task.initial, name == "semantic_beam_hf", force_reference=force_reference
            )
            policy = HFCausalScorer(model, encoded)
            if device == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            generation = (
                greedy_search(policy, backend, token_budget, False)
                if name == "greedy_unconstrained_hf"
                else beam_search(policy, backend, beam_width, token_budget)
            )
            if device == "cuda":
                torch.cuda.synchronize()
            generation["latency_ms"] = (perf_counter() - start) * 1000
            reports[name].append(score(task, generation))
    return {
        "experiment": "actual local HF original-likelihood greedy/compiled-or-reference beam search",
        "learned_model_evaluated": True,
        "model_path": str(path),
        "device": device,
        "beam_width": beam_width,
        "token_budget": token_budget,
        "kv_cache": False,
        "approaches": {name: summarize(rows) for name, rows in reports.items()},
    }
