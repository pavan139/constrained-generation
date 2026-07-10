"""Opt-in random-checkpoint regression: forced EOS must not override masks."""

import os
from pathlib import Path
import tempfile
import unittest

from constrained_generation.hf import run


@unittest.skipUnless(os.environ.get("RUN_ML_SMOKE") == "1", "Set RUN_ML_SMOKE=1 with the ml extra installed")
class LocalHuggingFaceSmoke(unittest.TestCase):
    def test_bytelevel_compiled_masks_match_reference_and_run_actual_beams(self):
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        from constrained_generation.beam_hf import run_beams
        from constrained_generation.compiled import tokenizer_backend
        from constrained_generation.core import COMMANDS, Grammar, SemanticGrammar, State, token_allowlist

        torch.set_num_threads(1)
        torch.manual_seed(7)
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            list(COMMANDS) + ["STOP"],
            trainers.BpeTrainer(
                vocab_size=300, special_tokens=["<eos>", "<unk>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
            ),
        )
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, eos_token="<eos>", unk_token="<unk>", model_max_length=1024
        )
        initial = State(items=(("a", 3, 2),))
        for semantic in (False, True):
            backend = tokenizer_backend(fast, len(fast), initial, semantic)
            self.assertEqual(backend.stats()["backend"], "compiled_token_trie")
            grammar = SemanticGrammar(initial) if semantic else Grammar()
            for prefix in ("", "M", "MOVE ", "MOVE E ", "MOVE E 1;", "MOVE E 1;P", "STOP"):
                ids = tuple(fast(prefix, add_special_tokens=False)["input_ids"])
                self.assertEqual(fast.decode(ids), prefix)
                state = backend.compiler.state_for_prefix(prefix)
                self.assertIsNotNone(state)
                self.assertEqual(
                    set(backend.allowed(ids, state)), set(token_allowlist(fast, list(ids), grammar, len(fast)))
                )
            fallback = tokenizer_backend(fast, len(fast), initial, semantic, force_reference=True)
            self.assertEqual(fallback.stats()["backend"], "slow_reference_tokenizer")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            fast.save_pretrained(path)
            model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=len(fast),
                    n_positions=1024,
                    n_ctx=1024,
                    n_embd=8,
                    n_layer=1,
                    n_head=2,
                    eos_token_id=fast.eos_token_id,
                )
            )
            model.save_pretrained(path)
            report = run_beams(str(path), count=1, token_budget=4, beam_width=2)
            self.assertTrue(report["learned_model_evaluated"])
            self.assertFalse(report["kv_cache"])
            for mode in report["approaches"].values():
                self.assertEqual(mode["count"], 1)
                self.assertLessEqual(mode["cases"][0]["tokens"], 4)

    def test_inherited_forced_eos_and_min_length_cannot_override_mask(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = Tokenizer(models.WordLevel({"<eos>": 0, "<unk>": 1, "STOP": 2, "<bos>": 3}, unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            fast = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                eos_token="<eos>",
                unk_token="<unk>",
                bos_token="<bos>",
                model_max_length=1024,
            )
            path = Path(directory)
            fast.save_pretrained(path)
            model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=4,
                    n_positions=1024,
                    n_ctx=1024,
                    n_embd=16,
                    n_layer=1,
                    n_head=2,
                    bos_token_id=3,
                    eos_token_id=0,
                )
            ).eval()
            model.generation_config.forced_eos_token_id = 0
            model.generation_config.min_new_tokens = 8
            model.save_pretrained(path)
            encoded = fast("Return a program:", return_tensors="pt")
            before = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=1,
                prefix_allowed_tokens_fn=lambda *_: [2],
                eos_token_id=0,
                pad_token_id=0,
            )
            self.assertEqual(before[0, encoded["input_ids"].shape[1] :].tolist(), [0])
            report = run(str(path), 1, 1)
            for mode in ("grammar_masked_hf", "semantic_masked_hf"):
                row = report["approaches"][mode]["cases"][0]
                self.assertEqual(row["text"], "STOP")
                self.assertTrue(row["syntax_valid"])
                self.assertTrue(row["budget_exhausted"])


if __name__ == "__main__":
    unittest.main()
