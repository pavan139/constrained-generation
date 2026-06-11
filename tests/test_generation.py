import unittest
from dataclasses import replace
from types import SimpleNamespace

from constrained_generation.core import (
    Grammar,
    ReplayPolicy,
    SemanticContractError,
    SemanticGrammar,
    State,
    SyntaxContractError,
    evaluate,
    execute,
    generate,
    parse,
    score,
    tasks,
    token_allowlist,
)
from constrained_generation.hf import context_limit


class GrammarTests(unittest.TestCase):
    def test_complete_and_partial_states(self):
        grammar = Grammar()
        self.assertEqual(grammar.status(""), (True, False))
        self.assertEqual(grammar.status("MOVE N"), (True, False))
        self.assertEqual(grammar.status("STOP"), (True, True))
        self.assertFalse(grammar.accepts("MOVE N 1;"))
        self.assertTrue(grammar.accepts("MOVE N 1;PICK a;DROP a;STOP"))

    def test_negative_syntax(self):
        for text in (
            "MOVE X 1;STOP",
            "MOVE N 4;STOP",
            "MOVE N 0;STOP",
            "MOVE N 1 STOP",
            "STOP extra",
            " STOP",
            "STOP\n",
            "PICK secret;STOP",
            "DROP a;",
            "",
            "MOVE N 1;;STOP",
            "STOPSTOP",
        ):
            with self.subTest(text=text):
                self.assertFalse(Grammar().accepts(text))
                with self.assertRaises(SyntaxContractError):
                    parse(text)

    def test_exact_command_budget(self):
        grammar = Grammar()
        self.assertTrue(grammar.accepts("PICK a;" * 8 + "STOP"))
        self.assertFalse(grammar.accepts("PICK a;" * 9 + "STOP"))
        self.assertEqual(grammar.allowed_characters("PICK a;" * 8), ("S",))

    def test_allowed_characters_at_decision_point(self):
        self.assertEqual(set(Grammar().allowed_characters("MOVE ")), set("NESW"))
        self.assertEqual(set(Grammar().allowed_characters("MOVE N ")), set("123"))
        self.assertEqual(Grammar().allowed_characters("STOP"), ())
        with self.assertRaises(SyntaxContractError):
            Grammar().allowed_characters("INVALID")

    def test_every_valid_program_prefix_is_valid(self):
        for task in tasks():
            for index in range(len(task.expected) + 1):
                self.assertTrue(Grammar().status(task.expected[:index])[0])


class SemanticTests(unittest.TestCase):
    def test_stop_and_pick_move_drop(self):
        state = State(items=(("a", 2, 2),))
        self.assertEqual(execute("STOP", state), state)
        self.assertEqual(execute("PICK a;MOVE E 1;DROP a;STOP", state), State(3, 2, items=(("a", 3, 2),)))

    def test_syntax_does_not_imply_semantics(self):
        for text in ("PICK b;STOP", "DROP a;STOP", "MOVE N 3;STOP"):
            with self.subTest(text=text):
                self.assertTrue(Grammar().accepts(text))
                with self.assertRaises(SemanticContractError):
                    execute(text)

    def test_total_movement_budget_boundary(self):
        valid = "MOVE E 2;MOVE W 2;MOVE E 2;STOP"
        self.assertEqual(execute(valid).x, 4)
        with self.assertRaisesRegex(SemanticContractError, "budget"):
            execute(valid[:-4] + "MOVE W 1;STOP")
        self.assertEqual(execute("STOP", movement_budget=0), State())
        with self.assertRaises(SemanticContractError):
            execute("MOVE E 1;STOP", movement_budget=0)

    def test_failed_execution_keeps_original_state(self):
        initial = State(items=(("a", 2, 2),))
        with self.assertRaises(SemanticContractError):
            execute("PICK a;MOVE N 3;STOP", initial)
        self.assertEqual(initial, State(items=(("a", 2, 2),)))

    def test_malformed_state_is_explicit(self):
        bad = (
            None,
            State(items=(("a", 2),)),
            State(inventory=("a",), items=(("a", 2, 2),)),
            State(x=True),
            State(items=(("a", -1, 0),)),
            State(inventory=({},)),
        )
        for initial in bad:
            with self.subTest(initial=initial), self.assertRaises(SemanticContractError):
                execute("STOP", initial)

    def test_golden_programs_are_executable(self):
        self.assertGreaterEqual(len(tasks()), 12)
        for task in tasks():
            self.assertEqual(execute(task.expected, task.initial), task.goal)

    def test_independent_goal_integrity(self):
        task = replace(tasks()[0], goal=State(4, 4))
        with self.assertRaisesRegex(ValueError, "independent goal"):
            score(task, {"text": task.expected})

    def test_semantic_mask_prunes_doomed_partial_commands(self):
        mask = SemanticGrammar()
        self.assertTrue(mask.status("MOVE N ")[0])
        self.assertFalse(mask.status("MOVE N 3")[0])
        self.assertFalse(mask.status("PICK ")[0])
        self.assertEqual(set(mask.allowed_characters("MOVE N ")), {"1", "2"})
        self.assertEqual(SemanticGrammar(movement_budget=0).allowed_characters(""), ("S",))

    def test_semantic_mask_tracks_inventory_and_movement(self):
        mask = SemanticGrammar(State(items=(("a", 2, 2),)))
        self.assertTrue(mask.accepts("PICK a;MOVE E 1;DROP a;STOP"))
        self.assertFalse(mask.status("PICK a;PICK a;")[0])
        self.assertFalse(mask.status("MOVE E 2;MOVE W 2;MOVE E 2;MOVE W ")[0])


class DecoderTests(unittest.TestCase):
    def test_mask_rejects_illegal_direction(self):
        policy = ReplayPolicy(["MOVE X 1;STOP", "MOVE E 1;STOP", "STOP"])
        raw = generate(policy, False)
        masked = generate(policy, True)
        self.assertEqual(raw["text"], "MOVE X 1;STOP")
        self.assertEqual(masked["text"], "MOVE E 1;STOP")
        self.assertGreater(masked["mask_interventions"], 0)

    def test_token_budget_is_not_silently_success(self):
        result = generate(ReplayPolicy(["MOVE N 1;STOP"]), True, token_budget=2)
        self.assertTrue(result["budget_exhausted"])
        self.assertFalse(result["eos"])
        self.assertFalse(Grammar().accepts(result["text"]))
        for budget in (0, 257, True):
            with self.assertRaises(ValueError):
                generate(ReplayPolicy(["STOP"]), True, budget)

    def test_syntax_and_semantics_separate_in_report(self):
        report = evaluate()
        masked = report["approaches"]["grammar_masked_replay"]["metrics"]
        raw = report["approaches"]["unconstrained_replay"]["metrics"]
        self.assertEqual(masked["syntax_valid"], 1)
        self.assertLess(masked["semantic_valid"], masked["syntax_valid"])
        self.assertLess(raw["syntax_valid"], masked["syntax_valid"])
        semantic = report["approaches"]["semantic_masked_replay"]["metrics"]
        self.assertEqual(semantic["semantic_valid"], 1)
        self.assertLess(semantic["goal_success"], semantic["semantic_valid"])

    def test_valid_but_wrong_goal(self):
        task = tasks()[0]
        row = score(task, {"text": "MOVE E 1;STOP"})
        self.assertTrue(row["syntax_valid"])
        self.assertTrue(row["semantic_valid"])
        self.assertFalse(row["goal_success"])

    def test_same_final_state_cannot_hide_skipped_item_actions(self):
        task = tasks()[2]
        row = score(task, {"text": "MOVE S 1;STOP"})
        self.assertTrue(row["goal_success"])
        self.assertFalse(row["instruction_success"])

    def test_equivalent_split_moves_preserve_instruction_success(self):
        task = tasks()[4]
        row = score(task, {"text": "MOVE N 1;MOVE N 1;PICK a;STOP"})
        self.assertTrue(row["instruction_success"])

    def test_deterministic_replay(self):
        self.assertEqual(evaluate(), evaluate())


class FakeTokenizer:
    eos_token_id = 0
    all_special_ids = [0, 1]
    pieces = ["<EOS>", "<BOS>", "MOVE ", "E 1;", "STOP", "X", "N", " ", "1;STOP"]

    def __len__(self):
        return len(self.pieces)

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[index] for index in ids)


class TokenizerAdapterTests(unittest.TestCase):
    def test_context_uses_all_real_limits(self):
        self.assertEqual(
            context_limit(
                SimpleNamespace(max_position_embeddings=1024, n_positions=512), SimpleNamespace(model_max_length=256)
            ),
            256,
        )
        self.assertEqual(context_limit(SimpleNamespace(n_positions=512), SimpleNamespace(model_max_length=10**30)), 512)

    def test_multi_character_token_crosses_terminals(self):
        tokenizer = FakeTokenizer()
        self.assertIn(2, token_allowlist(tokenizer, [], Grammar()))
        self.assertIn(3, token_allowlist(tokenizer, [2], Grammar()))
        self.assertIn(8, token_allowlist(tokenizer, [2, 6, 7], Grammar()))
        self.assertNotIn(0, token_allowlist(tokenizer, [], Grammar()))
        self.assertNotIn(5, token_allowlist(tokenizer, [2], Grammar()))

    def test_eos_only_after_complete_program(self):
        self.assertEqual(token_allowlist(FakeTokenizer(), [2, 3, 4], Grammar()), [0])

    def test_no_legal_continuation_is_an_error(self):
        tokenizer = FakeTokenizer()
        tokenizer.pieces = ["<EOS>", "<BOS>", "Z"]
        with self.assertRaises(SyntaxContractError):
            token_allowlist(tokenizer, [], Grammar())

    def test_sparse_token_ids_and_model_vocabulary_bounds(self):
        class Sparse:
            eos_token_id = 0
            all_special_ids = [0, 1]

            def __len__(self):
                return 3

            def get_vocab(self):
                return {"<eos>": 0, "<bos>": 1, "STOP": 9}

            def decode(self, ids, **_kwargs):
                return "".join({0: "<eos>", 1: "<bos>", 9: "STOP"}[index] for index in ids)

        self.assertEqual(token_allowlist(Sparse(), [], Grammar(), 10), [9])
        self.assertEqual(token_allowlist(Sparse(), [9], Grammar(), 10), [0])
        with self.assertRaises(SyntaxContractError):
            token_allowlist(Sparse(), [], Grammar(), 9)
        with self.assertRaises(ValueError):
            token_allowlist(Sparse(), [], Grammar(), 0)


if __name__ == "__main__":
    unittest.main()
