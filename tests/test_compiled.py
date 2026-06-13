import itertools
import math
import unittest

from constrained_generation.compiled import (
    CompiledConstraint,
    ReplayDistribution,
    TokenTrieMask,
    Vocabulary,
    beam_search,
    character_vocabulary,
    evaluate_beams,
    greedy_search,
    tokenizer_backend,
)
from constrained_generation.core import COMMANDS, Grammar, SemanticGrammar, State, execute


class CompiledStateTests(unittest.TestCase):
    def test_reference_equivalence_on_all_two_command_prefixes(self):
        programs = ["STOP", *(command + "STOP" for command in COMMANDS)]
        programs += [first + second + "STOP" for first, second in itertools.product(COMMANDS, repeat=2)]
        prefixes = {program[:index] for program in programs for index in range(len(program) + 1)}
        for semantic, initial, budget in (
            (False, State(), 6),
            (True, State(items=(("a", 2, 2),)), 6),
            (True, State(0, 0, ("b",)), 2),
            (True, State(), 0),
        ):
            reference = SemanticGrammar(initial, budget) if semantic else Grammar()
            compiler = CompiledConstraint(initial, semantic, budget)
            for prefix in prefixes:
                state = compiler.state_for_prefix(prefix)
                with self.subTest(semantic=semantic, prefix=prefix, budget=budget):
                    self.assertEqual((state is not None, bool(state and state.complete)), reference.status(prefix))
            for prefix in ("MOVE X", "STOPextra", "PICK z;", "MOVE N 0", "\n", "PICK a;" * 9):
                state = compiler.state_for_prefix(prefix)
                self.assertEqual((state is not None, bool(state and state.complete)), reference.status(prefix))

    def test_incremental_semantic_state_matches_independent_interpreter(self):
        initial = State(items=(("a", 2, 2),))
        compiler = CompiledConstraint(initial, semantic=True)
        state = compiler.initial_state
        for piece in ("PICK ", "a;MOVE E", " 2;DROP a;", "STOP"):
            state = compiler.advance(state, piece)
            self.assertIsNotNone(state)
        self.assertEqual(state.world, execute("PICK a;MOVE E 2;DROP a;STOP", initial))
        self.assertEqual(state.movement_left, 4)
        self.assertTrue(state.complete)
        self.assertIsNone(compiler.advance(state, "S"))

    def test_all_grid_movement_and_budget_boundaries_match_interpreter(self):
        for x, y, remaining in itertools.product(range(5), range(5), range(7)):
            initial = State(x, y)
            compiler = CompiledConstraint(initial, semantic=True, movement_budget=remaining)
            for command in COMMANDS[:12]:
                try:
                    expected = execute(command + "STOP", initial, remaining)
                except ValueError:
                    expected = None
                state = compiler.state_for_prefix(command + "STOP")
                self.assertEqual(state.world if state else None, expected)

    def test_item_ownership_primitives_match_interpreter(self):
        for x, y, item, placement in itertools.product((0, 2, 4), (0, 2, 4), "abc", range(4)):
            initial = State(
                x,
                y,
                (item,) if placement == 3 else (),
                ((item, x, y),) if placement == 1 else ((item, (x + 1) % 5, y),) if placement == 2 else (),
            )
            compiler = CompiledConstraint(initial, semantic=True)
            for command in (f"PICK {item};", f"DROP {item};"):
                try:
                    expected = execute(command + "STOP", initial)
                except ValueError:
                    expected = None
                state = compiler.state_for_prefix(command + "STOP")
                self.assertEqual(state.world if state else None, expected)

    def test_command_budget_and_viability_are_stateful(self):
        compiler = CompiledConstraint(max_commands=1)
        state = compiler.state_for_prefix("MOVE E 1;")
        self.assertIsNotNone(compiler.advance(state, "STOP"))
        self.assertIsNone(compiler.advance(state, "MOVE"))
        semantic = CompiledConstraint(semantic=True)
        self.assertIsNone(semantic.state_for_prefix("MOVE N 3"))
        self.assertIsNone(semantic.state_for_prefix("PICK a"))
        self.assertIsNotNone(CompiledConstraint().state_for_prefix("MOVE N 3"))

    def test_token_trie_matches_direct_token_transition_and_caches(self):
        initial = State(items=(("a", 3, 2),))
        vocabulary = Vocabulary({1: "MOVE ", 2: "MOVE E 1;", 3: "MOVE E 1;PICK a;", 4: "STOP", 8: "MOVE E 1;", 9: "X"})
        compiler = CompiledConstraint(initial, semantic=True)
        mask = TokenTrieMask(compiler, vocabulary, cache_size=2)
        states = [compiler.initial_state, compiler.state_for_prefix("MOVE E 1;"), compiler.state_for_prefix("STOP")]
        for state in states:
            expected = (
                {0: state}
                if state.complete
                else {
                    token: advanced
                    for token, piece in vocabulary.pieces.items()
                    if (advanced := compiler.advance(state, piece)) is not None
                }
            )
            self.assertEqual(mask.allowed((), state), expected)
            self.assertEqual(mask.allowed((), state), expected)
        self.assertGreater(mask.cache_hits, 0)
        self.assertLessEqual(len(mask._masks), 2)
        result = mask.allowed((), states[-1])
        result.clear()
        self.assertEqual(mask.allowed((), states[-1]), {0: states[-1]})

    def test_vocabulary_is_immutable_and_invalid_budgets_reject(self):
        pieces = {1: "STOP"}
        vocabulary = Vocabulary(pieces)
        pieces[1] = "INVALID"
        self.assertEqual(vocabulary.decode((1, 0)), "STOP")
        for arguments in ({"max_commands": True}, {"movement_budget": -1}, {"cache_size": 0}):
            with self.assertRaises(ValueError):
                CompiledConstraint(**arguments)


class BeamSearchTests(unittest.TestCase):
    def test_beam_preserves_original_probability_and_eos_score(self):
        vocabulary = Vocabulary({1: "STOP", 2: "MOVE E 1;STOP", 3: "MOVE W 1;STOP"})

        def policy(ids):
            probabilities = (
                {0: 0.05, 1: 0.45, 2: 0.35, 3: 0.15}
                if not ids
                else {
                    0: {1: 0.1, 2: 0.9, 3: 0.8}[ids[-1]],
                    1: (1 - {1: 0.1, 2: 0.9, 3: 0.8}[ids[-1]]) / 3,
                    2: (1 - {1: 0.1, 2: 0.9, 3: 0.8}[ids[-1]]) / 3,
                    3: (1 - {1: 0.1, 2: 0.9, 3: 0.8}[ids[-1]]) / 3,
                }
            )
            return {token: math.log(value) for token, value in probabilities.items()}

        beam = beam_search(policy, TokenTrieMask(CompiledConstraint(), vocabulary), 3, 2)
        greedy = greedy_search(policy, TokenTrieMask(CompiledConstraint(), vocabulary), 2, True)
        self.assertEqual(beam["text"], "MOVE E 1;STOP")
        self.assertEqual(greedy["text"], "STOP")
        self.assertAlmostEqual(beam["log_probability"], math.log(0.35 * 0.9))
        self.assertAlmostEqual(sum(beam["token_log_probs"]), beam["log_probability"])
        self.assertEqual(beam["token_ids"], [2, 0])

    def test_wide_beam_matches_independent_exhaustive_enumeration(self):
        vocabulary = Vocabulary({1: "STOP", 2: "MOVE E 1;", 3: "MOVE W 1;"})

        def policy(ids):
            raw = {token: 1 + ((sum(ids) + 1) * (token + 3)) % 7 for token in (0, 1, 2, 3)}
            total = sum(raw.values())
            return {token: math.log(weight / total) for token, weight in raw.items()}

        finished = []

        def enumerate_paths(ids=(), text="", log_probability=0.0):
            if len(ids) == 4:
                return
            for token, logp in policy(ids).items():
                if token == 0:
                    if Grammar().accepts(text):
                        finished.append((log_probability + logp, ids + (token,), text))
                else:
                    candidate = text + vocabulary.pieces[token]
                    if Grammar().status(candidate)[0]:
                        enumerate_paths(ids + (token,), candidate, log_probability + logp)

        enumerate_paths()
        expected = min(finished, key=lambda row: (-row[0], row[1]))
        result = beam_search(policy, TokenTrieMask(CompiledConstraint(), vocabulary), 64, 4)
        self.assertEqual(result["token_ids"], list(expected[1]))
        self.assertAlmostEqual(result["log_probability"], expected[0])

    def test_budget_exhaustion_is_explicit_and_score_is_auditable(self):
        vocabulary = character_vocabulary(["MOVE E 1;STOP"])
        policy = ReplayDistribution(["MOVE E 1;STOP"], vocabulary)
        result = beam_search(policy, TokenTrieMask(CompiledConstraint(), vocabulary), 2, 2)
        self.assertTrue(result["search_budget_exhausted"])
        self.assertFalse(result["eos"])
        self.assertAlmostEqual(result["log_probability"], sum(result["token_log_probs"]))
        result = beam_search(policy, TokenTrieMask(CompiledConstraint(), vocabulary), 2, 128, candidate_budget=1)
        self.assertEqual(result["stop_reason"], "candidate_budget")
        self.assertLessEqual(result["expanded_candidates"], 1)

    def test_invalid_distribution_and_bounds_fail(self):
        backend = TokenTrieMask(CompiledConstraint(), Vocabulary({1: "STOP"}))
        for scores in ({0: 0.0, 1: 0.0}, {0: float("nan")}, {0: 1.0}):
            with self.assertRaises(ValueError):
                beam_search(lambda _: scores, backend)
        for arguments in ({"beam_width": 0}, {"beam_width": True}, {"token_budget": 0}, {"candidate_budget": 0}):
            with self.assertRaises(ValueError):
                beam_search(lambda _: {0: 0.0}, backend, **arguments)

    def test_replay_comparison_is_reproducible_and_semantically_honest(self):
        report = evaluate_beams(count=4, beam_width=3, token_budget=64)
        self.assertEqual(report, evaluate_beams(count=4, beam_width=3, token_budget=64))
        self.assertFalse(report["learned_model_evaluated"])
        self.assertEqual(report["approaches"]["semantic_beam"]["metrics"]["semantic_valid"], 1)


class ContextTokenizer:
    eos_token_id = 0
    all_special_ids = [0]

    def get_vocab(self):
        return {"<eos>": 0, "MOVE": 1, "E": 2, "1;STOP": 3, "STOP": 4}

    def decode(self, ids, **_kwargs):
        return " ".join({0: "<eos>", 1: "MOVE", 2: "E", 3: "1;STOP", 4: "STOP"}[token] for token in ids)


class FallbackTests(unittest.TestCase):
    def test_context_dependent_decoder_explicitly_falls_back(self):
        backend = tokenizer_backend(ContextTokenizer(), 5)
        self.assertEqual(backend.stats()["backend"], "slow_reference_tokenizer")
        self.assertEqual(set(backend.allowed((), None)), {1, 4})
        self.assertEqual(set(backend.allowed((1,), None)), {2})
        self.assertEqual(set(backend.allowed((1, 2), None)), {3})
        self.assertEqual(backend.decode((1, 2, 3, 0)), "MOVE E 1;STOP")


if __name__ == "__main__":
    unittest.main()
