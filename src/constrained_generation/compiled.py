"""Compiled incremental constraints, token tries, and likelihood-scored beam search.

No model packages are imported here. The slow recognizers in core.py remain
independent reference implementations for equivalence checks.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .core import (
    ALPHABET,
    COMMANDS,
    DIRECTIONS,
    EOS,
    MAX_COMMANDS,
    MAX_TEXT,
    Grammar,
    ReplayPolicy,
    SemanticGrammar,
    State,
    SyntaxContractError,
    execute,
    score,
    summarize,
    tasks,
    token_allowlist,
)


@dataclass(frozen=True)
class MachineState:
    node: int = 0
    commands: int = 0
    complete: bool = False
    world: State = State()
    movement_left: int = 6


class CompiledConstraint:
    """Command trie × command counter × optional finite world state."""

    def __init__(
        self,
        initial: State = State(),
        semantic: bool = False,
        movement_budget: int = 6,
        max_commands: int = MAX_COMMANDS,
        cache_size: int = 4096,
    ):
        execute("STOP", initial, movement_budget)
        if type(max_commands) is not int or not 0 <= max_commands <= MAX_COMMANDS:
            raise ValueError("max_commands must be an integer in 0..8")
        if type(cache_size) is not int or not 1 <= cache_size <= 100_000:
            raise ValueError("cache_size must be in 1..100000")
        self.semantic, self.max_commands, self.cache_size = semantic, max_commands, cache_size
        self.initial_state = MachineState(world=initial, movement_left=movement_budget)
        self.children: list[dict[str, int]] = [{}]
        self.terminals: dict[int, str] = {}
        self.descendants: list[set[str]] = [set()]
        for command in (*COMMANDS, "STOP"):
            node = 0
            self.descendants[node].add(command)
            for character in command:
                if character not in self.children[node]:
                    self.children[node][character] = len(self.children)
                    self.children.append({})
                    self.descendants.append(set())
                node = self.children[node][character]
                self.descendants[node].add(command)
            self.terminals[node] = command
        self._viable: OrderedDict[tuple, frozenset[str]] = OrderedDict()
        self.transition_calls = 0

    def _apply(self, world: State, remaining: int, command: str) -> tuple[State, int] | None:
        """Independent incremental transition; no prefix replay or interpreter call."""
        if command == "STOP" or not self.semantic:
            return world, remaining
        parts = command[:-1].split(" ")
        x, y = world.x, world.y
        inventory, items = set(world.inventory), {name: (ix, iy) for name, ix, iy in world.items}
        if parts[0] == "MOVE":
            distance = int(parts[2])
            dx, dy = DIRECTIONS[parts[1]]
            x, y = x + dx * distance, y + dy * distance
            if distance > remaining or not (0 <= x < 5 and 0 <= y < 5):
                return None
            remaining -= distance
        elif parts[0] == "PICK":
            item = parts[1]
            if items.get(item) != (x, y):
                return None
            inventory.add(item)
            del items[item]
        else:
            item = parts[1]
            if item not in inventory:
                return None
            inventory.remove(item)
            items[item] = (x, y)
        result = State(x, y, tuple(sorted(inventory)), tuple(sorted((name, *point) for name, point in items.items())))
        return result, remaining

    def viable_commands(self, state: MachineState) -> frozenset[str]:
        key = (state.commands, state.world, state.movement_left)
        if key in self._viable:
            self._viable.move_to_end(key)
            return self._viable[key]
        allowed = {"STOP"}
        if state.commands < self.max_commands:
            allowed.update(
                command for command in COMMANDS if self._apply(state.world, state.movement_left, command) is not None
            )
        result = frozenset(allowed)
        self._viable[key] = result
        if len(self._viable) > self.cache_size:
            self._viable.popitem(last=False)
        return result

    def advance(self, state: MachineState, text: str) -> MachineState | None:
        if not isinstance(text, str):
            raise ValueError("Token text must be a string")
        current = state
        for character in text:
            self.transition_calls += 1
            if current.complete:
                return None
            node = self.children[current.node].get(character)
            if node is None or not (self.descendants[node] & self.viable_commands(current)):
                return None
            terminal = self.terminals.get(node)
            if terminal == "STOP":
                current = MachineState(node, current.commands, True, current.world, current.movement_left)
            elif terminal is not None:
                applied = self._apply(current.world, current.movement_left, terminal)
                if applied is None:
                    return None
                world, remaining = applied
                current = MachineState(0, current.commands + 1, False, world, remaining)
            else:
                current = MachineState(node, current.commands, False, current.world, current.movement_left)
        return current

    def state_for_prefix(self, prefix: str) -> MachineState | None:
        if not isinstance(prefix, str) or len(prefix) > MAX_TEXT:
            return None
        return self.advance(self.initial_state, prefix)


@dataclass(frozen=True)
class Vocabulary:
    pieces: Mapping[int, str]
    eos_id: int = 0

    def __post_init__(self):
        copied = dict(self.pieces)
        if type(self.eos_id) is not int or self.eos_id < 0 or self.eos_id in copied:
            raise ValueError("EOS must be a distinct nonnegative token ID")
        if not copied or any(
            type(key) is not int or key < 0 or not isinstance(value, str) or not 1 <= len(value) <= MAX_TEXT
            for key, value in copied.items()
        ):
            raise ValueError("Vocabulary requires bounded nonempty pieces and integer IDs")
        object.__setattr__(self, "pieces", MappingProxyType(copied))

    def decode(self, ids: tuple[int, ...]) -> str:
        return "".join(self.pieces[token] for token in ids if token != self.eos_id)


class TokenTrieMask:
    """Traverse shared token-character prefixes once per machine state."""

    def __init__(self, compiler: CompiledConstraint, vocabulary: Vocabulary, cache_size: int = 4096):
        if type(cache_size) is not int or not 1 <= cache_size <= 100_000:
            raise ValueError("cache_size must be in 1..100000")
        self.compiler, self.vocabulary, self.cache_size = compiler, vocabulary, cache_size
        self.initial_state, self.eos_id = compiler.initial_state, vocabulary.eos_id
        self.nodes: list[dict[str, int]] = [{}]
        self.endings: dict[int, list[int]] = {}
        for token_id, piece in vocabulary.pieces.items():
            node = 0
            for character in piece:
                if character not in self.nodes[node]:
                    self.nodes[node][character] = len(self.nodes)
                    self.nodes.append({})
                node = self.nodes[node][character]
            self.endings.setdefault(node, []).append(token_id)
        self._masks: OrderedDict[MachineState, dict[int, MachineState]] = OrderedDict()
        self.cache_hits = self.cache_misses = self.trie_visits = 0
        self.backend_name = "compiled_token_trie"

    def allowed(self, _ids: tuple[int, ...], state: MachineState) -> dict[int, MachineState]:
        if state in self._masks:
            self.cache_hits += 1
            self._masks.move_to_end(state)
            return dict(self._masks[state])
        self.cache_misses += 1
        result = {}
        if state.complete:
            result[self.eos_id] = state
        else:
            stack = [(0, state)]
            while stack:
                node, machine = stack.pop()
                self.trie_visits += 1
                for token_id in self.endings.get(node, ()):
                    result[token_id] = machine
                for character, child in self.nodes[node].items():
                    advanced = self.compiler.advance(machine, character)
                    if advanced is not None:
                        stack.append((child, advanced))
        self._masks[state] = result
        if len(self._masks) > self.cache_size:
            self._masks.popitem(last=False)
        return dict(result)

    def decode(self, ids: tuple[int, ...]) -> str:
        return self.vocabulary.decode(ids)

    def stats(self) -> dict:
        return {
            "backend": self.backend_name,
            "mask_cache_hits": self.cache_hits,
            "mask_cache_misses": self.cache_misses,
            "token_trie_visits": self.trie_visits,
            "character_transitions": self.compiler.transition_calls,
        }


class ReferenceTokenizerMask:
    """Explicit correctness fallback for context-dependent/unknown decoders."""

    def __init__(self, tokenizer, grammar, vocab_size: int, reason: str):
        self.tokenizer, self.grammar, self.vocab_size = tokenizer, grammar, vocab_size
        self.initial_state = None
        self.eos_id = tokenizer.eos_token_id
        self.reason = reason
        self.cache: OrderedDict[tuple[int, ...], dict[int, None]] = OrderedDict()
        self.hits = self.misses = 0

    def allowed(self, ids, _state):
        if ids in self.cache:
            self.hits += 1
            self.cache.move_to_end(ids)
            return dict(self.cache[ids])
        self.misses += 1
        result = {token: None for token in token_allowlist(self.tokenizer, list(ids), self.grammar, self.vocab_size)}
        self.cache[ids] = result
        if len(self.cache) > 4096:
            self.cache.popitem(last=False)
        return dict(result)

    def decode(self, ids):
        content = ids[:-1] if ids and ids[-1] == self.eos_id else ids
        return self.tokenizer.decode(list(content), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def stats(self):
        return {
            "backend": "slow_reference_tokenizer",
            "fallback_reason": self.reason,
            "mask_cache_hits": self.hits,
            "mask_cache_misses": self.misses,
        }


class ByteLevelTokenizerMask(TokenTrieMask):
    """Compiled constraints with full raw decoding for unconstrained baselines."""

    def __init__(self, compiler, vocabulary, tokenizer):
        super().__init__(compiler, vocabulary)
        self.tokenizer = tokenizer
        self.known_ids = frozenset(tokenizer.get_vocab().values())

    def decode(self, ids):
        content = ids[:-1] if ids and ids[-1] == self.eos_id else ids
        if any(token not in self.known_ids for token in content):
            return "<UNDECODABLE_TOKEN_IDS:" + ",".join(map(str, content)) + ">"
        return self.tokenizer.decode(list(content), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def tokenizer_backend(
    tokenizer,
    vocab_size: int,
    initial: State = State(),
    semantic: bool = False,
    movement_budget: int = 6,
    force_reference: bool = False,
):
    """Certify a ByteLevel decoder; all others explicitly use the slow path.

    Legal DSL text is ASCII. ByteLevel ASCII token pieces concatenate exactly;
    partial non-ASCII bytes cannot form a legal ASCII continuation.
    """
    if (
        type(vocab_size) is not int
        or vocab_size < 1
        or type(tokenizer.eos_token_id) is not int
        or not 0 <= tokenizer.eos_token_id < vocab_size
    ):
        raise ValueError("EOS must fit a positive model output vocabulary")
    reference = SemanticGrammar(initial, movement_budget) if semantic else Grammar()
    reason = "reference explicitly requested" if force_reference else "decoder is not a standalone ByteLevel decoder"
    try:
        decoder = json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())
    except (AttributeError, TypeError, ValueError):
        decoder = {}
    if not force_reference and decoder.get("type") == "ByteLevel":
        special = set(tokenizer.all_special_ids)
        pieces = {}
        for token_id in tokenizer.get_vocab().values():
            if type(token_id) is not int or token_id < 0:
                raise ValueError("Invalid tokenizer token ID")
            if token_id >= vocab_size or token_id in special:
                continue
            text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if text and all(character in ALPHABET for character in text):
                pieces[token_id] = text
        if pieces:
            vocabulary = Vocabulary(pieces, tokenizer.eos_token_id)
            return ByteLevelTokenizerMask(CompiledConstraint(initial, semantic, movement_budget), vocabulary, tokenizer)
        reason = "ByteLevel tokenizer has no legal standalone DSL token pieces"
    return ReferenceTokenizerMask(tokenizer, reference, vocab_size, reason)


@dataclass(frozen=True)
class Hypothesis:
    ids: tuple[int, ...]
    state: Any
    log_probability: float = 0.0
    token_log_probs: tuple[float, ...] = ()
    ended: bool = False


def _validated_scores(scores: Mapping[int, float]) -> dict[int, float]:
    result = dict(scores)
    if not result or any(
        type(key) is not int or key < 0 or not isinstance(value, (int, float)) or math.isnan(value) or value > 0
        for key, value in result.items()
    ):
        raise ValueError("Policy must return nonpositive log probabilities keyed by token ID")
    mass = sum(math.exp(value) for value in result.values())
    if not math.isclose(mass, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Policy scores must normalize over the original, unmasked vocabulary")
    return result


def beam_search(
    policy: Callable[[tuple[int, ...]], Mapping[int, float]],
    backend,
    beam_width: int = 4,
    token_budget: int = 128,
    constrained: bool = True,
    candidate_budget: int = 200_000,
) -> dict[str, Any]:
    """Bounded MAP beam search using original-model sequence log probabilities.

    Constraint masking never renormalizes scores. EOS contributes its own
    log probability. No length penalty or completion bonus is hidden here.
    """
    if type(beam_width) is not int or not 1 <= beam_width <= 64:
        raise ValueError("beam_width must be in 1..64")
    if type(token_budget) is not int or not 1 <= token_budget <= MAX_TEXT:
        raise ValueError("token_budget must be in 1..256")
    if type(candidate_budget) is not int or not 1 <= candidate_budget <= 1_000_000:
        raise ValueError("candidate_budget must be in 1..1000000")
    active = [Hypothesis((), backend.initial_state)]
    finished: list[Hypothesis] = []
    expanded = pruned = calls = 0
    reason = "token_budget"
    ranking = lambda hypothesis: (-hypothesis.log_probability, hypothesis.ids)
    for _ in range(token_budget):
        candidates = []
        exhausted = False
        for hypothesis in active:
            scores = _validated_scores(policy(hypothesis.ids))
            calls += 1
            legal = (
                backend.allowed(hypothesis.ids, hypothesis.state) if constrained else {token: None for token in scores}
            )
            for token in sorted(set(scores) & set(legal)):
                if not math.isfinite(scores[token]):
                    continue
                if expanded >= candidate_budget:
                    exhausted = True
                    break
                expanded += 1
                candidate = Hypothesis(
                    hypothesis.ids + (token,),
                    legal[token],
                    hypothesis.log_probability + scores[token],
                    hypothesis.token_log_probs + (scores[token],),
                    token == backend.eos_id,
                )
                if candidate.ended:
                    finished.append(candidate)
                else:
                    candidates.append(candidate)
            if exhausted:
                break
        if exhausted:
            reason = "candidate_budget"
            # Preserve pre-expansion hypotheses when a frontier is only partly
            # explored; returning an arbitrary half-frontier would hide loss.
            break
        candidates.sort(key=ranking)
        pruned += max(0, len(candidates) - beam_width)
        active = candidates[:beam_width]
        finished.sort(key=ranking)
        finished = finished[:beam_width]
        if not active:
            reason = "no_active_hypotheses"
            break
        if finished and finished[0].log_probability >= active[0].log_probability:
            reason = "completed_score_bound"
            break
    pool = finished if finished else active
    if not pool:
        raise SyntaxContractError("No finite-probability continuation survived")
    best = min(pool, key=ranking)
    return {
        "text": backend.decode(best.ids),
        "tokens": len(best.ids),
        "token_ids": list(best.ids),
        "eos": best.ended,
        "budget_exhausted": not best.ended,
        "search_budget_exhausted": reason in {"token_budget", "candidate_budget"},
        "stop_reason": reason,
        "log_probability": best.log_probability,
        "token_log_probs": list(best.token_log_probs),
        "beam_width": beam_width,
        "expanded_candidates": expanded,
        "pruned_active_candidates": pruned,
        "policy_calls": calls,
        "mask_interventions": None,
        "mask_stats": backend.stats(),
        "score_contract": "sum of unmasked conditional log probabilities, including EOS",
    }


def character_vocabulary(candidates: list[str]) -> Vocabulary:
    alphabet = sorted(set("".join(candidates)) | set(ALPHABET))
    return Vocabulary({index + 1: character for index, character in enumerate(alphabet)})


def greedy_search(policy, backend, token_budget: int = 128, constrained: bool = False) -> dict:
    """Actual greedy decoding; unlike beam width one, never retains EOS branches."""
    if type(token_budget) is not int or not 1 <= token_budget <= MAX_TEXT:
        raise ValueError("token_budget must be in 1..256")
    ids, token_scores, state = (), [], backend.initial_state
    ended = False
    for _ in range(token_budget):
        scores = _validated_scores(policy(ids))
        allowed = backend.allowed(ids, state) if constrained else {token: None for token in scores}
        options = [token for token in set(scores) & set(allowed) if math.isfinite(scores[token])]
        if not options:
            raise SyntaxContractError("No finite-probability continuation survived")
        token = min(options, key=lambda candidate: (-scores[candidate], candidate))
        ids += (token,)
        token_scores.append(scores[token])
        state = allowed[token]
        if token == backend.eos_id:
            ended = True
            break
    return {
        "text": backend.decode(ids),
        "tokens": len(ids),
        "token_ids": list(ids),
        "eos": ended,
        "budget_exhausted": not ended,
        "search_budget_exhausted": not ended,
        "stop_reason": "eos" if ended else "token_budget",
        "log_probability": sum(token_scores),
        "token_log_probs": token_scores,
        "beam_width": 1,
        "mask_interventions": None,
        "mask_stats": backend.stats(),
        "score_contract": "sum of unmasked conditional log probabilities, including EOS",
    }


class ReplayDistribution:
    """A fully normalized scored extension of the existing ranked replay policy."""

    def __init__(self, candidates: list[str], vocabulary: Vocabulary, sharpness: float = 1.0):
        if not math.isfinite(sharpness) or not 0 < sharpness <= 10:
            raise ValueError("sharpness must be finite in (0,10]")
        self.policy, self.vocabulary, self.sharpness = ReplayPolicy(candidates), vocabulary, sharpness
        self.reverse = {piece: token for token, piece in vocabulary.pieces.items()}

    def __call__(self, ids: tuple[int, ...]) -> dict[int, float]:
        ranked = [
            self.vocabulary.eos_id if token == EOS else self.reverse[token]
            for token in self.policy.rank(self.vocabulary.decode(ids))
        ]
        ranked.extend(token for token in self.vocabulary.pieces if token not in ranked)
        raw = {token: -self.sharpness * rank for rank, token in enumerate(ranked)}
        normalizer = math.log(sum(math.exp(value) for value in raw.values()))
        return {token: value - normalizer for token, value in raw.items()}


def evaluate_beams(count: int = 20, beam_width: int = 4, token_budget: int = 128) -> dict:
    dataset = tasks()
    if type(count) is not int or not 1 <= count <= len(dataset):
        raise ValueError("count must be in 1..20")
    approaches = {}
    for name, width, constrained, semantic in (
        ("greedy_unconstrained", 1, False, False),
        ("grammar_beam", beam_width, True, False),
        ("semantic_beam", beam_width, True, True),
    ):
        rows = []
        for task in dataset[:count]:
            candidates = [task.preferred, task.expected, "STOP"]
            vocabulary = character_vocabulary(candidates)
            policy = ReplayDistribution(candidates, vocabulary)
            backend = TokenTrieMask(CompiledConstraint(task.initial, semantic), vocabulary)
            generation = (
                greedy_search(policy, backend, token_budget, False)
                if name == "greedy_unconstrained"
                else beam_search(policy, backend, width, token_budget, constrained)
            )
            rows.append(score(task, generation))
        approaches[name] = summarize(rows)
    return {
        "experiment": "compiled/stateful beam search over normalized SYNTHETIC replay probabilities",
        "learned_model_evaluated": False,
        "count": count,
        "beam_width": beam_width,
        "token_budget": token_budget,
        "score_contract": "original-model likelihood, not renormalized after masking",
        "approaches": approaches,
    }
