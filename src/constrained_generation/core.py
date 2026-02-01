from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

MAX_COMMANDS = 8
MAX_TEXT = 256
EOS = "<EOS>"
DIRECTIONS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
COMMANDS = tuple(
    [f"MOVE {d} {n};" for d in DIRECTIONS for n in (1, 2, 3)]
    + [f"{verb} {item};" for verb in ("PICK", "DROP") for item in "abc"]
)
ALPHABET = tuple(sorted(set("".join(COMMANDS) + "STOP")))


class SyntaxContractError(ValueError):
    pass


class SemanticContractError(ValueError):
    pass


class Grammar:
    """Finite-state recognizer with a bounded command-count register.

    Canonical grammar: (MOVE [NESW] [123]; | PICK [abc]; | DROP [abc];){0,8}STOP.
    No whitespace beyond the literal spaces shown is accepted.
    """

    def status(self, text: str) -> tuple[bool, bool]:
        if not isinstance(text, str) or len(text) > MAX_TEXT:
            return False, False
        fragment = ""
        count = 0
        complete = False
        for char in text:
            if complete:
                return False, False
            fragment += char
            options = ("STOP",) if count >= MAX_COMMANDS else COMMANDS + ("STOP",)
            if not any(option.startswith(fragment) for option in options):
                return False, False
            if fragment == "STOP":
                complete = True
            elif fragment in COMMANDS:
                count += 1
                fragment = ""
        return True, complete

    def accepts(self, text: str) -> bool:
        return self.status(text) == (True, True)

    def allowed_characters(self, prefix: str) -> tuple[str, ...]:
        if not self.status(prefix)[0]:
            raise SyntaxContractError("Prefix is already outside the grammar")
        return tuple(char for char in ALPHABET if self.status(prefix + char)[0])


@dataclass(frozen=True)
class State:
    x: int = 2
    y: int = 2
    inventory: tuple[str, ...] = ()
    items: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class Task:
    id: str
    initial: State
    expected: str
    preferred: str
    instruction: str
    goal: State


def parse(text: str) -> list[tuple[str, ...]]:
    if not Grammar().accepts(text):
        raise SyntaxContractError("Incomplete or invalid DSL program")
    return [tuple(command.split(" ")) for command in text[:-4].split(";") if command]


def execute(text: str, initial: State = State(), movement_budget: int = 6) -> State:
    commands = parse(text)
    if not isinstance(initial, State):
        raise SemanticContractError("Expected a typed State record")
    if type(movement_budget) is not int or not 0 <= movement_budget <= 24:
        raise SemanticContractError("Movement budget must be an integer in 0..24")
    if type(initial.x) is not int or type(initial.y) is not int or not (0 <= initial.x < 5 and 0 <= initial.y < 5):
        raise SemanticContractError("Initial position must lie on the 5x5 grid")
    if (
        not isinstance(initial.inventory, tuple)
        or not isinstance(initial.items, tuple)
        or any(not isinstance(item, tuple) or len(item) != 3 for item in initial.items)
    ):
        raise SemanticContractError("State inventory and items must be typed tuples")
    names = list(initial.inventory) + [item[0] for item in initial.items]
    if any(not isinstance(name, str) or name not in {"a", "b", "c"} for name in names) or len(names) != len(set(names)):
        raise SemanticContractError("Initial item names must be unique a/b/c")
    if any(type(x) is not int or type(y) is not int or not (0 <= x < 5 and 0 <= y < 5) for _, x, y in initial.items):
        raise SemanticContractError("Invalid initial item position")
    x, y = initial.x, initial.y
    inventory = set(initial.inventory)
    items = {name: (ix, iy) for name, ix, iy in initial.items}
    distance = 0
    for command in commands:
        if command[0] == "MOVE":
            _, direction, raw_steps = command
            steps = int(raw_steps)
            distance += steps
            if distance > movement_budget:
                raise SemanticContractError("Total movement exceeds semantic budget")
            dx, dy = DIRECTIONS[direction]
            x, y = x + dx * steps, y + dy * steps
            if not (0 <= x < 5 and 0 <= y < 5):
                raise SemanticContractError("Robot moved outside 5x5 grid")
        elif command[0] == "PICK":
            item = command[1]
            if items.get(item) != (x, y):
                raise SemanticContractError("Requested item is not at robot position")
            inventory.add(item)
            del items[item]
        else:
            item = command[1]
            if item not in inventory:
                raise SemanticContractError("Cannot drop an unowned item")
            inventory.remove(item)
            items[item] = (x, y)
    return State(x, y, tuple(sorted(inventory)), tuple(sorted((name, *point) for name, point in items.items())))


class SemanticGrammar(Grammar):
    """Prefix mask for *executable* continuations, without consulting task goals.

    Look ahead to complete command terminals before allowing partial fragments.
    This prevents dead ends such as committing to an impossible MOVE N 3.
    """

    def __init__(self, initial: State = State(), movement_budget: int = 6):
        execute("STOP", initial, movement_budget)
        self.initial = initial
        self.movement_budget = movement_budget

    def status(self, text: str) -> tuple[bool, bool]:
        valid, complete = super().status(text)
        if not valid:
            return False, False
        cut = text.rfind(";") + 1
        head, fragment = text[:cut], text[cut:]
        try:
            execute(head + "STOP", self.initial, self.movement_budget)
        except (SyntaxContractError, SemanticContractError):
            return False, False
        if complete:
            return True, True
        count = head.count(";")
        options = ("STOP",) if count >= MAX_COMMANDS else COMMANDS + ("STOP",)
        for command in options:
            if not command.startswith(fragment):
                continue
            continuation = head + command + ("STOP" if command != "STOP" else "")
            try:
                execute(continuation, self.initial, self.movement_budget)
                return True, False
            except (SyntaxContractError, SemanticContractError):
                continue
        return False, False


class ReplayPolicy:
    """An ordered candidate-string trie, not a trained language model.

    Preferred malformed strings get higher rank than golden and STOP fallbacks.
    Both decoding modes see exactly the same next-character preferences.
    """

    def __init__(self, candidates: list[str]):
        if not candidates or any(not isinstance(value, str) or len(value) > MAX_TEXT for value in candidates):
            raise ValueError("Need nonempty bounded candidate strings")
        self.candidates = candidates

    def rank(self, prefix: str) -> list[str]:
        ranking: list[str] = []
        for candidate in self.candidates:
            if candidate.startswith(prefix):
                token = EOS if candidate == prefix else candidate[len(prefix)]
                if token not in ranking:
                    ranking.append(token)
        for token in (*ALPHABET, EOS):
            if token not in ranking:
                ranking.append(token)
        return ranking


def generate(
    policy: ReplayPolicy, constrained: bool, token_budget: int = 128, grammar: Grammar | None = None
) -> dict[str, Any]:
    if type(token_budget) is not int or not 1 <= token_budget <= MAX_TEXT:
        raise ValueError(f"Token budget must be 1..{MAX_TEXT}")
    prefix = ""
    grammar = grammar or Grammar()
    interventions = 0
    for step in range(token_budget):
        ranking = policy.rank(prefix)
        allowed = lambda token: (grammar.accepts(prefix) if token == EOS else grammar.status(prefix + token)[0])
        chosen = next((token for token in ranking if not constrained or allowed(token)), None)
        if chosen is None:
            raise SyntaxContractError("No legal continuation")
        interventions += chosen != ranking[0]
        if chosen == EOS:
            return {
                "text": prefix,
                "tokens": step + 1,
                "eos": True,
                "budget_exhausted": False,
                "mask_interventions": interventions,
            }
        prefix += chosen
    return {
        "text": prefix,
        "tokens": token_budget,
        "eos": False,
        "budget_exhausted": True,
        "mask_interventions": interventions,
    }


def tasks() -> list[Task]:
    result = []
    for index in range(18):
        direction = tuple(DIRECTIONS)[index % 4]
        distance = 1 + (index // 4) % 2
        dx, dy = DIRECTIONS[direction]
        kind = index % 3
        initial = State(items=(("a", 2 + dx * distance, 2 + dy * distance),) if kind else ())
        expected = (
            f"MOVE {direction} {distance};" + ("PICK a;" if kind else "") + ("DROP a;" if kind == 2 else "") + "STOP"
        )
        variant = index % 6
        if variant == 0:
            preferred = expected.replace(f"MOVE {direction}", "MOVE X", 1)
        elif variant == 1:
            preferred = expected.replace(";", " ", 1)
        elif variant == 2:
            opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}[direction]
            preferred = expected.replace(f"MOVE {direction}", f"MOVE {opposite}", 1)
        elif variant == 3:
            preferred = expected.replace(f" {distance};", " 3;", 1)
        elif variant == 4:
            preferred = expected + " explanation"
        else:
            preferred = expected
        instruction = (
            f"Move {distance} cells {direction}"
            + (", pick a" if kind else "")
            + (", then drop a" if kind == 2 else "")
            + "."
        )
        # Direct construction labels do not call the semantic interpreter.
        final_x, final_y = 2 + dx * distance, 2 + dy * distance
        goal = State(final_x, final_y, ("a",) if kind == 1 else (), (("a", final_x, final_y),) if kind == 2 else ())
        result.append(Task(f"synthetic-{index:02}", initial, expected, preferred, instruction, goal))
    result.append(
        Task(
            "movement-budget",
            State(),
            "STOP",
            "MOVE E 1;MOVE W 1;" * 4 + "STOP",
            "Remain at the starting position.",
            State(),
        )
    )
    result.append(Task("missing-item", State(), "STOP", "PICK b;STOP", "Do nothing; no item exists.", State()))
    return result


def score(task: Task, generation: dict[str, Any]) -> dict[str, Any]:
    if execute(task.expected, task.initial) != task.goal:
        raise ValueError(f"Golden program disagrees with independent goal for {task.id}")
    text = generation["text"]
    result = {
        **generation,
        "id": task.id,
        "instruction": task.instruction,
        "expected": task.expected,
        "syntax_valid": Grammar().accepts(text),
        "semantic_valid": False,
        "goal_success": False,
        "instruction_success": False,
    }
    try:
        state = execute(text, task.initial)
        result["semantic_valid"] = True
        result["goal_success"] = state == task.goal
        result["instruction_success"] = state == task.goal and canonical_actions(text) == canonical_actions(
            task.expected
        )
        result["final_state"] = asdict(state)
    except (SyntaxContractError, SemanticContractError) as exc:
        result["error"] = str(exc)
    return result


def canonical_actions(text: str) -> list[tuple[str, ...]]:
    """Action semantics up to splitting a MOVE into smaller MOVE commands."""
    actions = []
    for command in parse(text):
        if command[0] == "MOVE":
            actions.extend([("MOVE", command[1])] * int(command[2]))
        else:
            actions.append(command)
    return actions


def evaluate(token_budget: int = 128, count: int = 20) -> dict[str, Any]:
    dataset = tasks()
    if type(count) is not int or not 1 <= count <= len(dataset):
        raise ValueError(f"count must be 1..{len(dataset)}")
    approaches = {}
    for mode in ("unconstrained_replay", "grammar_masked_replay", "semantic_masked_replay"):
        rows = [
            score(
                task,
                generate(
                    ReplayPolicy([task.preferred, task.expected, "STOP"]),
                    mode != "unconstrained_replay",
                    token_budget,
                    SemanticGrammar(task.initial) if mode == "semantic_masked_replay" else Grammar(),
                ),
            )
            for task in dataset[:count]
        ]
        approaches[mode] = summarize(rows)
    return {
        "experiment": "deterministic synthetic candidate replay; NOT trained-model accuracy",
        "learned_model_evaluated": False,
        "dataset": "20 synthetic DSL tasks including syntax, semantics, and budget failures",
        "goal_oracle": "independently constructed final-state labels",
        "token_budget": token_budget,
        "approaches": approaches,
        "unrun_workflows": [
            "local HF unconstrained generation",
            "local HF grammar-masked generation",
            "local HF semantic-masked generation",
        ],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot score an empty run")
    return {
        "count": len(rows),
        "metrics": {
            key: sum(bool(row[key]) for row in rows) / len(rows)
            for key in ("syntax_valid", "semantic_valid", "goal_success", "instruction_success", "budget_exhausted")
        },
        "cases": rows,
    }


def token_allowlist(
    tokenizer: Any, generated_ids: list[int], grammar: Grammar, vocab_size: int | None = None
) -> list[int]:
    """Exhaustive tokenizer adapter, accounting for BPE tokens spanning grammar terminals.

    Decodes the entire generated suffix plus each candidate, not a token in
    isolation. This is deliberately a correctness-oriented baseline, not a fast
    production grammar compiler.
    """
    decode: Callable[[list[int]], str] = lambda ids: tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    current = decode(generated_ids)
    if not grammar.status(current)[0]:
        raise SyntaxContractError("Tokenizer output already violates grammar")
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise ValueError("Tokenizer must define an EOS token")
    if vocab_size is not None and (type(vocab_size) is not int or vocab_size < 1 or eos >= vocab_size):
        raise ValueError("EOS must exist in the model output vocabulary")
    if grammar.accepts(current):
        return [eos]
    special = set(tokenizer.all_special_ids)
    allowed = []
    # Added-token vocabularies can have non-contiguous IDs. len(tokenizer) is
    # not an upper bound on token IDs and can enumerate nonexistent tokens.
    token_ids = (
        sorted(set(tokenizer.get_vocab().values())) if hasattr(tokenizer, "get_vocab") else range(len(tokenizer))
    )
    for token_id in token_ids:
        if type(token_id) is not int or token_id < 0:
            raise ValueError("Tokenizer contains an invalid token ID")
        if vocab_size is not None and token_id >= vocab_size:
            continue
        if token_id in special:
            continue
        candidate = decode(generated_ids + [token_id])
        if candidate != current and candidate.startswith(current) and grammar.status(candidate)[0]:
            allowed.append(token_id)
    if not allowed:
        raise SyntaxContractError("Tokenizer has no legal continuation for this DSL prefix")
    return allowed
