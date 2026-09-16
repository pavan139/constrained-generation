# Review: tokenizer masks, semantic constraints and independent scoring

The review inspected finite-state recognition, parser/interpreter boundaries, replay ranking, EOS/budget accounting, tokenizer decoding, model configuration and the optional HF execution path.

## Reproduced defects and fixes

| Reproduction | Before | Correction and regression evidence |
| --- | --- | --- |
| A tokenizer with actual IDs `{0, 1, 9}` and length 3 | Enumerating `range(len(tokenizer))` attempted nonexistent ID 2 and raised `KeyError` | Enumerate real `get_vocab()` IDs, validate IDs and intersect with model output-vocabulary bounds. Regression tests require legal ID 9 and reject it when the output vocabulary is too small. |
| A checkpoint with `forced_eos_token_id=0`, `min_new_tokens=8`, and a one-token budget | Forced EOS overrode an allowlist containing only the STOP token; generated text was empty and syntactically invalid | Fresh greedy generation plus `use_model_defaults=False` where supported isolates the experiment. A real random-checkpoint regression now generates `STOP`; it correctly remains budget-exhausted because there is no token left for EOS. |
| Compare predicted and golden final states through the same interpreter only | Shared semantics errors could contaminate the oracle | Tasks carry directly constructed goal labels; inconsistent golden execution fails before predictions are scored. |
| Omit PICK/DROP actions without changing the final state | Final-state goal success alone credited the omission | A separate instruction metric compares canonical action traces, permitting split movement while rejecting skipped effects. |
| Non-State initial input or conflicting model/tokenizer context limits | Attribute error or an overly permissive context decision | Typed initial-state validation and minimum real context limits now fail explicitly. |

A fresh `GenerationConfig` alone was initially tested and proved insufficient on Transformers 4.57.1, which merges model-specific defaults into default-valued fields. The committed correction explicitly disables that merge when the installed API supports it. This behavior is covered by an actual checkpoint test, not only a fake tokenizer.

## Semantic prefix masking

`SemanticGrammar` adds state-aware lookahead to the finite-state grammar. It verifies completed actions and tests possible complete command terminals before admitting a partial command. Thus it prunes `MOVE N 3` from the center before committing to the invalid distance and rejects nonexistent item pickups. The mask never sees golden output or the task goal.

All three replay approaches receive the same candidate rankings. The new mode improves **executable output**, not guaranteed task correctness:

| Mode, 20 synthetic replay cases | Syntax valid | Semantically executable | Final-state goal | Instruction success |
| --- | --- | --- | --- | --- |
| Unconstrained replay | 11/20 | 3/20 | 3/20 | 3/20 |
| Grammar-only replay | 20/20 | 12/20 | 12/20 | 12/20 |
| New semantic-masked replay | 20/20 | 20/20 | 17/20 | 16/20 |

These are executed **synthetic candidate-replay** results, not LLM accuracy. The deliberately provided golden fallback and fixed ranking policy are part of the experiment. A finite budget can still terminate inside a prefix.

## Commands actually run

From this project directory, with optional ML dependencies already installed:

```bash
RUN_ML_SMOKE=1 PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m constrained_generation evaluate --output cases/replay-evaluation.json
PYTHONPATH=src python -m constrained_generation demo --output cases/demo.json
```

The suite passed **27/27 tests**, including the opt-in random-checkpoint integration regression. Default discovery runs 26 core tests and explicitly skips that optional test. Ruff checks and formatting checks passed. Actual measured replay results are committed in `cases/replay-evaluation.json`.

The integration test creates a random one-layer GPT-2 and a four-token local tokenizer, reproduces the old forced-EOS failure and checks grammar/semantic masking after the fix. It ran on CPU PyTorch 2.6.0 with Transformers 4.57.1; there were no pretrained downloads or paid calls. `cases/integration-smoke.json` records the separate executed reproduction. It is not model-quality evidence.

## Remaining limits

- Exhaustive vocabulary enumeration and semantic lookahead are correctness baselines, not optimized serving implementations.
- The adapter assumes the decoded ASCII suffix grows monotonically. Incompatible tokenizer schemes fail explicitly; broad tokenizer compatibility is unproven.
- State-aware legality does not encode the desired goal or guarantee successful instruction following.
- The grammar is deliberately small, canonical and bounded; arbitrary programming languages and application schemas are unsupported.
- No pretrained-model task accuracy, GPU benchmark or model-serving latency claim has been established.
