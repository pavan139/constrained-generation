# Stateful decoding and beam results

The compiled decoder and scored search use the following components:

- Compiled command-trie state machine composed with incremental world and movement-budget state.
- Shared vocabulary-token trie traversal and bounded state-keyed mask/terminal caches.
- Actual greedy search and bounded beam search with audited original-model log probabilities, EOS costs, deterministic ties, completed hypotheses and score-bound termination.
- Explicit slow-reference fallback for context-dependent tokenizers.
- Actual optional local Hugging Face full-prefix scoring for greedy, grammar-beam and semantic-beam comparisons.

## Reproduce

From this standalone repository:

```bash
python -m pip install -e . --no-build-isolation
PYTHONPATH=src python -m unittest discover -s tests -v
python -m constrained_generation beam-evaluate --count 20 --beam-width 4 --token-budget 128 --output cases/beam-evaluation.json
```

Optional local-model execution:

```bash
python -m pip install -e '.[ml]'
RUN_ML_SMOKE=1 PYTHONPATH=src python -m unittest discover -s tests -v
python -m constrained_generation hf-beam-evaluate --model /absolute/path/to/local-checkpoint --count 20 --beam-width 4 --token-budget 128 --output artifacts/hf-beams.json
python -m constrained_generation hf-beam-evaluate --model /absolute/path/to/local-checkpoint --force-reference --count 20 --beam-width 4 --token-budget 128 --output artifacts/hf-reference-beams.json
```

No pretrained weights are downloaded. The optional test creates a tiny random local model; it verifies integration, not useful learned accuracy.

## Actual measured replay scope

The committed `cases/beam-evaluation.json` was generated with 20 independently labeled synthetic tasks, beam width four and 128 tokens. All approaches receive the same scored candidate policy; scores are normalized over the original vocabulary.

| Approach | Syntax valid | Executable | Final-state goal | Instruction success |
| --- | --- | --- | --- | --- |
| True unconstrained greedy | 11/20 | 3/20 | 3/20 | 3/20 |
| Compiled grammar beam | 20/20 | 20/20 | 2/20 | 2/20 |
| Compiled semantic beam | 20/20 | 20/20 | 2/20 | 2/20 |

Grammar beam expanded 875 candidates across 536 policy calls; semantic beam expanded 468 candidates across 327 calls. These are reproducible search counters, not GPU speedups or real-model quality. The poor goal score is expected evidence of the raw-likelihood objective's short-STOP preference. We preserve it rather than inventing a success claim or silently adding a length reward.

The complete optional-ML suite passed **41/41 tests** on CPU PyTorch 2.6.0 and Transformers 4.57.1. Default offline discovery executes 39 tests and explicitly skips two ML integrations. The finite-equivalence and independent exhaustive-search tests described in [automata-and-search.md](automata-and-search.md) passed. Existing reference tests remain intact. Actual tiny CPU HF beam smoke uses a locally trained ByteLevel tokenizer, a random one-layer GPT-2, one task, four generated tokens and beam width two. Both compiled mask equivalence and explicit reference fallback are checked.

## Remaining scope

There has been no meaningful pretrained-model evaluation, optimized KV-cache implementation, large-vocabulary throughput benchmark, GPU benchmark, arbitrary-grammar compiler or guarantee of global MAP under a narrow beam. The finite DSL, replay distribution and ByteLevel certification boundary are explicit. See [automata-and-search.md](automata-and-search.md) for complexity and failure behavior.
