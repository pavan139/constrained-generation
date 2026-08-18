# Compiled constraints and likelihood-scored search

## State and compilation

The new `compiled.py` engine compiles the finite command vocabulary into a character trie once. A decoding state is the immutable tuple `(trie_node, command_count, terminated, world_state, movement_left)`. It does not contain the full generated program. Each completed command returns the grammar node to the trie root; completing `STOP` enters a terminal state.

Semantic mode composes the syntax machine with independent incremental transition functions for grid movement, item ownership and movement budget. Completed commands update copied finite state. Partial commands are admitted only if at least one descendant terminal is executable from the current state. Thus `MOVE N 3` is pruned before its semicolon when starting at the center. Neither compiler nor mask receives a requested goal, expected program or evaluation label.

The existing `Grammar`, `SemanticGrammar`, and `execute` remain separate slow references. The compiled transition code does not call those recognizers or replay the full program during decoding. Constructor validation uses `execute("STOP", ...)` once to validate the initial world.

Viable terminal sets are cached by `(command_count, world_state, movement_left)` using a bounded LRU. Two distinct token histories that reach the same finite state reuse these decisions.

## Token trie and mask cache

A second trie contains vocabulary token pieces. To compute a mask, traverse that trie and the compiled grammar together. A character prefix shared by many tokens is evaluated once. Terminating token-trie nodes emit their token IDs and the resulting machine states; tokens may span several commands. Duplicate textual pieces may map to multiple token IDs.

Masks are cached by complete machine state in another bounded LRU. Callers receive copies so mutating an allowlist cannot corrupt the cache. Defaults cap each cache at 4,096 entries. Reports expose hits, misses, trie visits and character transitions; those counters are mechanism evidence rather than a throughput claim.

## Tokenizer compatibility

The fast Hugging Face path recognizes a standalone ByteLevel decoder. Legal DSL text is ASCII: individually decoded ASCII pieces concatenate exactly under ByteLevel, while incomplete/non-ASCII byte pieces cannot form a legal ASCII token extension. Vocabulary IDs come from `get_vocab()` and are intersected with model output bounds. EOS is separate from textual pieces.

Unknown, composite or context-dependent decoders use `ReferenceTokenizerMask`, which decodes the complete suffix for each candidate and calls the existing slow reference. WordLevel decoding, for example, can insert spaces between tokens and must not be treated as concatenation of standalone pieces. `--force-reference` makes this path explicit for comparisons. Fallback identity and reason appear in each case report.

The optional ByteLevel integration test compares compiled allowlists directly with full-suffix reference allowlists at real BPE token boundaries. Broader tokenizer families remain on the slow path rather than being assumed compatible.

## Search objective

For token sequence `y = (t1, ..., tn)`, including its terminal EOS, the objective is:

```text
score(y | x) = sum_i log p_model(t_i | x, t_1 ... t_(i-1))
```

The scorer must return normalized log probabilities over its **original unmasked vocabulary**. Constraints remove illegal continuations without renormalizing the survivors. Renormalizing after masking would change sequence rankings and would incorrectly erase a low-probability EOS cost at a forced terminal state. There is no hidden length reward, length normalization, model grader, or task-goal bonus.

At each step, expand retained active hypotheses, collect completed EOS hypotheses separately, and keep the top `beam_width` active and completed hypotheses. Ties are broken deterministically by token IDs. A completed score that dominates every retained active prefix permits stopping because all future log probabilities are nonpositive. This is a bound over the retained search frontier; finite-width pruning remains approximate.

The engine returns the best completed hypothesis it found, or the best retained incomplete prefix if no completion was found. Token limits and candidate-expansion limits are explicit. A truncated search can return a completed answer while still setting `search_budget_exhausted`; `budget_exhausted` separately describes whether the selected answer itself lacks EOS. The engine never appends STOP or EOS as an unscored repair.

A true greedy implementation is provided separately. Beam width one is not silently called greedy: retaining alternative EOS branches can differ from choosing exactly one locally best token at every step.

## Correctness evidence

- Compiled syntax/semantic decisions match the independent slow references on 2,494 unique prefixes of all zero-, one-, and two-command programs in four initial-state/budget configurations: 9,976 comparisons, plus malformed prefixes.
- All 25 grid positions, seven remaining budgets, and twelve movement commands are checked against the interpreter: 2,100 movement transitions.
- Item placement/ownership checks cover 216 PICK/DROP transitions.
- Token-trie masks match direct compiled token transitions, including multi-command tokens, duplicate pieces, EOS and eviction.
- A hand-calculated example verifies that beam search selects probability `0.35 * 0.9 = 0.315` over greedy completion `0.45 * 0.1 = 0.045`; EOS probability is retained.
- Wide-beam results match an independently implemented exhaustive enumerator over all seven completed sequences in a small four-token horizon.

These are finite executable checks, not a claim of formal verification over arbitrary grammars or models.

## Complexity and limits

Compilation costs `O(sum(command_lengths) + sum(token_piece_lengths))`. An uncached token mask costs the number of token-trie nodes still compatible with the current finite state, plus cached/on-demand terminal viability checks. A beam run costs model scoring plus mask traversal and candidate sorting for up to `B * T` active prefixes; candidate expansion is capped independently. Memory includes the compiled tries, bounded caches and retained token histories.

The optional HF scorer deliberately recomputes full-prefix logits with `use_cache=False`. It is a transparent correctness backend, not an optimized KV-cache implementation. It performs no provider requests and loads only local checkpoints. Long prompts and incompatible vocabulary/context bounds fail explicitly.

Raw sequence likelihood favors short completions in the supplied replay distribution. The measured beam experiment often selects legal `STOP` programs instead of solving the requested task. Constraint correctness and task success are therefore reported separately. Choosing a different search objective would require an explicit option, new stopping bounds, and its own evaluation; no length bonus has been smuggled into this implementation.
