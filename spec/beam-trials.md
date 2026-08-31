# Running decoder comparisons

Start with `python -m constrained_generation evaluate --output artifacts/replay.json`. This uses the same ranked candidate strings for unconstrained, grammar-masked, and semantic-masked character generation. It measures mask mechanics; a reference completion is visible to every replay policy.

For the scored engine, run:

```bash
python -m constrained_generation beam-evaluate --count 20 --beam-width 4 --token-budget 128 --output artifacts/beam.json
```

Compare syntax validity, executability, final-state success, action-aware instruction success, exhausted budgets, candidate expansions, and policy calls. A valid empty-action program must not receive instruction credit for skipping a required pick/drop. Beam likelihood includes EOS and remains normalized over the original vocabulary, so a short legal answer can beat a longer correct one.

`cases/beam-evaluation.json` records the executed 20-case result. Preserve its low goal score when discussing the search objective. A proposed length penalty is a different objective and needs a separately named run.

The local HF path requires the `ml` extra and an existing checkpoint. Record checkpoint/tokenizer identity, output-vocabulary size, mask backend, context/output limits, and hardware. Compare `--force-reference` with the compiled backend where certification applies. The random tiny-model smoke checks integration only; it does not estimate useful language-model accuracy.

Historical traces retain paths from their original runs. Current commands use `devtools/`, `presets/`, and `cases/`.
