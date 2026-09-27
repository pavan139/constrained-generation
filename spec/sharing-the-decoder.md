# Sharing the decoder and its cases

Publish this directory as the `constrained-generation` repository, keeping `src/`, `tests/`, `presets/`, `cases/`, `spec/`, `devtools/`, package metadata, license, and the GitHub Actions workflow together. Its checks do not depend on a parent checkout.

Before pushing, run `python devtools/check_decoder.py --require-lint --wheel`. Inspect `reports/verification.json` for skipped optional integrations and installed-wheel results. A change to compiled masks should also include the prefix-equivalence result; a search change should include its original-score/EOS comparison.

The small JSON replay and beam reports belong with the source. Local model weights belong outside Git. For a checkpoint experiment, publish its exact identity, tokenizer backend, generation settings, and whether compiled or reference masking was selected. Keep incorrect `STOP`-preferred outputs and instruction failures in the report.

Create an empty remote under your account and push this repository's own branch. The nested CI workflow then runs for that remote. A successful synthetic replay should be described as a decoder experiment, not a pretrained model result.
