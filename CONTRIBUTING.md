# Working on the decoder

A mask change needs a prefix where the old and new behavior differ. Put that case in `tests/`, then compare the compiled state machine with the full-prefix reference. Check grammar acceptance, simulator execution, and the requested action sequence separately: reaching the same final cell can hide a skipped pick/drop operation.

For beam changes, preserve original conditional log probabilities and EOS costs. Do not introduce length normalization or rewards under an existing experiment name. Small exhaustive distributions are useful references for score ordering and completion retention.

Install the package and development tools with `python -m pip install -e . -r requirements-dev.txt`. Run `python -m unittest discover -s tests -v` and `python devtools/check_decoder.py --require-lint --wheel`. With `.[ml]` installed, set `RUN_ML_SMOKE=1` to exercise the random local-checkpoint integrations.

If tokenizer support changes, include a multi-character token, sparse ID case, output-vocabulary bound, and an explicit compiled/reference comparison. Keep replay fixtures labeled as replay. Review the raw failing generations before updating a recorded table in `cases/`.
