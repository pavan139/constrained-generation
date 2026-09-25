# Decoder check record

The current machine-readable run is [reports/verification.json](../reports/verification.json). It records the source digest, toolchain, executed/skipped test counts, CLI outcomes, and cold-wheel checks for this layout.

The recorded replay and tiny local-checkpoint examples in `cases/` establish the stated decoder and integration behavior. The [compiled decoding record](compiled-decoding-evidence.md) contains search counters and the 20-case measured beam comparison. Model-quality claims require a separate identified checkpoint experiment.

Run `python devtools/check_decoder.py --require-lint --wheel` after a change. The optional random-checkpoint tests require `.[ml]` and `RUN_ML_SMOKE=1`; skipped tests remain visible in the report.
