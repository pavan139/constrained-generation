.PHONY: test demo evaluate
test:
	PYTHONPATH=src python -m unittest discover -s tests -v
demo:
	PYTHONPATH=src python -m constrained_generation demo --output artifacts/demo.json
evaluate:
	PYTHONPATH=src python -m constrained_generation evaluate --output artifacts/replay.json
