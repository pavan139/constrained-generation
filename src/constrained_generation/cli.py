from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-state decoding with separate syntax and semantic metrics")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "evaluate", "hf-evaluate", "beam-evaluate", "hf-beam-evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--output")
        p.add_argument("--count", type=int, default=3 if name == "demo" else 20)
        p.add_argument("--token-budget", type=int, default=128)
        if name in {"beam-evaluate", "hf-beam-evaluate"}:
            p.add_argument("--beam-width", type=int, default=4)
        if name == "hf-beam-evaluate":
            p.add_argument("--force-reference", action="store_true")
        if name in {"hf-evaluate", "hf-beam-evaluate"}:
            p.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.command == "beam-evaluate":
        from .compiled import evaluate_beams

        report = evaluate_beams(args.count, args.beam_width, args.token_budget)
    elif args.command == "hf-beam-evaluate":
        from .beam_hf import run_beams

        report = run_beams(args.model, args.count, args.token_budget, args.beam_width, args.force_reference)
    elif args.command == "hf-evaluate":
        from .hf import run

        report = run(args.model, args.count, args.token_budget)
    else:
        report = evaluate(args.token_budget, args.count)
    value = json.dumps(report, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    else:
        print(value, end="")
