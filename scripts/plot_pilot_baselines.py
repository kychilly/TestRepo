#!/usr/bin/env python3
"""Plot test-only pilot baseline metrics from evaluator artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True, help="method=metrics.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names: list[str] = []
    macro_f1: list[float] = []
    balanced_accuracy: list[float] = []
    for item in args.metrics:
        name, raw_path = item.split("=", 1)
        point = json.loads(Path(raw_path).read_text(encoding="utf-8"))["point_estimate"]
        names.append(name)
        macro_f1.append(float(point["macro_f1"]))
        balanced_accuracy.append(float(point["balanced_accuracy"]))
    x = list(range(len(names)))
    figure, axis = plt.subplots(figsize=(7, 4))
    width = 0.36
    axis.bar([value - width / 2 for value in x], macro_f1, width, label="Macro-F1")
    axis.bar(
        [value + width / 2 for value in x], balanced_accuracy, width, label="Balanced accuracy"
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test-set score")
    axis.set_title("Pilot baseline comparison (test patients only)")
    axis.set_xticks(x, names)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"status=completed output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
