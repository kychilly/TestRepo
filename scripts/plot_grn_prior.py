#!/usr/bin/env python3
"""Render a compact, provenance-preserving view of a pilot GRN prior."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from models.grn import load_edges


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-prior", type=Path, required=True)
    parser.add_argument("--held-out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = load_edges(args.train_prior)
    held = load_edges(args.held_out)
    edges = [(edge.data["source_gene"], edge.data["target_gene"], "train") for edge in train]
    edges.extend((edge.data["source_gene"], edge.data["target_gene"], "held-out") for edge in held)
    nodes = sorted({node for source, target, _ in edges for node in (source, target)})
    positions = {
        node: (
            math.cos(2 * math.pi * index / len(nodes)),
            math.sin(2 * math.pi * index / len(nodes)),
        )
        for index, node in enumerate(nodes)
    }
    figure, axis = plt.subplots(figsize=(8, 8))
    for source, target, edge_type in edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        axis.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "color": "#d62728" if edge_type == "held-out" else "#1f77b4",
                "alpha": 0.75,
            },
        )
    axis.scatter(
        [positions[node][0] for node in nodes],
        [positions[node][1] for node in nodes],
        s=900,
        color="#f0f0f0",
        edgecolor="#333333",
        zorder=3,
    )
    for node, (x, y) in positions.items():
        axis.text(x, y, node, ha="center", va="center", fontsize=9, zorder=4)
    axis.set_title("Pilot GRN prior: train edges and Adit held-out edges")
    axis.text(0.02, 0.02, f"train={len(train)}  held-out={len(held)}", transform=axis.transAxes)
    axis.set_axis_off()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(
        f"status=completed output={args.output} train_edges={len(train)} held_out_edges={len(held)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
