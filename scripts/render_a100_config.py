#!/usr/bin/env python3
"""Render an untracked A100 runtime config from explicit immutable inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    import yaml  # type: ignore[import-untyped]

    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=Path("config/model_shared_gpu.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-dataset-id", required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--hf-split", required=True)
    parser.add_argument("--hf-expression-column", required=True)
    parser.add_argument("--hf-gene-ids-column", required=True)
    parser.add_argument("--patient-id-column", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--model-args", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument(
        "--precision", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    payload = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A100 template must contain a YAML object")
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    try:
        output.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ValueError("Runtime model config must be written outside the Git repository")

    updates: dict[str, Any] = {
        "data_source": "huggingface_stream",
        "hf_dataset_id": args.hf_dataset_id,
        "hf_revision": args.hf_revision,
        "hf_split": args.hf_split,
        "hf_expression_column": args.hf_expression_column,
        "hf_gene_ids_column": args.hf_gene_ids_column,
        "patient_id_column": args.patient_id_column,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "vocabulary_path": str(args.vocabulary.resolve()),
        "model_args_path": str(args.model_args.resolve()),
        "split_file": str(args.split_file.resolve()),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "requested_device": "cuda",
    }
    payload.update(updates)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Wrote untracked A100 runtime config: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
