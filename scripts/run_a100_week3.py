#!/usr/bin/env python3
"""Run the measured A100 preflight and scGPT benchmark with staged evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_stage(
    name: str, command: list[str], cwd: Path, output_dir: Path, env: dict[str, str]
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    (output_dir / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": command,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--deadline-utc", default="2026-08-18T06:41:00Z")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    scratch = (
        args.scratch or Path(os.environ.get("GBM_A100_SCRATCH", "/tmp/gbm-a100-run"))
    ).resolve()
    persistent = os.environ.get("GBM_PERSISTENT_OUTPUT_DIR")
    results_root = (args.results or Path(persistent) if persistent else args.results) or (
        scratch / "results"
    )
    session = Path(results_root).resolve() / datetime.now(timezone.utc).strftime(
        "week3-a100-%Y%m%dT%H%M%SZ"
    )
    session.mkdir(parents=True, exist_ok=False)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    env.setdefault("GBM_A100_SCRATCH", str(scratch))
    env.setdefault("HF_HOME", str(scratch / "huggingface"))
    env.setdefault("HF_DATASETS_CACHE", str(scratch / "huggingface" / "datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(scratch / "huggingface" / "transformers"))
    env.setdefault("PIP_NO_CACHE_DIR", "1")
    for path in (env["HF_HOME"], env["HF_DATASETS_CACHE"], env["TRANSFORMERS_CACHE"]):
        Path(path).mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "status": "running",
        "session": str(session),
        "config": str(args.config.resolve()),
        "stages": {},
    }
    atomic_json(session / "run_manifest.json", manifest)
    shutil.copy2(args.config, session / "model_config.yaml")

    preflight_path = session / "preflight.json"
    stages = [
        (
            "preflight",
            [
                sys.executable,
                "scripts/a100_preflight.py",
                "--config",
                str(args.config.resolve()),
                "--scratch",
                str(scratch),
                "--output",
                str(preflight_path),
                "--deadline-utc",
                args.deadline_utc,
            ],
        ),
        (
            "environment",
            [
                sys.executable,
                "scripts/check_environment.py",
                "--config",
                str(args.config.resolve()),
                "--json-out",
                str(session / "environment.json"),
                "--environment-export",
                str(session / "environment_export.json"),
            ],
        ),
        (
            "gpu_plan",
            [
                sys.executable,
                "scripts/plan_gpu.py",
                "--token-length",
                "2048",
                "--cells",
                "10000",
                "--batch-size",
                "32",
                "--output",
                str(session / "gpu_plan.json"),
            ],
        ),
        (
            "benchmark",
            [
                sys.executable,
                "scripts/benchmark_scgpt.py",
                "--config",
                str(args.config.resolve()),
                "--output",
                str(session / "scgpt_benchmark.json"),
            ],
        ),
    ]

    exit_code = 0
    for name, command in stages:
        stage = run_stage(name, command, repo, session, env)
        manifest["stages"][name] = stage
        atomic_json(session / "run_manifest.json", manifest)
        if stage["status"] != "completed":
            exit_code = int(stage["returncode"] or 2)
            break

    manifest["status"] = "completed" if exit_code == 0 else "blocked"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(session / "run_manifest.json", manifest)
    archive = shutil.make_archive(str(session), "gztar", root_dir=session)
    print(
        json.dumps(
            {"status": manifest["status"], "session": str(session), "archive": archive}, indent=2
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
