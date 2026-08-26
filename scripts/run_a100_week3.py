#!/usr/bin/env python3
"""Run the measured A100 preflight and scGPT benchmark with staged evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gbm_study.plain_english import write_json_with_explanation


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    write_json_with_explanation(path, value)


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
    parser.add_argument("--deadline-utc", default="2099-01-01T00:00:00Z")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--run-week3", action="store_true")
    parser.add_argument("--week3-config", type=Path, default=Path("config/week3_adit.yaml"))
    parser.add_argument("--week3-output", type=Path, default=None)
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    scratch = (
        args.scratch or Path(os.environ.get("GBM_A100_SCRATCH", "/tmp/gbm-a100-run"))
    ).resolve()
    persistent = os.environ.get("GBM_PERSISTENT_OUTPUT_DIR")
    results_root = (args.results or Path(persistent) if persistent else args.results) or (
        scratch / "results"
    )
    session_name = args.session_id or datetime.now(timezone.utc).strftime(
        "week3-a100-%Y%m%dT%H%M%SZ"
    )
    session = Path(results_root).resolve() / session_name
    session.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    env.setdefault("GBM_A100_SCRATCH", str(scratch))
    env.setdefault("HF_HOME", str(scratch / "huggingface"))
    env.setdefault("HF_DATASETS_CACHE", str(scratch / "huggingface" / "datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(scratch / "huggingface" / "transformers"))
    env.setdefault("PIP_NO_CACHE_DIR", "1")
    for path in (env["HF_HOME"], env["HF_DATASETS_CACHE"], env["TRANSFORMERS_CACHE"]):
        Path(path).mkdir(parents=True, exist_ok=True)

    manifest_path = session / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Persistent A100 run manifest must be a JSON object")
        manifest["status"] = "running"
    else:
        manifest = {
            "status": "running",
            "session": str(session),
            "config": str(args.config.resolve()),
            "stages": {},
        }
    atomic_json(session / "run_manifest.json", manifest)

    def checkpoint_on_stop(signum: int, _frame: Any) -> None:
        manifest["status"] = "interrupted"
        manifest["interrupted_signal"] = signum
        manifest["interrupted_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(session / "run_manifest.json", manifest)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, checkpoint_on_stop)
    signal.signal(signal.SIGINT, checkpoint_on_stop)
    if not (session / "model_config.yaml").is_file():
        shutil.copy2(args.config, session / "model_config.yaml")

    import yaml  # type: ignore[import-untyped]

    model_payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(model_payload, dict):
        raise ValueError("A100 model config must be a YAML object")
    token_length = int(model_payload.get("token_length", 1200))
    batch_size = int(model_payload.get("batch_size", 32))

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
                str(token_length),
                "--cells",
                "10000",
                "--batch-size",
                str(batch_size),
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

    if args.run_week3:
        week3_output = args.week3_output or (session / "week3_experiments")
        week3_config = session / "week3_runtime_config.yaml"
        week3_payload = yaml.safe_load(args.week3_config.read_text(encoding="utf-8"))
        if not isinstance(week3_payload, dict):
            raise ValueError("Week 3 runtime config must be a YAML object")
        week3_payload["week2_timing_path"] = str(session / "scgpt_benchmark.json")
        week3_config.write_text(yaml.safe_dump(week3_payload, sort_keys=False), encoding="utf-8")
        stages.append(
            (
                "week3_experiments",
                [
                    sys.executable,
                    "scripts/run_week3_experiments.py",
                    "--config",
                    str(week3_config),
                    "--output",
                    str(week3_output),
                ],
            )
        )

    exit_code = 0
    for name, command in stages:
        prior = manifest.get("stages", {}).get(name, {})
        if prior.get("status") == "completed":
            continue
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