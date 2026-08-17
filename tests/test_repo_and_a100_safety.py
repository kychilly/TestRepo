from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def test_dataset_patterns_are_ignored() -> None:
    repo = Path(__file__).parents[1]
    candidates = [
        "data/raw/full.h5ad",
        "TP53 Dataset(preprocessed) 2/pilot/new.h5ad",
        "incoming/cohort.zip",
        "models/checkpoint.pt",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=repo,
        input="\n".join(candidates) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert set(result.stdout.splitlines()) == set(candidates)


def test_precommit_hook_rejects_force_added_dataset(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / ".githooks" / "pre-commit"
    hook = tmp_path / ".githooks" / "pre-commit"
    hook.parent.mkdir()
    shutil.copy2(source, hook)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    dataset = tmp_path / "forced.h5ad"
    dataset.write_bytes(b"not-a-real-dataset")
    subprocess.run(["git", "add", "-f", dataset.name], cwd=tmp_path, check=True)
    result = subprocess.run([str(hook)], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 1
    assert "BLOCKED dataset/model artifact" in result.stderr


def test_a100_preflight_fails_closed_for_unresolved_template(tmp_path: Path) -> None:
    from scripts.a100_preflight import inspect

    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "model.yaml"
    config.write_text(
        "data_source: huggingface_stream\nhf_dataset_id: null\ncheckpoint_path: null\n",
        encoding="utf-8",
    )
    result = inspect(config, repo, tmp_path / "scratch", "2099-01-01T00:00:00Z")
    assert result["status"] == "blocked"
    assert any("Hugging Face" in blocker for blocker in result["blockers"])
    json.dumps(result)
