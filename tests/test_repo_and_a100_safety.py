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


def test_a100_asset_verification_detects_transfer_corruption(tmp_path: Path) -> None:
    from scripts.build_a100_asset_bundle import ASSET_PATHS, sha256, verify_assets

    files: dict[str, dict[str, int | str]] = {}
    for relative in ASSET_PATHS:
        asset = tmp_path / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(relative.encode("utf-8"))
        files[relative] = {"bytes": asset.stat().st_size, "sha256": sha256(asset)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    assert verify_assets(tmp_path, manifest)["status"] == "passed"
    asset = tmp_path / ASSET_PATHS[0]
    asset.write_bytes(b"corrupted")
    result = verify_assets(tmp_path, manifest)
    assert result["status"] == "blocked"
    assert result["mismatches"]


def test_a100_asset_verification_rejects_incomplete_manifest(tmp_path: Path) -> None:
    from scripts.build_a100_asset_bundle import verify_assets

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    "splits/combined_full_cohort_neftel_patient_splits.json": {
                        "bytes": 0,
                        "sha256": "",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result = verify_assets(tmp_path, manifest)
    assert result["status"] == "blocked"
    assert len(result["manifest_missing_entries"]) == 11


def test_incomplete_bundle_generation_preserves_trusted_manifest(tmp_path: Path) -> None:
    from scripts.build_a100_asset_bundle import main

    manifest = tmp_path / "trusted.json"
    trusted = '{"trusted": true}\n'
    manifest.write_text(trusted, encoding="utf-8")
    return_code = main(["--root", str(tmp_path), "--manifest", str(manifest)])
    assert return_code == 2
    assert manifest.read_text(encoding="utf-8") == trusted
    assert (tmp_path / "trusted_generation_error.json").is_file()
