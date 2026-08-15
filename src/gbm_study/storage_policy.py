"""Fail-closed storage rules for shared GPU hosts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class StoragePolicyError(ValueError):
    """Raised before a run would violate the shared-host storage budget."""


@dataclass(frozen=True)
class SharedGPUStoragePolicy:
    """Bound local run inputs and require explicit approval for larger assets."""

    max_single_file_bytes: int = 2 * 1024**3
    max_total_input_bytes: int = 5 * 1024**3
    large_downloads_approved: bool = False

    def validate(self, paths: Iterable[Path]) -> dict[str, int]:
        files = tuple(dict.fromkeys(path.resolve() for path in paths))
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise StoragePolicyError("Configured local input is missing: " + ", ".join(missing))
        sizes = {str(path): path.stat().st_size for path in files}
        total = sum(sizes.values())
        if not self.large_downloads_approved:
            oversized = [path for path, size in sizes.items() if size > self.max_single_file_bytes]
            if oversized:
                raise StoragePolicyError(
                    "Shared-GPU input exceeds the per-file storage limit; stream or request "
                    "approval first: " + ", ".join(oversized)
                )
            if total > self.max_total_input_bytes:
                raise StoragePolicyError(
                    f"Shared-GPU inputs total {total} bytes, above the configured "
                    f"{self.max_total_input_bytes}-byte limit; stream a split/subset instead"
                )
        return {"local_input_bytes": total, "local_input_files": len(files)}


def policy_from_config(config: dict[str, object]) -> SharedGPUStoragePolicy:
    raw = config.get("shared_gpu", {})
    settings = raw if isinstance(raw, dict) else {}
    return SharedGPUStoragePolicy(
        max_single_file_bytes=int(settings.get("max_single_file_bytes", 2 * 1024**3)),
        max_total_input_bytes=int(settings.get("max_total_input_bytes", 5 * 1024**3)),
        large_downloads_approved=bool(settings.get("large_downloads_approved", False)),
    )
