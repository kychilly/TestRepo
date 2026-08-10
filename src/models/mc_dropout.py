"""Monte Carlo dropout inference without changing the scGPT adapter contract."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .scgpt_adapter import ScGPTAdapter, elapsed_seconds


@dataclass(frozen=True)
class MCDropoutResult:
    mean: NDArray[np.float32]
    variance: NDArray[np.float32]
    pass_times: tuple[float, ...]
    single_pass_seconds: float

    @property
    def compute_multiplier(self) -> float:
        total = sum(self.pass_times)
        return total / self.single_pass_seconds if self.single_pass_seconds else float("nan")


def _dropout_modules(model: Any) -> list[tuple[Any, bool]]:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return []
    found: list[tuple[Any, bool]] = []
    for module in modules():
        if "dropout" in type(module).__name__.lower() and hasattr(module, "training"):
            found.append((module, bool(module.training)))
            if hasattr(module, "train"):
                module.train(True)
    return found


def infer_mc_dropout(
    adapter: ScGPTAdapter,
    prepared: Any,
    *,
    n_passes: int = 20,
    batch_size: int = 32,
    precision: str = "float32",
) -> MCDropoutResult:
    if n_passes < 2:
        raise ValueError("n_passes must be at least 2")
    outputs: list[NDArray[np.float32]] = []
    timings: list[float] = []
    single: float | None = None
    for _ in range(n_passes):
        changed = _dropout_modules(adapter.model)
        start = time.perf_counter()
        try:
            value = adapter.infer(prepared, batch_size=batch_size, precision=precision)
        finally:
            for module, training in changed:
                if hasattr(module, "train"):
                    module.train(training)
        duration = elapsed_seconds(start, time.perf_counter())
        outputs.append(value)
        timings.append(duration)
        if single is None:
            single = duration
    stacked = np.stack(outputs, axis=0)
    assert single is not None
    return MCDropoutResult(
        mean=np.asarray(stacked.mean(axis=0), dtype=np.float32),
        variance=np.asarray(stacked.var(axis=0), dtype=np.float32),
        pass_times=tuple(timings),
        single_pass_seconds=single,
    )


def multiplier_from_timings(
    single_pass_seconds: float, n_passes: int, total_seconds: float
) -> float:
    if single_pass_seconds <= 0 or n_passes < 1 or total_seconds < 0:
        raise ValueError("timings and pass count must be valid")
    return total_seconds / single_pass_seconds


def blocked_result(reason: str, *, n_passes: int, batch_size: int, device: str) -> dict[str, Any]:
    """Return a non-representative-safe artifact when real MC inference is unavailable."""
    return {
        "status": "blocked",
        "reason": reason,
        "timing": {
            "single_pass_seconds": None,
            "total_seconds": None,
            "compute_multiplier": None,
            "representative": False,
        },
        "model": {"n_passes": n_passes, "batch_size": batch_size, "device": device},
    }
