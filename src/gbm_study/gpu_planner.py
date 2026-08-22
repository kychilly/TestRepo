"""Actual CUDA discovery and load-balanced workload planning.

This module never invents GPU models, throughput, or CPU substitutes. A plan is
ready only when CUDA is visible and device memory can be queried. Timing comes
from the separate scGPT benchmark after a real forward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


class GPUPlanningError(ValueError):
    """Raised when a real CUDA plan cannot be created."""


@dataclass(frozen=True)
class CUDADevice:
    index: int
    name: str
    total_memory_bytes: int
    free_memory_bytes: int
    supports_bfloat16: bool


@dataclass(frozen=True)
class CUDAPlan:
    cells: int
    token_length: int
    batch_size: int
    precision: str
    devices: tuple[CUDADevice, ...]
    shards: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "cells": self.cells,
            "token_length": self.token_length,
            "batch_size": self.batch_size,
            "precision": self.precision,
            "devices": [
                {
                    "index": device.index,
                    "name": device.name,
                    "total_memory_bytes": device.total_memory_bytes,
                    "free_memory_bytes": device.free_memory_bytes,
                    "supports_bfloat16": device.supports_bfloat16,
                }
                for device in self.devices
            ],
            "shards": list(self.shards),
            "measured_gpu_seconds_per_10000_cells": None,
        }


def weighted_shards(cells: int, workers: Iterable[tuple[str, int]]) -> tuple[dict[str, Any], ...]:
    """Assign cells proportional to currently free CUDA memory."""
    workers = tuple(workers)
    if cells <= 0 or not workers or any(weight <= 0 for _, weight in workers):
        raise GPUPlanningError("cells and worker free-memory weights must be positive")
    total_weight = sum(weight for _, weight in workers)
    raw = [cells * weight / total_weight for _, weight in workers]
    counts = [math.floor(value) for value in raw]
    remainder = cells - sum(counts)
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[
        :remainder
    ]:
        counts[index] += 1
    return tuple(
        {"device": name, "cells": count, "free_memory_weight": weight}
        for (name, weight), count in zip(workers, counts)
    )


def build_plan(
    devices: tuple[CUDADevice, ...],
    *,
    cells: int,
    token_length: int,
    batch_size: int,
    precision: str = "auto",
) -> CUDAPlan:
    """Build a plan from observed devices; all values must come from CUDA."""
    if not devices:
        raise GPUPlanningError("No CUDA devices were observed")
    if cells <= 0 or token_length <= 0 or batch_size <= 0:
        raise GPUPlanningError("cells, token_length, and batch_size must be positive")
    if precision not in {"auto", "float16", "bfloat16", "float32"}:
        raise GPUPlanningError("precision must be auto, float16, bfloat16, or float32")
    if precision == "auto":
        precision = "bfloat16" if all(device.supports_bfloat16 for device in devices) else "float16"
    if precision == "bfloat16" and not all(device.supports_bfloat16 for device in devices):
        raise GPUPlanningError(
            "bfloat16 was requested but at least one CUDA device does not support it"
        )
    shards = weighted_shards(
        cells,
        tuple((str(device.index), device.free_memory_bytes) for device in devices),
    )
    return CUDAPlan(cells, token_length, batch_size, precision, devices, shards)


def discover_cuda_devices(device_ids: tuple[int, ...] | None = None) -> tuple[CUDADevice, ...]:
    """Discover actual visible CUDA devices and their current free memory."""
    try:
        import torch
    except ImportError as exc:
        raise GPUPlanningError("PyTorch is not installed") from exc
    if not torch.cuda.is_available():
        raise GPUPlanningError("CUDA is unavailable; no GPU plan was created")
    available = tuple(range(torch.cuda.device_count()))
    selected = available if device_ids is None else device_ids
    if not selected or any(index not in available for index in selected):
        raise GPUPlanningError(f"Requested CUDA devices are not visible: {selected}")
    devices = []
    for index in selected:
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        with torch.cuda.device(index):
            supports_bfloat16 = bool(torch.cuda.is_bf16_supported())
        devices.append(
            CUDADevice(
                index=index,
                name=str(properties.name),
                total_memory_bytes=int(total_bytes),
                free_memory_bytes=int(free_bytes),
                supports_bfloat16=supports_bfloat16,
            )
        )
    return tuple(devices)


def plan_cuda_work(
    *,
    cells: int,
    token_length: int,
    batch_size: int,
    precision: str = "auto",
    device_ids: tuple[int, ...] | None = None,
) -> CUDAPlan:
    return build_plan(
        discover_cuda_devices(device_ids),
        cells=cells,
        token_length=token_length,
        batch_size=batch_size,
        precision=precision,
    )
