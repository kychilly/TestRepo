from __future__ import annotations

import pytest

from gbm_study.gpu_planner import (
    CUDADevice,
    GPUPlanningError,
    build_plan,
    weighted_shards,
)


def devices() -> tuple[CUDADevice, ...]:
    return (
        CUDADevice(0, "A100", 80_000, 60_000, True),
        CUDADevice(1, "T4", 16_000, 10_000, False),
    )


def test_shards_use_observed_free_memory_and_conserve_cells() -> None:
    shards = weighted_shards(1000, (("0", 60_000), ("1", 10_000)))
    assert sum(item["cells"] for item in shards) == 1000
    assert shards[0]["cells"] > shards[1]["cells"]


def test_plan_uses_real_device_metadata_and_safe_precision() -> None:
    plan = build_plan(devices(), cells=1000, token_length=2048, batch_size=32)
    assert plan.precision == "float16"
    assert [device.name for device in plan.devices] == ["A100", "T4"]
    assert plan.to_dict()["measured_gpu_seconds_per_10000_cells"] is None


def test_bfloat16_requires_support_on_every_selected_device() -> None:
    with pytest.raises(GPUPlanningError, match="bfloat16"):
        build_plan(devices(), cells=100, token_length=128, batch_size=4, precision="bfloat16")


def test_empty_cuda_inventory_is_rejected() -> None:
    with pytest.raises(GPUPlanningError, match="No CUDA devices"):
        build_plan((), cells=100, token_length=128, batch_size=4)
