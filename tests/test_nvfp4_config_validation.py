import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.tune import get_heuristics_config
from humming.tune.candidate import (
    DeviceProfile,
    ScheduleCandidate,
    TuningProblem,
    analyze_candidate,
    get_problem_rejection_reasons,
)
from humming.tune.sm90 import Sm90Heuristics
from humming.tune.sm90_h20 import Sm90H20Heuristics


def _layer(**updates) -> LayerConfig:
    config = {
        "shape_n": 512,
        "shape_k": 256,
        "num_experts": 8,
        "a_dtype": dtypes.float8e4m3,
        "b_dtype": dtypes.float4e2m1,
        "c_dtype": dtypes.bfloat16,
        "bs_dtype": dtypes.float8e4m3,
        "weight_scale_group_size": 16,
        "mma_type": MmaType.WGMMA,
    }
    config.update(updates)
    return LayerConfig(**config)


def _device() -> DeviceProfile:
    return DeviceProfile(
        name="H200",
        sm_version=90,
        num_sms=132,
        max_smem_size=227 * 1024,
    )


def test_nvfp4_auto_selection_and_explicit_folded_scale_selection():
    assert _layer().use_nvfp4_w4a8

    folded_auto = _layer(bs_dtype=dtypes.bfloat16)
    folded_explicit = _layer(
        bs_dtype=dtypes.bfloat16,
        use_nvfp4_w4a8=True,
    )

    assert not folded_auto.use_nvfp4_w4a8
    assert folded_explicit.use_nvfp4_w4a8
    assert folded_explicit.kernel_bs_dtype == dtypes.float32


@pytest.mark.parametrize("bs_dtype", [dtypes.float8e4m3, dtypes.bfloat16])
def test_explicitly_disabling_required_nvfp4_promotion_fails_early(bs_dtype):
    with pytest.raises(AssertionError, match="only supported through NVFP4 W4A8"):
        _layer(bs_dtype=bs_dtype, use_nvfp4_w4a8=False)


@pytest.mark.parametrize(
    "conflicting_mode",
    ["use_fused_e8m0_scale", "use_int_weight_scale"],
)
def test_special_weight_scale_modes_are_mutually_exclusive(conflicting_mode):
    with pytest.raises(AssertionError, match="modes are mutually exclusive"):
        _layer(**{conflicting_mode: True})


def test_fused_and_integer_weight_scale_modes_are_mutually_exclusive():
    with pytest.raises(AssertionError, match="modes are mutually exclusive"):
        _layer(
            bs_dtype=dtypes.float8e8m0,
            use_fused_e8m0_scale=True,
            use_int_weight_scale=True,
        )


def test_tuning_problem_and_heuristics_reject_nvfp4_f16_accumulation():
    layer = _layer()

    with pytest.raises(ValueError, match="requires FP32 accumulation"):
        TuningProblem(
            layer_config=layer,
            shape_m=64,
            gemm_type=GemmType.INDEXED,
            device=_device(),
            use_f16_accum=True,
        )

    with pytest.raises(ValueError, match="requires FP32 accumulation"):
        Sm90Heuristics.get_config(
            layer,
            shape_m=64,
            use_f16_accum=True,
            gemm_type=GemmType.INDEXED,
        )

    with pytest.raises(ValueError, match="requires FP32 accumulation"):
        Sm90H20Heuristics.get_config(
            layer,
            shape_m=64,
            use_f16_accum=True,
            gemm_type=GemmType.INDEXED,
        )

    with pytest.raises(ValueError, match="requires FP32 accumulation"):
        get_heuristics_config(
            layer,
            shape_m=64,
            use_f16_accum=True,
            gemm_type=GemmType.INDEXED,
        )

    assert get_problem_rejection_reasons(
        layer,
        use_f16_accum=True,
    ) == ("NVFP4 W4A8 promotion requires FP32 accumulation",)


def test_candidate_rejects_nvfp4_f16_accumulation():
    layer = _layer()
    problem = TuningProblem(
        layer_config=layer,
        shape_m=64,
        gemm_type=GemmType.INDEXED,
        device=_device(),
    )
    candidate = ScheduleCandidate.from_config(
        "nvfp4_f16_accum",
        {
            "block_shape": (8, 128, 128),
            "warp_shape": (8, 32, 64),
            "use_stream_k": False,
            "use_f16_accum": True,
            "num_stages": 3,
        },
    )

    analysis = analyze_candidate(problem, candidate)

    assert not analysis.launchable
    assert "NVFP4 W4A8 promotion requires FP32 accumulation" in analysis.hard_violations
