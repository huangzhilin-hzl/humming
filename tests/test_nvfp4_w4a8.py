import re
from pathlib import Path

import pytest
import torch

from humming import dtypes
from humming.config import LayerConfig, MmaType, WeightScale2Type
from humming.transform import process_nvfp4_w4a8_scale
from humming.tune.candidate import get_problem_rejection_reasons

REPO_ROOT = Path(__file__).resolve().parents[1]


def _layer(**updates) -> LayerConfig:
    config = {
        "shape_n": 128,
        "shape_k": 256,
        "a_dtype": dtypes.float8e4m3,
        "b_dtype": dtypes.float4e2m1,
        "c_dtype": dtypes.bfloat16,
        "bs_dtype": dtypes.float8e4m3,
        "weight_scale_group_size": 16,
        "weight_scale_2_type": WeightScale2Type.TENSOR,
        "mma_type": MmaType.WGMMA,
    }
    config.update(updates)
    return LayerConfig(**config)


def test_sm90_nvfp4_w4a8_path_is_selected_and_uses_bf16_residuals(monkeypatch):
    config = _layer()

    assert config.use_nvfp4_w4a8
    assert config.kernel_bs_dtype == dtypes.float32
    assert config.weight_scale_2_type == WeightScale2Type.TENSOR
    assert config.as_dtype == dtypes.float32
    monkeypatch.setattr(LayerConfig, "use_native_dequant", False)
    assert "kUseNvfp4W4A8 = true" in config.to_cpp_str()


def test_nvfp4_scale_factorization_preserves_effective_scale():
    weight_scale = torch.tensor(
        [[0.25, 1.0, 4.0, 0.0], [2.0, 0.5, 8.0, 1.0]],
        dtype=torch.float32,
    )
    packed_weight = torch.full((2, 8), 0x11111111, dtype=torch.int32)
    packed_scale = process_nvfp4_w4a8_scale(weight_scale, weight=packed_weight)
    packed_bits = packed_scale.view(torch.int32)
    residual = (packed_bits & 0xFFFF).to(torch.uint16).view(torch.bfloat16)
    coarse_scale = (packed_bits >> 16).to(torch.uint16).view(torch.bfloat16)

    reconstructed = residual.float() * coarse_scale.float()
    torch.testing.assert_close(reconstructed, weight_scale, rtol=1e-2, atol=1e-5)
    assert packed_scale.dtype == torch.float32
    assert (residual.float().abs() * 0.5 <= 448).all()
    torch.testing.assert_close(coarse_scale[..., ::2], coarse_scale[..., 1::2])


@pytest.mark.parametrize(
    ("weight_scale", "error"),
    [
        (torch.tensor([[float("nan"), 1.0]]), "finite"),
        (torch.tensor([[float("inf"), 1.0]]), "finite"),
        (torch.tensor([[-1.0, 1.0]]), "non-negative"),
        (torch.tensor([[-0.0, 1.0]]), "negative zero"),
    ],
)
def test_nvfp4_scale_factorization_rejects_invalid_scales(weight_scale, error):
    with pytest.raises(ValueError, match=error):
        process_nvfp4_w4a8_scale(weight_scale)


def test_nvfp4_decoder_residuals_match_repacked_n_and_k_layout():
    source = (REPO_ROOT / "humming/include/humming/datatype/nvfp4_w4a8.cuh").read_text()
    assignments = re.findall(
        r"promoted_weight\[(\d)\] = promote_nvfp4_nibbles\(packed(\d), residual(\d+), (true|false)\)",
        source,
    )

    # Repack emits one word per N row; low/high nibbles select K0/K16.
    assert assignments == [
        ("0", "0", "10", "true"),
        ("1", "0", "00", "false"),
        ("2", "1", "11", "true"),
        ("3", "1", "01", "false"),
    ]


@pytest.mark.parametrize(
    ("byte_spaced", "expected"),
    [
        (0x04030201, 0x4321),
        (0x07060500, 0x7650),
        (0xFFFFFFFF, 0x7777),
    ],
)
def test_nvfp4_prmt_selectors_are_packed_into_contiguous_nibbles(byte_spaced, expected):
    packed = (
        (byte_spaced & 0x00000007)
        | ((byte_spaced & 0x00000700) >> 4)
        | ((byte_spaced & 0x00070000) >> 8)
        | ((byte_spaced & 0x07000000) >> 12)
    )
    assert packed == expected

    source = (REPO_ROOT / "humming/include/humming/datatype/nvfp4_w4a8.cuh").read_text()
    assert "selectors = nvfp4_pack_prmt_selectors(packed >> 4);" in source
    assert "selectors = nvfp4_pack_prmt_selectors(packed);" in source


def test_nvfp4_warp_n64_scale_loader_reads_contiguous_vectors():
    source = (REPO_ROOT / "humming/include/humming/memory/s2r_loader/loader_bs.cuh").read_text()
    layout1 = source.split("void load_layout1", 1)[1].split("void load_layout2", 1)[0]

    assert "logical_slot * kLoadsPerScaleGroup + pack_block_id * kPackBlockSize" in layout1
    assert "smem_ptr_load[s_sh_rd + group * kGroupStride + load]" in layout1

    # Warp-N=64 with FP32 scales uses two uint4 loads per lane quartet.
    logical_slots = range(8)
    assert [(slot * 2, slot * 2 + 1) for slot in logical_slots] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
    ]


def test_nvfp4_w4a8_requires_sm90():
    config = _layer()

    config.check_sm_version(90)
    for unsupported_sm in (89, 100):
        with pytest.raises(AssertionError, match="requires SM90"):
            config.check_sm_version(unsupported_sm)


def test_nvfp4_k16_weight_scale_is_legal_for_fp8_wgmma():
    assert get_problem_rejection_reasons(_layer()) == ()
