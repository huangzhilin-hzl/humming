import torch

from humming import dtypes
from humming.config import LayerConfig, MmaType, WeightScale2Type
from humming.transform import process_nvfp4_w4a8_scale
from humming.tune.candidate import get_problem_rejection_reasons


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


def test_nvfp4_k16_weight_scale_is_legal_for_fp8_wgmma():
    assert get_problem_rejection_reasons(_layer()) == ()
