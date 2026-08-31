import pytest
import torch

from humming import dtypes
from humming.config import MmaType
from humming.schema.compressed_tensors import CompressedTensorsWeightSchema
from humming.schema.humming import HummingInputSchema, HummingWeightSchema
from humming.schema.modelopt import ModeloptNvfp4WeightSchema
from humming.transform import prepare_layer_config


def _nvfp4_tensors(
    shape_n_stacks: list[int],
    shape_k_stacks: list[int],
    global_scale_key: str,
    global_scales: list[float],
) -> dict[str, torch.Tensor]:
    shape_n = sum(shape_n_stacks)
    shape_k = sum(shape_k_stacks)
    return {
        "weight": torch.zeros((shape_n, shape_k // 2), dtype=torch.uint8),
        "weight_scale": torch.ones(
            (shape_n, shape_k // 16),
            dtype=torch.float8_e4m3fn,
        ),
        global_scale_key: torch.tensor(global_scales, dtype=torch.float32),
    }


def _assert_selects_sm90_w4a8(
    monkeypatch: pytest.MonkeyPatch,
    weight_schema: HummingWeightSchema,
    shape_n_stacks: list[int],
    shape_k_stacks: list[int],
    param_dtype: torch.dtype,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    layer_config = prepare_layer_config(
        shape_n=sum(shape_n_stacks),
        shape_k=sum(shape_k_stacks),
        weight_schema=weight_schema,
        input_schema=HummingInputSchema(a_dtype=dtypes.float8e4m3),
        torch_dtype=param_dtype,
    )

    assert layer_config.use_nvfp4_w4a8
    assert layer_config.bs_dtype == dtypes.DataType.from_torch_dtype(param_dtype)
    assert layer_config.kernel_bs_dtype == dtypes.float32


def _assert_preserves_sm80_fallback(
    monkeypatch: pytest.MonkeyPatch,
    weight_schema: HummingWeightSchema,
    shape_n_stacks: list[int],
    shape_k_stacks: list[int],
    param_dtype: torch.dtype,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    layer_config = prepare_layer_config(
        shape_n=sum(shape_n_stacks),
        shape_k=sum(shape_k_stacks),
        weight_schema=weight_schema,
        input_schema=HummingInputSchema(),
        torch_dtype=param_dtype,
    )

    assert layer_config.a_dtype == dtypes.DataType.from_torch_dtype(param_dtype)
    assert layer_config.mma_type == MmaType.MMA
    assert not layer_config.use_nvfp4_w4a8
    assert layer_config.bs_dtype == dtypes.DataType.from_torch_dtype(param_dtype)


@pytest.mark.parametrize(
    ("shape_n_stacks", "shape_k_stacks", "global_scales", "expected_scale", "param_dtype"),
    [
        (
            [2, 1],
            [32],
            [2.0, 4.0],
            [[2.0, 2.0], [2.0, 2.0], [4.0, 4.0]],
            torch.bfloat16,
        ),
        (
            [2],
            [16, 32],
            [2.0, 4.0],
            [[2.0, 4.0, 4.0], [2.0, 4.0, 4.0]],
            torch.float16,
        ),
    ],
    ids=["n-stack", "k-stack"],
)
def test_modelopt_nvfp4_heterogeneous_stack_preserves_w4a8_semantics(
    monkeypatch: pytest.MonkeyPatch,
    shape_n_stacks: list[int],
    shape_k_stacks: list[int],
    global_scales: list[float],
    expected_scale: list[list[float]],
    param_dtype: torch.dtype,
) -> None:
    source_schema = ModeloptNvfp4WeightSchema()
    tensors = _nvfp4_tensors(
        shape_n_stacks,
        shape_k_stacks,
        "weight_scale_2",
        global_scales,
    )

    weight_schema, output_tensors = source_schema.convert_humming(
        tensors=tensors,
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=param_dtype,
    )

    assert weight_schema.is_nvfp4
    assert weight_schema.bs_dtype is None
    assert "weight_scale_2" not in output_tensors
    assert output_tensors["weight_scale"].dtype == param_dtype
    torch.testing.assert_close(
        output_tensors["weight_scale"].float(),
        torch.tensor(expected_scale),
    )
    _assert_selects_sm90_w4a8(
        monkeypatch,
        weight_schema,
        shape_n_stacks,
        shape_k_stacks,
        param_dtype,
    )
    _assert_preserves_sm80_fallback(
        monkeypatch,
        weight_schema,
        shape_n_stacks,
        shape_k_stacks,
        param_dtype,
    )


@pytest.mark.parametrize(
    ("shape_n_stacks", "shape_k_stacks", "global_scales", "expected_scale", "param_dtype"),
    [
        (
            [2, 1],
            [32],
            [2.0, 4.0],
            [[0.5, 0.5], [0.5, 0.5], [0.25, 0.25]],
            torch.bfloat16,
        ),
        (
            [2],
            [16, 32],
            [2.0, 4.0],
            [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]],
            torch.float16,
        ),
    ],
    ids=["n-stack", "k-stack"],
)
def test_compressed_tensors_nvfp4_heterogeneous_stack_preserves_w4a8_semantics(
    monkeypatch: pytest.MonkeyPatch,
    shape_n_stacks: list[int],
    shape_k_stacks: list[int],
    global_scales: list[float],
    expected_scale: list[list[float]],
    param_dtype: torch.dtype,
) -> None:
    source_schema = CompressedTensorsWeightSchema(
        format="nvfp4-pack-quantized",
        type="float",
        num_bits=4,
        strategy="group",
        group_size=16,
    )
    tensors = _nvfp4_tensors(
        shape_n_stacks,
        shape_k_stacks,
        "weight_global_scale",
        global_scales,
    )
    tensors["weight_packed"] = tensors.pop("weight")

    weight_schema, output_tensors = source_schema.convert_humming(
        tensors=tensors,
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=param_dtype,
    )

    assert weight_schema.is_nvfp4
    assert weight_schema.bs_dtype is None
    assert "weight_scale_2" not in output_tensors
    assert output_tensors["weight_scale"].dtype == param_dtype
    torch.testing.assert_close(
        output_tensors["weight_scale"].float(),
        torch.tensor(expected_scale),
    )
    _assert_selects_sm90_w4a8(
        monkeypatch,
        weight_schema,
        shape_n_stacks,
        shape_k_stacks,
        param_dtype,
    )
    _assert_preserves_sm80_fallback(
        monkeypatch,
        weight_schema,
        shape_n_stacks,
        shape_k_stacks,
        param_dtype,
    )
