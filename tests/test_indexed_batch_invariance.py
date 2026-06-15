import json
import math

import pytest
import torch

from humming.config import GemmType
from humming.layer import HummingLayer, HummingMethod
from humming.tune import get_heuristics_config
from humming.utils.test import skip_if_unsupported


def _skip_unless_h20():
    if "H20" not in torch.cuda.get_device_name():
        pytest.skip("H20-specific indexed W4A8 heuristics test")


def _select_config(configs, valid_shape_m):
    for min_shape_m, max_shape_m, config in configs:
        if valid_shape_m > min_shape_m and valid_shape_m <= max_shape_m:
            return config
    raise AssertionError(f"no config found for valid_shape_m={valid_shape_m}")


def _make_indexed_metadata(topk_ids, num_experts, block_size):
    flat_topk_ids = topk_ids.reshape(-1)
    sorted_ids_by_expert = []
    expert_ids = []
    invalid_id = flat_topk_ids.numel()
    for expert_id in range(num_experts):
        token_ids = torch.where(flat_topk_ids == expert_id)[0].to(torch.int32)
        num_blocks = math.ceil(token_ids.numel() / block_size)
        padded_size = num_blocks * block_size
        if padded_size != token_ids.numel():
            token_ids = torch.nn.functional.pad(
                token_ids,
                pad=(0, padded_size - token_ids.numel()),
                value=invalid_id,
            )
        sorted_ids_by_expert.append(token_ids)
        expert_ids.extend([expert_id] * num_blocks)

    sorted_ids = torch.cat(sorted_ids_by_expert, dim=0).to(torch.int32)
    expert_ids = torch.tensor(expert_ids, dtype=torch.int32, device=topk_ids.device)
    num_tokens_padded = torch.tensor(
        sorted_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    return sorted_ids, expert_ids, num_tokens_padded


def _make_topk_ids(shape_m, num_experts, top_k, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scores = torch.randn((shape_m, num_experts), generator=generator, device=device)
    return torch.topk(scores, k=top_k, dim=1).indices.to(torch.int32)


def _make_inputs(shape_m, shape_k, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    inputs = torch.randn(
        (shape_m, shape_k),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    return inputs / inputs.float().std().to(torch.bfloat16)


def _run_indexed_gemm(layer, inputs, topk_ids, tuning_configs):
    valid_shape_m = topk_ids.numel()
    meta = layer.humming_metas[""]
    config = _select_config(tuning_configs, valid_shape_m)
    block_size = config["block_shape"][0]
    sorted_ids, expert_ids, num_tokens_padded = _make_indexed_metadata(
        topk_ids,
        meta.num_experts,
        block_size,
    )
    output = torch.empty(
        (valid_shape_m, meta.shape_n),
        dtype=meta.param_dtype,
        device=inputs.device,
    )
    HummingMethod.forward_layer(
        layer=layer,
        inputs=inputs,
        outputs=output,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        top_k=topk_ids.shape[1],
        valid_shape_m=valid_shape_m,
        compute_config=json.dumps(
            {"use_f16_accum": False, "gemm_type": GemmType.INDEXED.value}
        ),
        tuning_config=json.dumps(tuning_configs),
    )
    torch.cuda.synchronize(inputs.device)
    return output


def test_h20_indexed_w4a8_streamk_uses_fp32_reduction():
    skip_if_unsupported(a_dtype="float8e4m3", mma_type="wgmma")
    _skip_unless_h20()

    device = torch.device("cuda:0")
    shape_k = 512
    shape_n = 1024
    num_experts = 16
    top_k = 8
    small_m = 32
    large_m = 128

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        weight_config={
            "dtype": "float4e2m1",
            "group_size": 32,
            "scale_dtype": "float8e8m0",
            "has_zero_point": False,
            "is_fp_zero_point": False,
        },
        input_config={"dtype": "float8e4m3", "group_size": 0},
        torch_dtype=torch.bfloat16,
    ).to(device)

    weight_generator = torch.Generator(device=device)
    weight_generator.manual_seed(100)
    unquantized_weight = torch.randn(
        (num_experts, shape_n, shape_k),
        generator=weight_generator,
        dtype=torch.bfloat16,
        device=device,
    )
    layer.load_from_unquantized(unquantized_weight)
    layer.transform()

    tuning_configs = get_heuristics_config(
        meta=layer.humming_metas[""],
        use_f16_accum=False,
        gemm_type=GemmType.INDEXED,
    )
    selected_large_config = _select_config(tuning_configs, large_m * top_k)
    assert selected_large_config["use_stream_k"] is True

    small_inputs = _make_inputs(small_m, shape_k, device, seed=200)
    small_topk_ids = _make_topk_ids(small_m, num_experts, top_k, device, seed=300)
    filler_inputs = _make_inputs(large_m - small_m, shape_k, device, seed=400)
    filler_topk_ids = _make_topk_ids(
        large_m - small_m,
        num_experts,
        top_k,
        device,
        seed=500,
    )
    large_inputs = torch.cat([small_inputs, filler_inputs], dim=0).contiguous()
    large_topk_ids = torch.cat([small_topk_ids, filler_topk_ids], dim=0).contiguous()

    small_output = _run_indexed_gemm(layer, small_inputs, small_topk_ids, tuning_configs)
    large_output = _run_indexed_gemm(layer, large_inputs, large_topk_ids, tuning_configs)
    torch.testing.assert_close(
        small_output.float(),
        large_output[: small_topk_ids.numel()].float(),
        atol=0.02,
        rtol=0.02,
    )
