// Copyright (c) 2026 FlashInfer team.
// Copyright (c) 2026 Humming contributors.
// SPDX-License-Identifier: Apache-2.0
//
// The E2M1-to-E4M3 LUT promotion follows the generic residual decoder from
// FlashInfer PR 4589, adapted to Humming's register-packed WGMMA B fragments.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>

#include <humming/utils/all.cuh>


struct Nvfp4W4A8DecodeLut {
  uint32_t low;
  uint32_t high;
  uint32_t residual_sign;
};


CUDA_INLINE uint8_t nvfp4_encode_e4m3(float value) {
  const float finite = fminf(fmaxf(value, -448.0f), 448.0f);
  const __nv_fp8_e4m3 encoded(finite);
  return static_cast<uint8_t>(encoded.__x);
}


CUDA_INLINE uint32_t nvfp4_pack_bytes(uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3) {
  return static_cast<uint32_t>(x0) |
      (static_cast<uint32_t>(x1) << 8) |
      (static_cast<uint32_t>(x2) << 16) |
      (static_cast<uint32_t>(x3) << 24);
}


CUDA_INLINE Nvfp4W4A8DecodeLut make_nvfp4_w4a8_decode_lut(__nv_bfloat16 residual) {
  const uint16_t residual_bits = __bfloat16_as_ushort(residual);
  const uint32_t residual_sign = (static_cast<uint32_t>(residual_bits) & 0x8000u) >> 8;
  const float magnitude = fabsf(__bfloat162float(residual));
  const uint32_t low = nvfp4_pack_bytes(
      0,
      nvfp4_encode_e4m3(__fmul_rn(magnitude, 0.5f)),
      nvfp4_encode_e4m3(magnitude),
      nvfp4_encode_e4m3(__fmul_rn(magnitude, 1.5f)));
  const uint32_t high = nvfp4_pack_bytes(
      nvfp4_encode_e4m3(__fmul_rn(magnitude, 2.0f)),
      nvfp4_encode_e4m3(__fmul_rn(magnitude, 3.0f)),
      nvfp4_encode_e4m3(__fmul_rn(magnitude, 4.0f)),
      nvfp4_encode_e4m3(__fmul_rn(magnitude, 6.0f)));
  return {low, high, residual_sign * 0x01010101u};
}


CUDA_INLINE uint32_t nvfp4_prmt_lookup(uint32_t low, uint32_t high, uint32_t selectors) {
  uint32_t result;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(result) : "r"(low), "r"(high), "r"(selectors));
  return result;
}


CUDA_INLINE uint32_t promote_nvfp4_nibbles(
    uint32_t packed, __nv_bfloat16 residual, bool high_nibbles) {
  const Nvfp4W4A8DecodeLut lut = make_nvfp4_w4a8_decode_lut(residual);
  uint32_t selectors;
  uint32_t signs;
  if (high_nibbles) {
    selectors = (packed >> 4) & 0x07070707u;
    signs = packed & 0x80808080u;
  } else {
    selectors = packed & 0x07070707u;
    signs = (packed & 0x08080808u) << 4;
  }
  return nvfp4_prmt_lookup(lut.low, lut.high, selectors) ^ signs ^ lut.residual_sign;
}


CUDA_INLINE void dequant_nvfp4_w4a8(
    const uint32_t *packed_weight,
    uint32_t *promoted_weight,
    uint32_t fragment_index,
    const uint32_t *promotion_scales,
    uint32_t scale_group_stride) {
  const uint32_t *group0_scales = promotion_scales + fragment_index * 2;
  const uint32_t *group1_scales = promotion_scales + scale_group_stride + fragment_index * 2;
  const __nv_bfloat16 residual00 = __ushort_as_bfloat16(group0_scales[0] & 0xffffu);
  const __nv_bfloat16 residual01 = __ushort_as_bfloat16(group0_scales[1] & 0xffffu);
  const __nv_bfloat16 residual10 = __ushort_as_bfloat16(group1_scales[0] & 0xffffu);
  const __nv_bfloat16 residual11 = __ushort_as_bfloat16(group1_scales[1] & 0xffffu);
  const uint32_t packed0 = packed_weight[fragment_index * 2];
  const uint32_t packed1 = packed_weight[fragment_index * 2 + 1];

  // Each packed word holds one N row: low nibbles are the first K16 group and
  // high nibbles are the second. Match dequant_b1248's high/low output order.
  promoted_weight[0] = promote_nvfp4_nibbles(packed0, residual10, true);
  promoted_weight[1] = promote_nvfp4_nibbles(packed0, residual00, false);
  promoted_weight[2] = promote_nvfp4_nibbles(packed1, residual11, true);
  promoted_weight[3] = promote_nvfp4_nibbles(packed1, residual01, false);
}
