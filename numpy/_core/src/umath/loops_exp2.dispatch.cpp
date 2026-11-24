#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "loops_explog.h"
#include "loops.h"

#if SIMD_ARM
#include "simd/simd.hpp"
#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;
#endif

#if SIMD_ARM

/* ---- Highway (Google Highway) AOR advsimd exp2 kernels -------------------
 * Algorithm ported from ARM optimized-routines math/aarch64/advsimd/exp2f.c
 * and exp2.c (MIT OR Apache-2.0 WITH LLVM-exception), expressed with Highway
 * scalable vectors following the official numpy SIMD mode (ScalableTag +
 * HWY_ATTR, LoadN/StoreN/GatherIndexN tails, GatherIndex table lookup).
 *
 * Copyright (c) 2019-2026, Arm Limited.
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 *
 * numpy contract adaptations (see PR #200 / commit c73289f):
 *  - NaN sign+payload passthrough; +-inf mapped to constants. NaN/inf lanes
 *    are sanitized to 0 before the polynomial so the raw AOR s1*s1 overflow
 *    term never raises a spurious OF/UF on them.
 *  - the s1*s2 bias-split special path keeps OF/UF raised naturally by the
 *    split arithmetic (no libm, no manual flag writes).
 *  - tiny-input product flooring so subnormal/near-zero remainders do not
 *    leak a spurious underflow flag (f32 |r|>=2^-21 even powers; f64
 *    |r|>=2^-511 for r^2 and |r|>=2^-1019 for the linear term).
 */
const hn::ScalableTag<float> exp2_f32;
const hn::ScalableTag<int32_t> exp2_s32;
const hn::ScalableTag<uint32_t> exp2_u32;
using vec_e_f32 = hn::Vec<decltype(exp2_f32)>;
using vec_e_s32 = hn::Vec<decltype(exp2_s32)>;
using vec_e_u32 = hn::Vec<decltype(exp2_u32)>;

const hn::ScalableTag<double> exp2_f64;
const hn::ScalableTag<int64_t> exp2_s64;
const hn::ScalableTag<uint64_t> exp2_u64;
using vec_e_f64 = hn::Vec<decltype(exp2_f64)>;
using vec_e_s64 = hn::Vec<decltype(exp2_s64)>;
using vec_e_u64 = hn::Vec<decltype(exp2_u64)>;

template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE vtype
exp2_load_vector(type_t *src, npy_intp ssrc, npy_intp len)
{
    auto D = hn::DFromV<vtype>();
    using DI = hn::RebindToSigned<decltype(D)>;
    DI di;
    auto indices = hn::Mul(hn::Iota(di, 0), hn::Set(di, ssrc));
    if (len < hn::Lanes(D)) {
        if (ssrc == 1) return hn::LoadN(D, src, len);
        return hn::GatherIndexN(D, src, indices, len);
    }
    if (ssrc == 1) return hn::LoadU(D, src);
    return hn::GatherIndex(D, src, indices);
}

template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE void
exp2_store_vector(vtype vec, type_t *dst, npy_intp sdst, npy_intp len)
{
    auto D = hn::DFromV<vtype>();
    using DI = hn::RebindToSigned<decltype(D)>;
    DI di;
    auto indices = hn::Mul(hn::Iota(di, 0), hn::Set(di, sdst));
    if (len < hn::Lanes(D)) {
        if (sdst == 1) { hn::StoreN(vec, D, dst, len); return; }
        hn::ScatterIndexN(vec, D, dst, indices, len);
        return;
    }
    if (sdst == 1) { hn::StoreU(vec, D, dst); return; }
    hn::ScatterIndex(vec, D, dst, indices);
}

HWY_ATTR static void
simd_exp2_hwy_FLOAT(const npy_float *src, npy_intp ssrc,
                    npy_float *dst, npy_intp sdst, npy_intp len)
{
    const auto D = exp2_f32;
    const int nlanes = hn::Lanes(D);
    const vec_e_f32 zero = hn::Zero(D);
    const vec_e_f32 one = hn::Set(D, 1.0f);
    const vec_e_f32 shift = hn::Set(D, 0x1.8p+23f);
    const vec_e_u32 bias = hn::Set(exp2_u32, 0x3f800000u);
    const vec_e_f32 c0 = hn::Set(D, 0x1.59977ap-10f);
    const vec_e_f32 c1 = hn::Set(D, 0x1.3ce9e4p-7f);
    const vec_e_f32 c2 = hn::Set(D, 0x1.c6bd32p-5f);
    const vec_e_f32 c3 = hn::Set(D, 0x1.ebf9bcp-3f);
    const vec_e_f32 c4 = hn::Set(D, 0x1.62e422p-1f);
    const vec_e_f32 r_floor_even = hn::Set(D, 0x1p-21f);
    /* expf_special: 2^n split into s1*s2; |n| > 192 overflows even split. */
    const vec_e_u32 sp_offset = hn::Set(exp2_u32, 0x82000000u);
    const vec_e_u32 sp_bias = hn::Set(exp2_u32, 0x7f000000u);
    const vec_e_f32 scale_bound = hn::Set(D, 0x1.8p+7f);
    const vec_e_u32 abs_mask = hn::Set(exp2_u32, 0x7fffffffu);
    const vec_e_u32 inf_bits = hn::Set(exp2_u32, 0x7f800000u);
    /* |x| >= 126.5 (AOR exp2f SpecialBound) as an unsigned abs-bits compare;
     * the same ordering also separates NaN/inf (exponent bits all ones). */
    const vec_e_u32 bound_bits = hn::Set(exp2_u32, 0x42fd0000u);

    for (; len > 0; len -= nlanes, src += ssrc * nlanes, dst += sdst * nlanes) {
        vec_e_f32 x = exp2_load_vector<vec_e_f32>(src, ssrc, len);
        vec_e_u32 abs_bits = hn::And(hn::BitCast(exp2_u32, x), abs_mask);
        auto naninf_m = hn::Ge(abs_bits, inf_bits);
        auto special_any = hn::Ge(abs_bits, bound_bits);
        /* Sanitize NaN/inf to 0 so the reduction never forms inf - inf
         * (spurious invalid); they are patched back to constants below. */
        vec_e_f32 xs = hn::IfThenElse(hn::RebindMask(D, naninf_m), zero, x);
        /* exp2(x) = 2^n (1 + poly(r)), x = n + r, r in [-1/2, 1/2]. */
        vec_e_f32 z = hn::Add(xs, shift);
        vec_e_f32 n = hn::Sub(z, shift);
        vec_e_f32 r = hn::Sub(xs, n);
        vec_e_u32 e = hn::ShiftLeft<23>(hn::BitCast(exp2_u32, hn::NearestInt(n)));
        vec_e_f32 scale = hn::BitCast(D, hn::Add(e, bias));
        /* Floor |r| for the even powers so a tiny remainder cannot square
         * into the subnormal range and raise a spurious underflow flag
         * (the near-multiple lattice makes an |x| mask incomplete). */
        vec_e_f32 r_even = hn::Max(hn::Abs(r), r_floor_even);
        vec_e_f32 r2 = hn::Mul(r_even, r_even);
        vec_e_f32 p = hn::MulAdd(r, c0, c1);
        vec_e_f32 q = hn::MulAdd(r, c2, c3);
        q = hn::MulAdd(p, r2, q);
        /* poly = c4*r + q*r2; the FMA addend q*r2 (>= ~2^-44 after the
         * even-power floor) keeps the result normal for tiny r. */
        vec_e_f32 poly = hn::MulAdd(r, c4, hn::Mul(q, r2));

        vec_e_f32 result;
        if (__builtin_expect(!hn::AllFalse(exp2_u32, special_any), 0)) {
            auto nan_m = hn::Gt(abs_bits, inf_bits);
            auto inf_m = hn::Eq(abs_bits, inf_bits);
            auto cmp = hn::And(special_any, hn::Lt(abs_bits, inf_bits));
            /* Sanitize the reconstruction scale on overflow lanes so the
             * main path never forms inf - inf (spurious invalid). */
            vec_e_f32 scale_r = hn::IfThenElse(hn::RebindMask(D, cmp), one, scale);
            result = hn::MulAdd(poly, scale_r, scale_r);
            if (__builtin_expect(!hn::AllFalse(exp2_u32, cmp), 0)) {
                /* s1*s2 bias split (AOR expf_special); OF/UF arise naturally.
                 * NaN/inf were sanitized (cmp false) so never reach s1*s1. */
                vec_e_u32 b = hn::IfThenElse(
                    hn::RebindMask(exp2_u32, hn::Le(n, zero)),
                    sp_offset, hn::Zero(exp2_u32));
                vec_e_f32 s1 = hn::BitCast(D, hn::Add(b, sp_bias));
                vec_e_f32 s2 = hn::BitCast(D, hn::Sub(e, b));
                auto cmp2 = hn::Gt(hn::Abs(n), scale_bound);
                /* s1*s1 overflows (n>0 -> s1=2^127) even on lanes where it
                 * is not selected; sanitize so only |n|>192 lanes flag. */
                vec_e_f32 s1_ov = hn::IfThenElse(cmp2, s1, one);
                vec_e_f32 r_ov = hn::Mul(s1_ov, s1_ov);
                vec_e_f32 r_sp = hn::Mul(hn::MulAdd(poly, s2, s2), s1);
                vec_e_f32 sp_val = hn::IfThenElse(cmp2, r_ov, r_sp);
                result = hn::IfThenElse(hn::RebindMask(D, cmp), sp_val, result);
            }
            /* NaN payload passthrough; +-inf -> constants (exp2(+inf)=+inf,
             * exp2(-inf)=+0) without the split's spurious OF/UF. */
            result = hn::IfThenElse(hn::RebindMask(D, nan_m), x, result);
            result = hn::IfThenElse(hn::RebindMask(D, inf_m), hn::Max(x, zero), result);
        } else {
            result = hn::MulAdd(poly, scale, scale);
        }
        exp2_store_vector<vec_e_f32>(result, dst, sdst, len);
    }
}

static inline float64x2_t __attribute__((always_inline))
neon_exp2_inline_core_f64(float64x2_t x,
                           float64x2_t shift,
                           float64x2_t poly0, float64x2_t poly1,
                           float64x2_t poly2, float64x2_t poly3)
{
  float64x2_t z = vaddq_f64(shift, x);
  uint64x2_t u = vreinterpretq_u64_f64(z);
  float64x2_t n = vsubq_f64(z, shift);
  float64x2_t r = vsubq_f64(x, n);
  uint64x2_t e = vshlq_n_u64(u, 52 - V_EXP_TABLE_BITS);
  uint64x2_t tab = neon_exp_lookup_sbits(u);
  float64x2_t scale = vreinterpretq_f64_u64(vaddq_u64(tab, e));
  float64x2_t r2 = vmulq_f64(r, r);
  float64x2_t p = vfmaq_f64(poly0, r, poly1);
  float64x2_t q = vfmaq_f64(poly2, r, poly3);
  /*
   * Estrin evaluation of the degree-4 Taylor series of 2^r - 1 with
   * {poly0..poly3} = {ln2, ln2^2/2, ln2^3/6, ln2^4/24}:
   *   poly0*r + poly1*r^2 + poly2*r^3 + poly3*r^4
   *     = r*(poly0 + r*poly1) + r^3*(poly2 + r*poly3)
   * The previous code multiplied q by r^2, which double-counted the r^2
   * term and dropped the r^4 one (~5.4e-7 relative error for any x that
   * is not a multiple of 1/128).
   */
  float64x2_t pv = vfmaq_f64(vmulq_f64(r, p), vmulq_f64(r2, r), q);
  return vfmaq_f64(scale, scale, pv);
}

static void
simd_exp2_neon_DOUBLE(const npy_double *src, npy_intp ssrc,
                      npy_double *dst, npy_intp sdst, npy_intp len)
{
  const float64x2_t special_bound = vdupq_n_f64(1022.0);
  /* Nonzero |x| < 2^-30: r == x there and the polynomial's r*r, r*p
   * and r^3 products (and the pv sum) would go subnormal, raising a
   * spurious underflow flag although exp2(x) = fl(1 + x*ln2) ~ 1.0.
   * libm returns that value with the scalar path's (no) flags; +-0
   * stays in-band (exp2(+-0) is exact).  Lanes with n != 0 keep
   * |r| >~ 2^-57, far above every product's subnormal boundary. */
  const float64x2_t tiny_bound = vdupq_n_f64(0x1p-30);
  const float64x2_t zero_val = vdupq_n_f64(0.0);
  const float64x2_t shift = vdupq_n_f64(0x1.8p+52 / V_EXP_N);
  const float64x2_t poly0 = vdupq_n_f64(0x1.62e42fefa3686p-1);
  const float64x2_t poly1 = vdupq_n_f64(0x1.ebfbdff82c241p-3);
  const float64x2_t poly2 = vdupq_n_f64(0x1.c6b09b16de99ap-5);
  const float64x2_t poly3 = vdupq_n_f64(0x1.3b2abf5571ad8p-7);
  const uint64x2_t all_bits = vdupq_n_u64(0xFFFFFFFFFFFFFFFFULL);

  const npy_double *infp = src;
  npy_double *outp = dst;
  npy_intp remaining = len;

  const npy_intp UNROLL = 8;
  const npy_intp VEC_SIZE = 2;

  while (remaining >= UNROLL * VEC_SIZE) {
    float64x2_t x[UNROLL], result[UNROLL];

    if (ssrc == 1) {
      for (int i = 0; i < UNROLL; i++)
        x[i] = vld1q_f64(infp + i * VEC_SIZE);
    } else {
      for (int i = 0; i < UNROLL; i++) {
        double vals[VEC_SIZE];
        for (int j = 0; j < VEC_SIZE; j++)
          vals[j] = infp[(i * VEC_SIZE + j) * ssrc];
        x[i] = vld1q_f64(vals);
      }
    }

    const float64x2_t safe_val = vdupq_n_f64(0.0);
    uint64x2_t any_libm = vdupq_n_u64(0);
    uint64x2_t need_libm[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      /* |x| > 1022 (incl. +-inf) or NaN: the scale bit-trick behind the
       * polynomial breaks down across the overflow/subnormal boundaries;
       * evaluate those lanes via libm, which also raises the overflow/
       * underflow FP flags like the scalar path does. */
      uint64x2_t is_nan = veorq_u64(vceqq_f64(x[i], x[i]), all_bits);
      /* mask NaN lanes out of the inequality compare (FCAGT raises IOC on
       * QNaN lanes, unlike vceqq) */
      float64x2_t x_nn = vbslq_f64(is_nan, vdupq_n_f64(0.0), x[i]);
      need_libm[i] =
          vorrq_u64(vcagtq_f64(x_nn, special_bound), is_nan);
      need_libm[i] = vorrq_u64(
          need_libm[i],
          vbicq_u64(vcltq_f64(vabsq_f64(x_nn), tiny_bound),
                    vceqq_f64(x_nn, zero_val)));
      any_libm = vorrq_u64(any_libm, need_libm[i]);
      /* sanitized polynomial evaluation (speculation-safe, see the
       * FLOAT kernel) */
      float64x2_t xs = vbslq_f64(need_libm[i], safe_val, x[i]);
      result[i] = neon_exp2_inline_core_f64(xs, shift, poly0, poly1, poly2, poly3);
    }
    /* decide once per block: the common (all-in-window) case stays
     * branch-free, see the FLOAT kernel */
    if (__builtin_expect(neon_has_any_lane_u64(any_libm), 0)) {
      for (int i = 0; i < UNROLL; i++) {
        if (__builtin_expect(neon_has_any_lane_u64(need_libm[i]), 0)) {
          double vals[2], outs[2];
          uint64_t m[2];
          vst1q_f64(vals, x[i]);
          vst1q_f64(outs, result[i]);
          vst1q_u64(m, need_libm[i]);
          for (int j = 0; j < 2; j++)
            if (m[j]) outs[j] = npy_exp2(vals[j]);
          result[i] = vld1q_f64(outs);
        }
      }
    }

    if (sdst == 1) {
      for (int i = 0; i < UNROLL; i++)
        vst1q_f64(outp + i * VEC_SIZE, result[i]);
    } else {
      for (int j = 0; j < UNROLL; j++) {
        for (int i = 0; i < VEC_SIZE; i++)
          vst1q_lane_f64(outp + (i + j * VEC_SIZE) * sdst, result[j], i);
      }
    }

    infp += ssrc * UNROLL * VEC_SIZE;
    outp += sdst * UNROLL * VEC_SIZE;
    remaining -= UNROLL * VEC_SIZE;
  }

  while (remaining > 0) {
    int current_len = (remaining < VEC_SIZE) ? remaining : VEC_SIZE;
    float64x2_t x;
    if (ssrc == 1) {
      /* broadcast the single valid lane: loading the full vector would
       * read one element past the buffer (and its garbage bits must
       * never reach the polynomial) */
      if (current_len == 1) x = vld1q_dup_f64(infp);
      else x = vld1q_f64(infp);
    } else {
      double vals[2] = {0.0, 0.0};
      for (int i = 0; i < current_len; i++) vals[i] = infp[i * ssrc];
      x = vld1q_f64(vals);
    }

    uint64x2_t is_nan = veorq_u64(vceqq_f64(x, x), all_bits);
    /* mask NaN lanes out of the inequality compare (FCAGT raises IOC on
     * QNaN lanes, unlike vceqq) */
    float64x2_t x_nn = vbslq_f64(is_nan, vdupq_n_f64(0.0), x);
    uint64x2_t need_libm =
        vorrq_u64(vcagtq_f64(x_nn, special_bound), is_nan);
    need_libm = vorrq_u64(
        need_libm,
        vbicq_u64(vcltq_f64(vabsq_f64(x_nn), tiny_bound),
                  vceqq_f64(x_nn, zero_val)));

    /* sanitized polynomial evaluation (see the main loop) */
    const float64x2_t safe_val = vdupq_n_f64(0.0);
    float64x2_t xs = vbslq_f64(need_libm, safe_val, x);
    float64x2_t result = neon_exp2_inline_core_f64(xs, shift, poly0, poly1, poly2, poly3);
    if (__builtin_expect(neon_has_any_lane_u64(need_libm), 0)) {
      double vals[2], outs[2];
      uint64_t m[2];
      vst1q_f64(vals, x);
      vst1q_f64(outs, result);
      vst1q_u64(m, need_libm);
      for (int j = 0; j < current_len; j++)
        if (m[j]) outs[j] = npy_exp2(vals[j]);
      result = vld1q_f64(outs);
    }

    if (sdst == 1) {
      if (current_len == 1) vst1q_lane_f64(outp, result, 0);
      else vst1q_f64(outp, result);
    } else {
      for (int i = 0; i < current_len; i++)
        vst1q_lane_f64(outp + i * sdst, result, i);
    }

    infp += ssrc * current_len;
    outp += sdst * current_len;
    remaining -= current_len;
  }
}

static void __attribute__((optimize("O3")))
simd_exp2_neon_HALF(const npy_half *src, npy_intp ssrc,
                     npy_half *dst, npy_intp sdst, npy_intp len)
{
  const npy_intp LARGE_BLOCK = 8192;
  const npy_intp SMALL_BLOCK = 256;
  const npy_intp VEC_SIZE = 8;
  const npy_intp UNROLL = 4;
  
  npy_float tmp_src_small[SMALL_BLOCK];
  npy_float tmp_dst_small[SMALL_BLOCK];
  
  npy_float *tmp_src = NULL;
  npy_float *tmp_dst = NULL;
  
  if (len > SMALL_BLOCK) {
    tmp_src = (npy_float *)malloc(LARGE_BLOCK * sizeof(npy_float));
    tmp_dst = (npy_float *)malloc(LARGE_BLOCK * sizeof(npy_float));
    if (!tmp_src || !tmp_dst) {
      if (tmp_src) free(tmp_src);
      if (tmp_dst) free(tmp_dst);
      for (npy_intp i = 0; i < len; i++) {
        npy_float in1 = npy_half_to_float(src[i * ssrc]);
        dst[i * sdst] = npy_float_to_half(npy_exp2f(in1));
      }
      return;
    }
  }
  
  const npy_intp BLOCK_SIZE = (len > SMALL_BLOCK) ? LARGE_BLOCK : SMALL_BLOCK;
  npy_float *src_buf = (len > SMALL_BLOCK) ? tmp_src : tmp_src_small;
  npy_float *dst_buf = (len > SMALL_BLOCK) ? tmp_dst : tmp_dst_small;

  for (npy_intp offset = 0; offset < len; offset += BLOCK_SIZE) {
    npy_intp block_len = (len - offset > BLOCK_SIZE) ? BLOCK_SIZE : (len - offset);
    const npy_half *block_src = src + offset * ssrc;
    npy_half *block_dst = dst + offset * sdst;
    
    npy_intp i = 0;
    if (ssrc == 1) {
      for (; i + UNROLL * VEC_SIZE <= block_len; i += UNROLL * VEC_SIZE) {
        float16x8_t h[4];
        for (int u = 0; u < 4; u++)
          h[u] = vld1q_f16((const float16_t *)(block_src + i + u * 8));
        for (int u = 0; u < 4; u++) {
          vst1q_f32(src_buf + i + u * 8, vcvt_f32_f16(vget_low_f16(h[u])));
          vst1q_f32(src_buf + i + u * 8 + 4, vcvt_f32_f16(vget_high_f16(h[u])));
        }
      }
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
        float16x8_t h0 = vld1q_f16((const float16_t *)(block_src + i));
        vst1q_f32(src_buf + i, vcvt_f32_f16(vget_low_f16(h0)));
        vst1q_f32(src_buf + i + 4, vcvt_f32_f16(vget_high_f16(h0)));
      }
    } else {
      for (; i + 8 <= block_len; i += 8) {
        for (int j = 0; j < 8; j++) {
          src_buf[i + j] = npy_half_to_float(block_src[(i + j) * ssrc]);
        }
      }
    }
    for (; i < block_len; i++) {
      src_buf[i] = npy_half_to_float(block_src[i * ssrc]);
    }
    
    simd_exp2_hwy_FLOAT(src_buf, 1, dst_buf, 1, block_len);
    
    i = 0;
    /* The vectorized f32->f16 narrowing conversion raises OFC/UFC for
     * results that overflow or go subnormal in half precision, but these
     * are legitimate: the scalar path converts through npy_float_to_half,
     * which uses the same hardware fcvt on aarch64 (and a software
     * conversion that explicitly generates the same flags elsewhere), so
     * the flags must flow to the caller unmodified for parity. No
     * save/clear/restore is wanted here. */
    if (sdst == 1) {
      for (; i + UNROLL * VEC_SIZE <= block_len; i += UNROLL * VEC_SIZE) {
        for (int u = 0; u < 4; u++) {
          float16x8_t hv = vcombine_f16(vcvt_f16_f32(vld1q_f32(dst_buf + i + u * 8)),
                                         vcvt_f16_f32(vld1q_f32(dst_buf + i + u * 8 + 4)));
          vst1q_f16((float16_t *)(block_dst + i + u * 8), hv);
        }
      }
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
        float16x8_t h_combined = vcombine_f16(vcvt_f16_f32(vld1q_f32(dst_buf + i)), vcvt_f16_f32(vld1q_f32(dst_buf + i + 4)));
        vst1q_f16((float16_t *)(block_dst + i), h_combined);
      }
    } else {
      for (; i + 8 <= block_len; i += 8) {
        for (int j = 0; j < 8; j++) {
          block_dst[(i + j) * sdst] = npy_float_to_half(dst_buf[i + j]);
        }
      }
    }
    for (; i < block_len; i++) {
      block_dst[i * sdst] = npy_float_to_half(dst_buf[i]);
    }
  }
  
  if (len > SMALL_BLOCK) {
    free(tmp_src);
    free(tmp_dst);
  }
}

#endif // SIMD_ARM

#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
static void
simd_exp2_f32(const npyv_lanetype_f32 *src, npy_intp ssrc,
              npyv_lanetype_f32 *dst, npy_intp sdst, npy_intp len)
{
    const int vstep = npyv_nlanes_f32;
    for (; len > 0; len -= vstep, src += ssrc*vstep, dst += sdst*vstep) {
        npyv_f32 x;
        if (ssrc == 1) {
            x = npyv_load_tillz_f32(src, len);
        } else {
            x = npyv_loadn_tillz_f32(src, ssrc, len);
        }
        npyv_f32 out = __svml_exp2f16(x);
        if (sdst == 1) {
            npyv_store_till_f32(dst, len, out);
        } else {
            npyv_storen_till_f32(dst, sdst, len, out);
        }
    }
    npyv_cleanup();
}

static void
simd_exp2_f64(const npyv_lanetype_f64 *src, npy_intp ssrc,
              npyv_lanetype_f64 *dst, npy_intp sdst, npy_intp len)
{
    const int vstep = npyv_nlanes_f64;
    for (; len > 0; len -= vstep, src += ssrc*vstep, dst += sdst*vstep) {
        npyv_f64 x;
        if (ssrc == 1) {
            x = npyv_load_tillz_f64(src, len);
        } else {
            x = npyv_loadn_tillz_f64(src, ssrc, len);
        }
        npyv_f64 out = __svml_exp28_ha(x);
        if (sdst == 1) {
            npyv_store_till_f64(dst, len, out);
        } else {
            npyv_storen_till_f64(dst, sdst, len, out);
        }
    }
    npyv_cleanup();
}
#endif
// Trigonometric Half
#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
  #define NPY__SVML_IS_ENABLED 1
#else
  #define NPY__SVML_IS_ENABLED 0
#endif

#if NPY__SVML_IS_ENABLED && !defined(NPY_HAVE_AVX512_SPR)
typedef __m256i npyvh_f16;
#define npyv_cvt_f16_f32 _mm512_cvtph_ps
#define npyv_cvt_f32_f16 _mm512_cvtps_ph
NPY_FINLINE npyvh_f16 npyvh_load_f16(const void *ptr) {
    return _mm256_loadu_si256((const __m256i*)(ptr));
}
NPY_FINLINE void npyvh_store_f16(void *ptr, npyvh_f16 data) {
    _mm256_storeu_si256((__m256i*)ptr, data);
}
NPY_FINLINE npyvh_f16 npyvh_load_till_f16(const npy_half *ptr, npy_uintp nlane, npy_half fill)
{
    assert(nlane > 0);
    const __m256i vfill = _mm256_set1_epi16(fill);
    const __mmask16 mask = (0x0001 << nlane) - 0x0001;
    return _mm256_mask_loadu_epi16(vfill, mask, ptr);
}
NPY_FINLINE void npyvh_store_till_f16(npy_half *ptr, npy_uintp nlane, npyvh_f16 data)
{
    assert(nlane > 0);
    const __mmask16 mask = (0x0001 << nlane) - 0x0001;
    _mm256_mask_storeu_epi16(ptr, mask, data);
}

template<typename Func>

static void avx512_exponent_f16(const npy_half *src, npy_half *dst, npy_intp len, Func math_func, npy_uintp val)
{
    const int num_lanes = npyv_nlanes_f32;
    npyvh_f16 x, out;
    npyv_f32 x_ps, out_ps;
    
    for (; len > 0; len -= num_lanes, src += num_lanes, dst += num_lanes) {
        if (len >= num_lanes) {
            x       = npyvh_load_f16(src);
            x_ps    = npyv_cvt_f16_f32(x);
            out_ps  = math_func(x_ps); 
            out     = npyv_cvt_f32_f16(out_ps, 0);
            npyvh_store_f16(dst, out);
        }
        else {
            x       = npyvh_load_till_f16(src, len, val);
            x_ps    = npyv_cvt_f16_f32(x);
            out_ps  = math_func(x_ps);
            out     = npyv_cvt_f32_f16(out_ps, 0);
            npyvh_store_till_f16(dst, len, out);
        }
    }
    npyv_cleanup();
}

static void avx512_exp2_f16(const npy_half *src, npy_half *dst, npy_intp len)
{
    avx512_exponent_f16(src, dst, len, __svml_exp2f16, 0);
}
#endif

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_exp2)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_float in1 = *(npy_float *)ip1;
            *(npy_float *)op1 = npy_exp2f(in1);
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_float *src = (npy_float*)args[0];
        npy_float *dst = (npy_float*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);
        if (ssrc <= 2) {
            simd_exp2_hwy_FLOAT(src, ssrc, dst, sdst, len);
        } else {
            const npy_float * __restrict ip = src;
            npy_float * __restrict op = dst;
            for (npy_intp i = 0; i < len; i++) {
                *op = npy_exp2f(*ip);
                ip += ssrc;
                op += sdst;
            }
        }
    }
    else {
        UNARY_LOOP {
            simd_exp2_hwy_FLOAT((npy_float *)ip1, 1, (npy_float *)op1, 1, 1);
        }
    }
    return;
#elif NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
    const npy_float *src = (npy_float*)args[0];
          npy_float *dst = (npy_float*)args[1];
    const npy_intp len = dimensions[0];

    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        npyv_loadable_stride_f32(steps[0]) &&
        npyv_storable_stride_f32(steps[1]))
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);
        simd_exp2_f32(src, ssrc, dst, sdst, len);
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = npy_exp2f(in1);
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_exp2)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_double in1 = *(npy_double *)ip1;
            *(npy_double *)op1 = npy_exp2(in1);
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_double *src = (npy_double*)args[0];
        npy_double *dst = (npy_double*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_double);
        const npy_intp sdst = steps[1] / sizeof(npy_double);
        if (ssrc <= 2) {
            simd_exp2_neon_DOUBLE(src, ssrc, dst, sdst, len);
        } else {
            const npy_double * __restrict ip = src;
            npy_double * __restrict op = dst;
            for (npy_intp i = 0; i < len; i++) {
                *op = npy_exp2(*ip);
                ip += ssrc;
                op += sdst;
            }
        }
    }
    else {
        UNARY_LOOP {
            simd_exp2_neon_DOUBLE((npy_double *)ip1, 1, (npy_double *)op1, 1, 1);
        }
    }
    return;
#elif NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
    const npy_double *src = (npy_double*)args[0];
          npy_double *dst = (npy_double*)args[1];
    const npy_intp len = dimensions[0];

    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        npyv_loadable_stride_f64(steps[0]) &&
        npyv_storable_stride_f64(steps[1]))
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_double);
        const npy_intp sdst = steps[1] / sizeof(npy_double);
        simd_exp2_f64(src, ssrc, dst, sdst, len);
        return;
    }
#endif
    UNARY_LOOP {
        const npy_double in1 = *(npy_double *)ip1;
        *(npy_double *)op1 = npy_exp2(in1);
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_exp2)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_exp2f(in1));
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_half *src = (npy_half*)args[0];
        npy_half *dst = (npy_half*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_half);
        const npy_intp sdst = steps[1] / sizeof(npy_half);
        simd_exp2_neon_HALF(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_exp2f(in1));
        }
    }
    return;
#elif NPY__SVML_IS_ENABLED
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];

    const npy_intp len = dimensions[0];

    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        (steps[0] == sizeof(npy_half)) &&
        (steps[1] == sizeof(npy_half))) {
    #ifdef NPY_HAVE_AVX512_SPR
        __svml_exp2s32(src, dst, len);
    #else
        avx512_exp2_f16(src, dst, len);
    #endif
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_exp2f(in1));
    }
}
