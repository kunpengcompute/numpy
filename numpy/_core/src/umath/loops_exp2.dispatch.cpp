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

        auto cmp = hn::And(special_any, hn::Lt(abs_bits, inf_bits));
        vec_e_f32 result;
        /* Scalable (SVE) compilers hoist the clean-path fma(poly,scale,scale)
         * above the branch and run it unpredicated; with scale==inf (e.g.
         * exp2(127.5)) and poly<0 that forms inf-inf and raises a spurious
         * invalid. Form the default result with scale_r==1 on overflow lanes
         * BEFORE the branch so the hoisted fma is flag-safe; the special path
         * overrides those lanes. Fixed-width (NEON) keeps the lean branch
         * structure (no hoist, no extra select on the clean path). */
        if constexpr (HWY_HAVE_SCALABLE) {
            vec_e_f32 scale_r = hn::IfThenElse(hn::RebindMask(D, cmp), one, scale);
            result = hn::MulAdd(poly, scale_r, scale_r);
        }
        if (__builtin_expect(!hn::AllFalse(exp2_u32, special_any), 0)) {
            auto nan_m = hn::Gt(abs_bits, inf_bits);
            auto inf_m = hn::Eq(abs_bits, inf_bits);
            if constexpr (!HWY_HAVE_SCALABLE) {
                vec_e_f32 scale_r = hn::IfThenElse(hn::RebindMask(D, cmp), one, scale);
                result = hn::MulAdd(poly, scale_r, scale_r);
            }
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
            if constexpr (!HWY_HAVE_SCALABLE) {
                result = hn::MulAdd(poly, scale, scale);
            }
        }
        exp2_store_vector<vec_e_f32>(result, dst, sdst, len);
    }
}

/* Per-lane sbits lookup. AOR indexes __v_exp_data[i & 127] per lane; on the
 * 2-lane NEON f64 target an explicit ExtractLane + scalar load + InsertLane
 * lowers to the same fmov/ldr/ins sequence as AOR lookup_sbits without the
 * stack spills hn::GatherIndex's generic lowering can introduce. Wider
 * targets fall back to GatherIndex. */
HWY_ATTR NPY_FINLINE vec_e_f64
exp2_lookup_sbits(vec_e_u64 idx)
{
    if constexpr (hn::MaxLanes(exp2_f64) == 2) {
        uint64_t i0 = hn::ExtractLane(idx, 0);
        uint64_t i1 = hn::ExtractLane(idx, 1);
        /* table holds bit patterns: insert as u64 then BitCast (a numeric
         * (double) cast would convert the integer, not reinterpret it). */
        vec_e_u64 t = hn::InsertLane(hn::Zero(exp2_u64), 0, neon_exp_table[i0]);
        t = hn::InsertLane(t, 1, neon_exp_table[i1]);
        return hn::BitCast(exp2_f64, t);
    } else {
        return hn::GatherIndex(exp2_f64, (const double *)neon_exp_table,
                               hn::BitCast(exp2_s64, idx));
    }
}

HWY_ATTR static void
simd_exp2_hwy_DOUBLE(const npy_double *src, npy_intp ssrc,
                     npy_double *dst, npy_intp sdst, npy_intp len)
{
    const auto D = exp2_f64;
    const int nlanes = hn::Lanes(D);
    const vec_e_f64 zero = hn::Zero(D);
    const vec_e_f64 one = hn::Set(D, 1.0);
    const vec_e_f64 shift = hn::Set(D, 0x1.8p+52 / V_EXP_N);
    const vec_e_f64 poly0 = hn::Set(D, 0x1.62e42fefa3686p-1);
    const vec_e_f64 poly1 = hn::Set(D, 0x1.ebfbdff82c241p-3);
    const vec_e_f64 poly2 = hn::Set(D, 0x1.c6b09b16de99ap-5);
    const vec_e_f64 poly3 = hn::Set(D, 0x1.3b2abf5571ad8p-7);
    const vec_e_f64 ln2 = hn::Set(D, 0x1.62e42fefa39efp-1);
    const vec_e_u64 abs_mask = hn::Set(exp2_u64, 0x7fffffffffffffffull);
    const vec_e_u64 inf_bits = hn::Set(exp2_u64, 0x7ff0000000000000ull);
    const vec_e_u64 bound_bits = hn::Set(exp2_u64, 0x408ff00000000000ull); /*1022*/
    const vec_e_u64 tiny_bits = hn::Set(exp2_u64, 0x3e10000000000000ull); /*2^-30*/
    const vec_e_u64 idx_mask = hn::Set(exp2_u64, V_EXP_INDEX_MASK);

    for (; len > 0; len -= nlanes, src += ssrc * nlanes, dst += sdst * nlanes) {
        vec_e_f64 x = exp2_load_vector<vec_e_f64>(src, ssrc, len);
        vec_e_u64 abs_bits = hn::And(hn::BitCast(exp2_u64, x), abs_mask);
        auto special_m = hn::Gt(abs_bits, bound_bits);   /* |x|>1022, NaN, inf */
        auto tiny_m = hn::And(hn::Lt(abs_bits, tiny_bits),
                              hn::Gt(abs_bits, hn::Zero(exp2_u64)));  /* nonzero |x|<2^-30 */
        vec_e_f64 result;
        if (__builtin_expect(!hn::AllFalse(exp2_u64, hn::Or(special_m, tiny_m)), 0)) {
            auto nan_m = hn::Gt(abs_bits, inf_bits);
            auto inf_m = hn::Eq(abs_bits, inf_bits);
            auto cmp_f = hn::And(special_m, hn::Lt(abs_bits, inf_bits));
            auto care_m = hn::Or(hn::Or(nan_m, inf_m), tiny_m);
            vec_e_f64 xs = hn::IfThenElse(hn::RebindMask(D, care_m), zero, x);
            vec_e_f64 z = hn::Add(xs, shift);
            vec_e_u64 u = hn::BitCast(exp2_u64, z);
            vec_e_f64 n = hn::Sub(z, shift);
            vec_e_f64 r = hn::Sub(xs, n);
            vec_e_u64 e = hn::ShiftLeft<52 - V_EXP_TABLE_BITS>(u);
            vec_e_f64 tab = exp2_lookup_sbits(hn::And(u, idx_mask));
            vec_e_f64 scale = hn::BitCast(D, hn::Add(hn::BitCast(exp2_u64, tab), e));
            vec_e_f64 r2 = hn::Mul(r, r);
            vec_e_f64 p01 = hn::MulAdd(poly1, r, poly0);
            vec_e_f64 p23 = hn::MulAdd(poly3, r, poly2);
            vec_e_f64 poly = hn::Mul(r, hn::MulAdd(p23, r2, p01));
            vec_e_f64 scale_r = hn::IfThenElse(hn::RebindMask(D, cmp_f), one, scale);
            result = hn::MulAdd(poly, scale_r, scale_r);
            if (__builtin_expect(!hn::AllFalse(exp2_u64, cmp_f), 0)) {
                const vec_e_u64 sp_offset = hn::Set(exp2_u64, 0x6000000000000000ull);
                const vec_e_u64 sp_bias1 = hn::Set(exp2_u64, 0x7000000000000000ull);
                const vec_e_u64 sp_bias2 = hn::Set(exp2_u64, 0x3010000000000000ull);
                const vec_e_f64 scale_bound = hn::Set(D, 0x1.4p+10);
                vec_e_u64 b = hn::IfThenElse(
                    hn::RebindMask(exp2_u64, hn::Le(n, zero)),
                    sp_offset, hn::Zero(exp2_u64));
                vec_e_f64 s1 = hn::BitCast(D, hn::Sub(sp_bias1, b));
                vec_e_f64 s2 = hn::BitCast(D, hn::Add(
                    hn::Sub(hn::BitCast(exp2_u64, scale), sp_bias2), b));
                auto cmp2 = hn::Gt(hn::Abs(n), scale_bound);
                auto cmp_neg = hn::And(hn::RebindMask(D, cmp_f), hn::Le(n, zero));
                vec_e_f64 s1_ov = hn::IfThenElse(cmp2, s1, one);
                vec_e_f64 r_ov = hn::Mul(s1_ov, s1_ov);
                vec_e_f64 r_sp = hn::Mul(hn::MulAdd(poly, s2, s2), s1);
                vec_e_f64 sp_val = hn::IfThenElse(cmp2, r_ov, r_sp);
                result = hn::IfThenElse(hn::RebindMask(D, cmp_f), sp_val, result);
                /* SVE predicates the s1_ov*s1_ov product by cmp2 (its
                 * downstream select), so subnormal-result lanes (cmp_neg,
                 * cmp2 false) never execute it and would miss the UF flag
                 * scalar libm raises for exact power-of-2 subnormals such as
                 * exp2(-1030). Force an underflowing product on exactly those
                 * lanes (predicated by cmp_neg); its rounded value is 0, so
                 * adding it leaves the result unchanged. */
                vec_e_f64 s1_uf = hn::IfThenElse(cmp_neg, s1, one);
                vec_e_f64 uf = hn::Mul(s1_uf, s1_uf);
                result = hn::Add(result, hn::IfThenElse(cmp_neg, uf, zero));
            }
            /* exp2(x) = fl(1 + x*ln2) for |x|<2^-30; the FMA rounds once
             * so a subnormal x*ln2 intermediate never flags UF (result 1.0). */
            vec_e_f64 tiny_res = hn::MulAdd(x, ln2, one);
            result = hn::IfThenElse(hn::RebindMask(D, tiny_m), tiny_res, result);
            result = hn::IfThenElse(hn::RebindMask(D, nan_m), x, result);
            result = hn::IfThenElse(hn::RebindMask(D, inf_m), hn::Max(x, zero), result);
        } else {
            /* ---- clean fast path: pure AOR (tiny lanes routed away) ---- */
            vec_e_f64 z = hn::Add(x, shift);
            vec_e_u64 u = hn::BitCast(exp2_u64, z);
            vec_e_f64 n = hn::Sub(z, shift);
            vec_e_f64 r = hn::Sub(x, n);
            vec_e_u64 e = hn::ShiftLeft<52 - V_EXP_TABLE_BITS>(u);
            vec_e_f64 tab = exp2_lookup_sbits(hn::And(u, idx_mask));
            vec_e_f64 scale = hn::BitCast(D, hn::Add(hn::BitCast(exp2_u64, tab), e));
            vec_e_f64 r2 = hn::Mul(r, r);
            vec_e_f64 p01 = hn::MulAdd(poly1, r, poly0);
            vec_e_f64 p23 = hn::MulAdd(poly3, r, poly2);
            vec_e_f64 poly = hn::Mul(r, hn::MulAdd(p23, r2, p01));
            result = hn::MulAdd(poly, scale, scale);
        }
        exp2_store_vector<vec_e_f64>(result, dst, sdst, len);
    }
}static void __attribute__((optimize("O3")))
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
            simd_exp2_hwy_DOUBLE(src, ssrc, dst, sdst, len);
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
            simd_exp2_hwy_DOUBLE((npy_double *)ip1, 1, (npy_double *)op1, 1, 1);
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
