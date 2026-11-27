#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "loops_explog.h"
#include "loops.h"

/*
 * exp f16 inner loop, dispatched separately from exp f32/f64
 * (loops_exp.dispatch.cpp) so the two can use different target lists:
 * f16 keeps the SVE target (SVE f16 is ~2.5-3x faster than the NEON
 * promote/demote path on Kunpeng), while exp f64 is NEON-only full-stride
 * (the Kunpeng 920 SVE gather bottleneck regressed strided f64 runs).
 * Meson dispatch granularity is per file, hence the split.
 *
 * ARM: Highway f32 kernel on promoted f16 halves (full-stride wrapper:
 * stride 1 keeps the LoadU/StoreU fast path, other element-aligned strides
 * stage through a contiguous stack temp — no gather hardware, SVE kept).
 * x86: SVML f16 path unchanged.
 */

#if SIMD_ARM
#include "simd/simd.hpp"
#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;
#include "loops_exp_hwy.h"

/* f32 kernel used on the promoted halves of each f16 vector (the f16
 * result is the f32 exp demoted back to f16, round-to-nearest-even).
 * Algorithm ported from ARM optimized-routines math/aarch64/advsimd/expf.c
 * with the numpy contract adaptations (NaN sign+payload passthrough,
 * +-inf constants, sanitized reduction, bias-split OF/UF parity). */
const hn::ScalableTag<float> exp_f32;
const hn::ScalableTag<uint32_t> exp_u32;
using vec_x_f32 = hn::Vec<decltype(exp_f32)>;
using vec_x_u32 = hn::Vec<decltype(exp_u32)>;

HWY_ATTR NPY_FINLINE vec_x_f32
exp_f32_kernel(vec_x_f32 x)
{
    const auto D = exp_f32;
    const vec_x_f32 zero = hn::Zero(D);
    const vec_x_f32 one = hn::Set(D, 1.0f);
    const vec_x_f32 shift = hn::Set(D, 0x1.8p+23f);
    const vec_x_u32 bias = hn::Set(exp_u32, 0x3f800000u);
    const vec_x_f32 inv_ln2 = hn::Set(D, 0x1.715476p+0f);
    const vec_x_f32 ln2_hi = hn::Set(D, -0x1.62e4p-1f);
    const vec_x_f32 ln2_lo = hn::Set(D, -0x1.7f7d1cp-20f);
    const vec_x_f32 c0 = hn::Set(D, 0x1.0e4020p-7f);
    const vec_x_f32 c1 = hn::Set(D, 0x1.573e2ep-5f);
    const vec_x_f32 c2 = hn::Set(D, 0x1.555e66p-3f);
    const vec_x_f32 c3 = hn::Set(D, 0x1.fffdb6p-2f);
    const vec_x_f32 c4 = hn::Set(D, 0x1.ffffecp-1f);
    const vec_x_f32 r_floor_even = hn::Set(D, 0x1p-21f);
    const vec_x_u32 sp_offset = hn::Set(exp_u32, 0x82000000u);
    const vec_x_u32 sp_bias = hn::Set(exp_u32, 0x7f000000u);
    const vec_x_f32 scale_bound = hn::Set(D, 0x1.8p+7f);
    const vec_x_u32 abs_mask = hn::Set(exp_u32, 0x7fffffffu);
    const vec_x_u32 inf_bits = hn::Set(exp_u32, 0x7f800000u);
    const vec_x_u32 bound_bits = hn::Set(exp_u32, 0x42af5dc2u); /* ~87.683 */
    vec_x_u32 abs_bits = hn::And(hn::BitCast(exp_u32, x), abs_mask);
    auto special_m = hn::Ge(abs_bits, bound_bits);
    vec_x_f32 result;
    if (__builtin_expect(!hn::AllFalse(exp_u32, special_m), 0)) {
        auto nan_m = hn::Gt(abs_bits, inf_bits);
        auto inf_m = hn::Eq(abs_bits, inf_bits);
        auto cmp = hn::And(special_m, hn::Lt(abs_bits, inf_bits));
        /* Clamp finite lanes to +-256 so the reduction, NearestInt and the
         * split polynomial never see |x| >= 2^31 (FCVTNS would raise a
         * spurious invalid and the unselected r_sp could overflow). The
         * huge-lane result is the cmp2 r_ov constant, which only needs the
         * sign of n; |x| <= 256 lanes are untouched, so every finite
         * special result is unchanged. Clamp before the NaN/inf select so
         * it overlaps the mask computation (NaN/inf lanes select zero). */
        const vec_x_f32 clamp = hn::Set(D, 256.0f);
        vec_x_f32 xc = hn::Min(hn::Max(x, hn::Neg(clamp)), clamp);
        vec_x_f32 xs = hn::IfThenElse(
            hn::RebindMask(D, hn::Or(nan_m, inf_m)), zero, xc);
        vec_x_f32 z = hn::MulAdd(xs, inv_ln2, shift);
        vec_x_f32 n = hn::Sub(z, shift);
        vec_x_f32 r = hn::MulAdd(n, ln2_hi, xs);
        r = hn::MulAdd(n, ln2_lo, r);
        vec_x_u32 e = hn::ShiftLeft<23>(hn::BitCast(exp_u32, hn::NearestInt(n)));
        vec_x_f32 scale = hn::BitCast(D, hn::Add(e, bias));
        vec_x_f32 r_even = hn::Max(hn::Abs(r), r_floor_even);
        vec_x_f32 r2 = hn::Mul(r_even, r_even);
        vec_x_f32 p = hn::MulAdd(r, c0, c1);
        vec_x_f32 q = hn::MulAdd(r, c2, c3);
        q = hn::MulAdd(p, r2, q);
        vec_x_f32 poly = hn::MulAdd(r, c4, hn::Mul(q, r2));
        vec_x_f32 scale_r = hn::IfThenElse(hn::RebindMask(D, cmp), one, scale);
        result = hn::MulAdd(poly, scale_r, scale_r);
        if (__builtin_expect(!hn::AllFalse(exp_u32, cmp), 0)) {
            vec_x_u32 b = hn::IfThenElse(
                hn::RebindMask(exp_u32, hn::Le(n, zero)),
                sp_offset, hn::Zero(exp_u32));
            vec_x_f32 s1 = hn::BitCast(D, hn::Add(b, sp_bias));
            vec_x_f32 s2 = hn::BitCast(D, hn::Sub(e, b));
            auto cmp2 = hn::Gt(hn::Abs(n), scale_bound);
            vec_x_f32 s1_ov = hn::IfThenElse(cmp2, s1, one);
            vec_x_f32 r_ov = hn::Mul(s1_ov, s1_ov);
            vec_x_f32 r_sp = hn::Mul(hn::MulAdd(poly, s2, s2), s1);
            vec_x_f32 sp_val = hn::IfThenElse(cmp2, r_ov, r_sp);
            result = hn::IfThenElse(hn::RebindMask(D, cmp), sp_val, result);
        }
        /* NaN passthrough via x + 0: raises INVALID on sNaN and hardware-
         * quietizes it (sign + payload preserved), matching libm; qNaN is
         * unchanged. No other FP op in this kernel sees the raw sNaN (the
         * reduction is sanitized), so the add is the flag source too. */
        result = hn::IfThenElse(hn::RebindMask(D, nan_m), hn::Add(x, zero), result);
        result = hn::IfThenElse(hn::RebindMask(D, inf_m), hn::Max(x, zero), result);
    } else {
        vec_x_f32 z = hn::MulAdd(x, inv_ln2, shift);
        vec_x_f32 n = hn::Sub(z, shift);
        vec_x_f32 r = hn::MulAdd(n, ln2_hi, x);
        r = hn::MulAdd(n, ln2_lo, r);
        vec_x_u32 e = hn::ShiftLeft<23>(hn::BitCast(exp_u32, hn::NearestInt(n)));
        vec_x_f32 scale = hn::BitCast(D, hn::Add(e, bias));
        vec_x_f32 r_even = hn::Max(hn::Abs(r), r_floor_even);
        vec_x_f32 r2 = hn::Mul(r_even, r_even);
        vec_x_f32 p = hn::MulAdd(r, c0, c1);
        vec_x_f32 q = hn::MulAdd(r, c2, c3);
        q = hn::MulAdd(p, r2, q);
        vec_x_f32 poly = hn::MulAdd(r, c4, hn::Mul(q, r2));
        result = hn::MulAdd(poly, scale, scale);
    }
    return result;
}

#if NPY_HWY_F16
HWY_ATTR static void
simd_exp_hwy_HALF(const npy_half *src, npy_intp ssrc,
                  npy_half *dst, npy_intp sdst, npy_intp len)
{
    /* Vectorized f16 path: load f16, promote lower/upper halves to f32, run
     * the f32 kernel on each, demote back to f16 (round-to-nearest-even, the
     * same hardware fcvt as scalar npy_float_to_half) and combine. The
     * f32->f16 narrowing raises OFC/UFC for out-of-half-range results exactly
     * as the scalar conversion does, so flags flow to the caller for parity.
     * No temporary buffer or raw NEON intrinsics. */
    const auto D16 = hn::ScalableTag<hwy::float16_t>();
    using vec_x_f16 = hn::Vec<decltype(D16)>;
    const int nlanes = hn::Lanes(D16);
    const auto dh = hn::Half<decltype(D16)>();
    for (; len > 0; len -= nlanes, src += ssrc * nlanes, dst += sdst * nlanes) {
        vec_x_f16 h = exp_load_vector<vec_x_f16>(
            reinterpret_cast<const hwy::float16_t *>(src), ssrc, len);
        vec_x_f32 lo = exp_f32_kernel(hn::PromoteLowerTo(exp_f32, h));
        vec_x_f32 hi = exp_f32_kernel(hn::PromoteUpperTo(exp_f32, h));
        vec_x_f16 out = hn::Combine(D16, hn::DemoteTo(dh, hi), hn::DemoteTo(dh, lo));
        exp_store_vector<vec_x_f16>(
            out, reinterpret_cast<hwy::float16_t *>(dst), sdst, len);
    }
}
#endif

#endif // SIMD_ARM

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

static void avx512_exp_f16(const npy_half *src, npy_half *dst, npy_intp len)
{
    avx512_exponent_f16(src, dst, len, __svml_expf16, 0);
}

#endif

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_exp)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if SIMD_ARM && NPY_HWY_F16
    npy_intp len = dimensions[0];
    /* Full-stride: vectorize every element-aligned, non-overlapping
     * stride. 16-bit load/store: stride 1 keeps LoadU/StoreU; other
     * strides scalar-gather/scatter the owned elements through a
     * contiguous stack temp + one ld1h/st1h (SVE gather is 32/64-bit
     * only), so no gather hardware is involved and the Kunpeng 920 SVE
     * gather bottleneck does not apply. Guards:
     * no mem overlap, element-aligned byte strides, nonzero output stride
     * (scatter-to-one-address race), and the int64 lane-index bound;
     * other runs fall back to scalar libm (policy note in loops_explog.h).
     * The lane index is i * stride with i < Lanes <= MaxLanes; the bound
     * below only matters for the wide SVE target (and the divisor is
     * guarded against MaxLanes == 1). */
    const npy_intp ssrc = steps[0] / (npy_intp)sizeof(npy_half);
    const npy_intp sdst = steps[1] / (npy_intp)sizeof(npy_half);
    bool guard_ok =
        !is_mem_overlap(args[0], steps[0], args[1], steps[1], len) &&
        steps[0] % (npy_intp)sizeof(npy_half) == 0 &&
        steps[1] % (npy_intp)sizeof(npy_half) == 0 &&
        sdst != 0;
    const hn::ScalableTag<hwy::float16_t> d16;
    if constexpr (hn::MaxLanes(d16) > 2) {
        const npy_intp ibound = INT64_MAX / ((npy_intp)hn::MaxLanes(d16) - 1);
        guard_ok = guard_ok && ssrc >= -ibound && ssrc <= ibound &&
            sdst >= -ibound && sdst <= ibound;
    }
    if (guard_ok) {
        simd_exp_hwy_HALF((npy_half *)args[0], ssrc, (npy_half *)args[1], sdst, len);
    } else {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_expf(in1));
        }
    }
    npyv_cleanup();
    return;
#elif NPY__SVML_IS_ENABLED
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];

    const npy_intp len = dimensions[0];

    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        (steps[0] == sizeof(npy_half)) &&
        (steps[1] == sizeof(npy_half))) {
    #ifdef NPY_HAVE_AVX512_SPR
        __svml_exps32(src, dst, len);
    #else
        avx512_exp_f16(src, dst, len);
    #endif
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_expf(in1));
    }
}
