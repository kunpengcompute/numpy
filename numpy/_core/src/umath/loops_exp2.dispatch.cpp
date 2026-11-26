#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "loops_explog.h"
#include "loops.h"

#if SIMD_ARM
#include "simd/simd.hpp"
#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;

/* SVE dispatch target: raw-ACLE FEXPA f64 kernel; NEON/ASIMD fall back to
 * the portable Highway advsimd kernel (see loops_exp.dispatch.cpp). */
#if defined(NPY_HAVE_SVE)
#define NPY_EXPLOG_SVE_FEXPA 1
#else
#define NPY_EXPLOG_SVE_FEXPA 0
#endif
#endif

#if SIMD_ARM

/* ---- Highway (Google Highway) AOR advsimd exp2 kernels -------------------
 * Algorithm ported from ARM optimized-routines math/aarch64/advsimd/exp2f.c
 * and exp2.c, expressed with Highway
 * scalable vectors following the official numpy SIMD mode (ScalableTag +
 * HWY_ATTR, LoadN/StoreN/GatherIndexN tails, GatherIndex table lookup).
 *
 * numpy contract adaptations (see PR #200 / commit c73289f):
 *  - NaN sign+payload passthrough; +-inf mapped to constants. NaN/inf lanes
 *    are sanitized to 0 before the polynomial so the raw AOR s1*s1 overflow
 *    term never raises a spurious OF/UF on them.
 *  - the s1*s2 bias-split special path keeps OF/UF raised naturally by the
 *    split arithmetic (no libm, no manual flag writes).
 *  - f32 tiny-remainder product flooring (|r|>=2^-21 even powers); f64 tiny
 *    nonzero |x|<2^-30 routed to the special path (single-rounding 1 + x*ln2).
 */
const hn::ScalableTag<float> exp2_f32;
const hn::ScalableTag<uint32_t> exp2_u32;
using vec_e_f32 = hn::Vec<decltype(exp2_f32)>;
using vec_e_u32 = hn::Vec<decltype(exp2_u32)>;

const hn::ScalableTag<double> exp2_f64;
const hn::ScalableTag<int64_t> exp2_s64;
const hn::ScalableTag<uint64_t> exp2_u64;
using vec_e_f64 = hn::Vec<decltype(exp2_f64)>;
using vec_e_u64 = hn::Vec<decltype(exp2_u64)>;

template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE vtype
exp2_load_vector(type_t *src, npy_intp ssrc, npy_intp len)
{
    auto D = hn::DFromV<vtype>();
    if constexpr (sizeof(type_t) == 2) {
        /* 16-bit lanes: SVE gather is 32/64-bit only, so build the vector
         * with scalar inserts (portable across NEON and SVE). */
        const int nlanes = hn::Lanes(D);
        if (ssrc == 1) {
            if (len < nlanes) return hn::LoadN(D, src, len);
            return hn::LoadU(D, src);
        }
        vtype v = hn::Zero(D);
        const int n = (len < nlanes) ? static_cast<int>(len) : nlanes;
        for (int i = 0; i < n; i++) v = hn::InsertLane(v, i, src[i * ssrc]);
        return v;
    } else {
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
}

template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE void
exp2_store_vector(vtype vec, type_t *dst, npy_intp sdst, npy_intp len)
{
    auto D = hn::DFromV<vtype>();
    if constexpr (sizeof(type_t) == 2) {
        const int nlanes = hn::Lanes(D);
        if (sdst == 1) {
            if (len < nlanes) { hn::StoreN(vec, D, dst, len); return; }
            hn::StoreU(vec, D, dst);
            return;
        }
        const int n = (len < nlanes) ? static_cast<int>(len) : nlanes;
        for (int i = 0; i < n; i++) dst[i * sdst] = hn::ExtractLane(vec, i);
    } else {
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
}

HWY_ATTR NPY_FINLINE vec_e_f32
exp2_f32_kernel(vec_e_f32 x)
{
    const auto D = exp2_f32;
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
    /* n << 23 via integer subtraction of the shift trick: exact for
     * |n| < 2^22 (all clean/finite-special lanes) and flag-safe for huge
     * lanes, where NearestInt (FCVTNS) would raise a spurious invalid.
     * Out-of-window e bits only reach lanes overridden by r_ov below. */
    vec_e_u32 e = hn::ShiftLeft<23>(
        hn::Sub(hn::BitCast(exp2_u32, z), hn::BitCast(exp2_u32, shift)));
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
            auto cmp2 = hn::Gt(hn::Abs(n), scale_bound);
            /* |n| > 192 lanes take r_ov; sanitize s2 there so the
             * unselected r_sp fma never sees inf/NaN bits from the
             * out-of-window integer exponent (spurious OF/invalid). */
            vec_e_f32 s2 = hn::IfThenElse(
                hn::RebindMask(D, cmp2), one, hn::BitCast(D, hn::Sub(e, b)));
            /* s1*s1 overflows (n>0 -> s1=2^127) even on lanes where it
             * is not selected; sanitize so only |n|>192 lanes flag. */
            vec_e_f32 s1_ov = hn::IfThenElse(cmp2, s1, one);
            vec_e_f32 r_ov = hn::Mul(s1_ov, s1_ov);
            vec_e_f32 r_sp = hn::Mul(hn::MulAdd(poly, s2, s2), s1);
            vec_e_f32 sp_val = hn::IfThenElse(cmp2, r_ov, r_sp);
            result = hn::IfThenElse(hn::RebindMask(D, cmp), sp_val, result);
        }
        /* NaN payload passthrough via x + 0: raises INVALID on sNaN and
         * hardware-quietizes it (sign + payload preserved), matching libm;
         * qNaN unchanged. The reduction is sanitized, so this add is the
         * only FP op that sees the raw sNaN (flag source). +-inf ->
         * constants (exp2(+inf)=+inf, exp2(-inf)=+0) without the split's
         * spurious OF/UF. */
        result = hn::IfThenElse(hn::RebindMask(D, nan_m), hn::Add(x, zero), result);
        result = hn::IfThenElse(hn::RebindMask(D, inf_m), hn::Max(x, zero), result);
    } else {
        if constexpr (!HWY_HAVE_SCALABLE) {
            result = hn::MulAdd(poly, scale, scale);
        }
    }
    return result;
}

HWY_ATTR static void
simd_exp2_hwy_FLOAT(const npy_float *src, npy_intp ssrc,
                    npy_float *dst, npy_intp sdst, npy_intp len)
{
    const int nlanes = hn::Lanes(exp2_f32);
    for (; len > 0; len -= nlanes, src += ssrc * nlanes, dst += sdst * nlanes) {
        vec_e_f32 x = exp2_load_vector<vec_e_f32>(src, ssrc, len);
        vec_e_f32 out = exp2_f32_kernel(x);
        exp2_store_vector<vec_e_f32>(out, dst, sdst, len);
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

#if NPY_EXPLOG_SVE_FEXPA
/* ---- SVE-native FEXPA exp2 f64 kernel (SVE dispatch target only) --------
 * ARM optimized-routines math/aarch64/sve/exp2.c (svexpa hardware 64-entry
 * 2^(i/64) table, no gather), numpy-contract adapted as the exp kernel
 * above; the negative clamp bound is -1075.0 (not AOR's -1076.0078125) so
 * clamped lanes keep scale >= 2^-1022 and FSCALE raises UF naturally
 * (exp2(x) rounds to 0 for every x <= -1075, tie-to-even).
 */

static inline svfloat64_t
sv_f64(double x)
{
    return svdup_f64(x);
}
static inline svuint64_t
sv_u64(uint64_t x)
{
    return svdup_u64(x);
}
static inline svint64_t
sv_s64(int64_t x)
{
    return svdup_s64(x);
}
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif
#ifndef ptr_barrier
#define ptr_barrier(ptr) ({ __typeof(ptr) __p = (ptr); __asm__("" : "+r"(__p)); __p; })
#endif
#define InfBound 0x1p10
#define ZeroBound -0x1.0ccp+10 /* -1075.0 */

static const struct exp2_sve_data
{
    double c2, c4;
    double c0, c1, c3;
    double shift;
    double inf_bound, zero_bound;
    double ln2;
} exp2_sve_data = {
    .c2 = 0x1.c6b08d706c8a5p-5,
    .c4 = 0x1.5d8761184beb3p-10,
    .c0 = 0x1.62e42fefa39efp-1,
    .c1 = 0x1.ebfbdff82a31bp-3,
    .c3 = 0x1.3b2ad2ff7d2f3p-7,
    .shift = 0x1.800000000ffc0p+46,
    .inf_bound = InfBound,
    .zero_bound = ZeroBound,
    .ln2 = 0x1.62e42fefa39efp-1,

};

static inline svfloat64_t
exp2_inline(svfloat64_t x, svbool_t pg, const struct exp2_sve_data *d)
{
    svfloat64_t z = svadd_x(svptrue_b64(), x, d->shift);
    svfloat64_t n = svsub_x(pg, z, d->shift);
    svfloat64_t r = svsub_x(pg, x, n);

    svfloat64_t scale = svexpa(svreinterpret_u64(z));

    svfloat64_t r2 = svmul_x(svptrue_b64(), r, r);
    svfloat64_t c24 = svld1rq(svptrue_b64(), &d->c2);

    svfloat64_t p12 = svmla_lane(sv_f64(d->c1), r, c24, 0);
    svfloat64_t p34 = svmla_lane(sv_f64(d->c3), r, c24, 1);
    svfloat64_t p = svmla_x(pg, p12, p34, r2);
    p = svmad_x(pg, p, r, d->c0);
    svfloat64_t poly = svmul_x(svptrue_b64(), r, p);

    return svmla_x(pg, scale, scale, poly);
}

static svfloat64_t __attribute__((always_inline)) inline
special_case(svfloat64_t x, svbool_t pg, const struct exp2_sve_data *d)
{
    svbool_t is_negative = svcmplt(pg, x, 0.0);
    svfloat64_t offset = svneg_m(sv_f64(53.0), is_negative, sv_f64(53.0));
    svint64_t scale_adjust = svneg_m(sv_s64(53), is_negative, sv_s64(53));

    x = svmin_x(pg, svmax_x(pg, x, d->zero_bound), d->inf_bound);

    svfloat64_t z = svadd_x(svptrue_b64(), x, d->shift);
    svfloat64_t n = svsub_x(pg, z, d->shift);
    svfloat64_t r = svsub_x(pg, x, n);

    z = svsub_x(pg, z, offset);
    svfloat64_t scale = svexpa(svreinterpret_u64(z));

    svfloat64_t r2 = svmul_x(svptrue_b64(), r, r);
    svfloat64_t c24 = svld1rq(svptrue_b64(), &d->c2);

    svfloat64_t p12 = svmla_lane(sv_f64(d->c1), r, c24, 0);
    svfloat64_t p34 = svmla_lane(sv_f64(d->c3), r, c24, 1);
    svfloat64_t p = svmla_x(pg, p12, p34, r2);
    p = svmad_x(pg, p, r, d->c0);
    svfloat64_t poly = svmul_x(svptrue_b64(), r, p);

    svfloat64_t y = svmla_x(pg, scale, scale, poly);

    y = svscale_x(pg, y, scale_adjust);
    /* glibc raises UF for every tiny result, even exact ones (exp2(-1074));
       FSCALE/FMUL only flag inexact tininess on HIP12.  Supply UF via an
       inexact underflowing product on lanes whose result is subnormal/zero.
       y + 0 is exact, so the value is unchanged.  */
    svuint64_t yexp
        = svand_x(pg, svreinterpret_u64(y), sv_u64(0x7ff0000000000000ULL));
    svbool_t tiny_out = svcmpeq_u64(pg, yexp, sv_u64(0));
    svfloat64_t uf = svmul_m(tiny_out, sv_f64(0x1p-600), sv_f64(0x1p-600));
    return svadd_m(tiny_out, y, uf);
}

static inline svfloat64_t
exp2_f64_sve_fexpa(svfloat64_t x, svbool_t pg)
{
    const struct exp2_sve_data *d = ptr_barrier(&exp2_sve_data);
    svuint64_t abs_bits
        = svand_x(pg, svreinterpret_u64(x), sv_u64(0x7fffffffffffffffULL));
    svbool_t special
        = svcmpge_u64(pg, abs_bits, sv_u64(0x408ff00000000000ULL)); /*1022*/
    svbool_t tiny = svand_b_z(
        pg, svcmplt_u64(pg, abs_bits, sv_u64(0x3e10000000000000ULL)),
        svcmpne_u64(pg, abs_bits, sv_u64(0)));
    if (unlikely(svptest_any(pg, svorr_z(pg, special, tiny)))) {
        svbool_t nan_m
            = svcmpgt_u64(pg, abs_bits, sv_u64(0x7ff0000000000000ULL));
        svbool_t inf_m
            = svcmpeq_u64(pg, abs_bits, sv_u64(0x7ff0000000000000ULL));
        svbool_t sp_finite = svand_b_z(pg, special,
            svnot_b_z(pg, svorr_z(pg, nan_m, inf_m)));
        svbool_t care = svorr_z(pg, svorr_z(pg, nan_m, inf_m), tiny);
        svfloat64_t xs = svsel_f64(care, sv_f64(0.0), x);
        svfloat64_t y;
        if (svptest_any(pg, sp_finite)) {
            /* finite OF/UF lanes present: one heavy compute for all lanes
               (clamp + offset + FEXPA + FSCALE), correct on clean lanes too */
            y = special_case(xs, pg, d);
        } else {
            /* NaN/inf/tiny/zero only: light clean-form compute on sanitized
               input (finite-special lanes sanitized to 0 as well) */
            svfloat64_t xc
                = svsel_f64(svorr_z(pg, special, tiny), sv_f64(0.0), x);
            y = exp2_inline(xc, pg, d);
        }
        y = svsel_f64(tiny, svmla_x(pg, sv_f64(1.0), x, d->ln2), y);
        /* NaN passthrough preserving sign+payload; set the quiet bit so an
           sNaN is quieted exactly as scalar libm does (qNaN unchanged).  */
        svfloat64_t qn = svreinterpret_f64(
            svorr_x(pg, svreinterpret_u64(x), sv_u64(0x0008000000000000ULL)));
        y = svsel_f64(nan_m, qn, y);
        y = svsel_f64(inf_m, svmax_x(pg, x, sv_f64(0.0)), y);
        return y;
    }
    return exp2_inline(x, pg, d);
}

HWY_ATTR NPY_FINLINE vec_e_f64
exp2_f64_kernel(vec_e_f64 x)
{
    return exp2_f64_sve_fexpa(x, svptrue_b64());
}

#else
HWY_ATTR NPY_FINLINE vec_e_f64
exp2_f64_kernel(vec_e_f64 x)
{
    const auto D = exp2_f64;
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
        /* NaN passthrough via x + 0: hardware-quietizes sNaN (sign +
         * payload preserved) matching libm, as the SVE FEXPA path and
         * PR #203 log do; qNaN unchanged. INVALID is already raised by
         * the tiny_res FMA above, which sees the raw sNaN. */
        result = hn::IfThenElse(hn::RebindMask(D, nan_m), hn::Add(x, zero), result);
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
    return result;
}

#endif /* NPY_EXPLOG_SVE_FEXPA */

HWY_ATTR static void
simd_exp2_hwy_DOUBLE(const npy_double *src, npy_intp ssrc,
                     npy_double *dst, npy_intp sdst, npy_intp len)
{
    const int nlanes = hn::Lanes(exp2_f64);
    for (; len > 0; len -= nlanes, src += ssrc * nlanes, dst += sdst * nlanes) {
        vec_e_f64 x = exp2_load_vector<vec_e_f64>(src, ssrc, len);
        vec_e_f64 out = exp2_f64_kernel(x);
        exp2_store_vector<vec_e_f64>(out, dst, sdst, len);
    }
}

#if NPY_HWY_F16
HWY_ATTR static void
simd_exp2_hwy_HALF(const npy_half *src, npy_intp ssrc,
                   npy_half *dst, npy_intp sdst, npy_intp len)
{
    /* Vectorized f16 path: load f16, promote the lower/upper halves to f32,
     * run the f32 kernel on each, demote back to f16 (round-to-nearest-even,
     * same hardware fcvt as the scalar npy_float_to_half) and combine. The
     * f32->f16 narrowing raises OFC/UFC for out-of-half-range results exactly
     * as the scalar conversion does, so the flags flow to the caller for
     * parity. No temporary buffer or raw NEON intrinsics. */
    const auto D16 = hn::ScalableTag<hwy::float16_t>();
    using vec_e_f16 = hn::Vec<decltype(D16)>;
    const int nlanes = hn::Lanes(D16);
    const auto dh = hn::Half<decltype(D16)>();
    for (; len > 0; len -= nlanes, src += ssrc * nlanes, dst += sdst * nlanes) {
        vec_e_f16 h = exp2_load_vector<vec_e_f16>(
            reinterpret_cast<const hwy::float16_t *>(src), ssrc, len);
        vec_e_f32 lo = exp2_f32_kernel(hn::PromoteLowerTo(exp2_f32, h));
        vec_e_f32 hi = exp2_f32_kernel(hn::PromoteUpperTo(exp2_f32, h));
        vec_e_f16 out = hn::Combine(D16, hn::DemoteTo(dh, hi),
                                    hn::DemoteTo(dh, lo));
        exp2_store_vector<vec_e_f16>(
            out, reinterpret_cast<hwy::float16_t *>(dst), sdst, len);
    }
}
#endif

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
    npy_intp len = dimensions[0];
    /* Vectorize only contiguous non-overlapping runs; strided/overlap
     * falls back to scalar libm (policy note in loops_explog.h). */
    if (steps[0] == (npy_intp)sizeof(npy_float) &&
        steps[1] == (npy_intp)sizeof(npy_float) &&
        !is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        simd_exp2_hwy_FLOAT((npy_float *)args[0], 1, (npy_float *)args[1], 1, len);
    } else {
        UNARY_LOOP {
            const npy_float in1 = *(npy_float *)ip1;
            *(npy_float *)op1 = npy_exp2f(in1);
        }
    }
    npyv_cleanup();
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
    npy_intp len = dimensions[0];
    /* Vectorize only contiguous non-overlapping runs; strided/overlap
     * falls back to scalar libm (policy note in loops_explog.h). */
    if (steps[0] == (npy_intp)sizeof(npy_double) &&
        steps[1] == (npy_intp)sizeof(npy_double) &&
        !is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        simd_exp2_hwy_DOUBLE((npy_double *)args[0], 1, (npy_double *)args[1], 1, len);
    } else {
        UNARY_LOOP {
            const npy_double in1 = *(npy_double *)ip1;
            *(npy_double *)op1 = npy_exp2(in1);
        }
    }
    npyv_cleanup();
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
#if SIMD_ARM && NPY_HWY_F16
    npy_intp len = dimensions[0];
    /* Vectorize only contiguous non-overlapping runs; strided/overlap
     * falls back to scalar libm (policy note in loops_explog.h). */
    if (steps[0] == (npy_intp)sizeof(npy_half) &&
        steps[1] == (npy_intp)sizeof(npy_half) &&
        !is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        simd_exp2_hwy_HALF((npy_half *)args[0], 1, (npy_half *)args[1], 1, len);
    } else {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_exp2f(in1));
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
