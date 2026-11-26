#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "loops_explog.h"
#include "loops.h"

#ifdef SIMD_AVX2_FMA3
/*
 * Vectorized implementation of exp using AVX2 and AVX512:
 * 1) if x >= xmax; return INF (overflow)
 * 2) if x <= xmin; return 0.0f (underflow)
 * 3) Range reduction (using Coyd-Waite):
 *      a) y = x - k*ln(2); k = rint(x/ln(2)); y \in [0, ln(2)]
 * 4) Compute exp(y) = P/Q, ratio of 2 polynomials P and Q
 *      b) P = 5th order and Q = 2nd order polynomials obtained from Remez's
 *      algorithm (mini-max polynomial approximation)
 * 5) Compute exp(x) = exp(y) * 2^k
 * 6) Max ULP error measured across all 32-bit FP's = 2.52 (x = 0xc2781e37)
 * 7) Max relative error measured across all 32-bit FP's= 2.1264E-07 (for the
 * same x = 0xc2781e37)
 */
static void
simd_exp_FLOAT(npy_float * op,
                npy_float * ip,
                const npy_intp array_size,
                const npy_intp steps)
{
    const npy_intp stride = steps/(npy_intp)sizeof(npy_float);
    const npy_int num_lanes = 32/(npy_intp)sizeof(npy_float);
    npy_float xmax = 88.72283935546875f;
    npy_float xmin = -103.97208404541015625f;

    npy_int32 indexarr[16];
    for (npy_int32 ii = 0; ii < 16; ii++) {
        indexarr[ii] = ii*stride;
    }

    __m256 codyw_c1 = _mm256_set1_ps(NPY_CODY_WAITE_LOGE_2_HIGHf);
    __m256 codyw_c2 = _mm256_set1_ps(NPY_CODY_WAITE_LOGE_2_LOWf);
    __m256 exp_p0 = _mm256_set1_ps(NPY_COEFF_P0_EXPf);
    __m256 exp_p1 = _mm256_set1_ps(NPY_COEFF_P1_EXPf);
    __m256 exp_p2 = _mm256_set1_ps(NPY_COEFF_P2_EXPf);
    __m256 exp_p3 = _mm256_set1_ps(NPY_COEFF_P3_EXPf);
    __m256 exp_p4 = _mm256_set1_ps(NPY_COEFF_P4_EXPf);
    __m256 exp_p5 = _mm256_set1_ps(NPY_COEFF_P5_EXPf);
    __m256 exp_q0 = _mm256_set1_ps(NPY_COEFF_Q0_EXPf);
    __m256 exp_q1 = _mm256_set1_ps(NPY_COEFF_Q1_EXPf);
    __m256 exp_q2 = _mm256_set1_ps(NPY_COEFF_Q2_EXPf);
    __m256 cvt_magic = _mm256_set1_ps(NPY_RINT_CVT_MAGICf);
    __m256 log2e = _mm256_set1_ps(NPY_LOG2Ef);
    __m256 inf = _mm256_set1_ps(NPY_INFINITYF);
    __m256 ninf = _mm256_set1_ps(-1*NPY_INFINITYF);
    __m256 zeros_f = _mm256_set1_ps(0.0f);
    __m256 poly, num_poly, denom_poly, quadrant;
    __m256i vindex = _mm256_loadu_si256((__m256i*)&indexarr[0]);

    __m256 xmax_mask, xmin_mask, nan_mask, inf_mask, ninf_mask;
    __m256 overflow_mask = fma_get_partial_load_mask_ps(0, num_lanes);
    __m256 underflow_mask = fma_get_partial_load_mask_ps(0, num_lanes);
    __m256 load_mask = fma_get_full_load_mask_ps();
    npy_intp num_remaining_elements = array_size;

    while (num_remaining_elements > 0) {

        if (num_remaining_elements < num_lanes) {
            load_mask = fma_get_partial_load_mask_ps(num_remaining_elements,
                                                       num_lanes);
        }

        __m256 x;
        if (stride == 1) {
            x = fma_masked_load_ps(load_mask, ip);
        }
        else {
            x = fma_masked_gather_ps(zeros_f, ip, vindex, load_mask);
        }

        nan_mask = _mm256_cmp_ps(x, x, _CMP_NEQ_UQ);
        x = fma_set_masked_lanes_ps(x, zeros_f, nan_mask);

        xmax_mask = _mm256_cmp_ps(x, _mm256_set1_ps(xmax), _CMP_GE_OQ);
        xmin_mask = _mm256_cmp_ps(x, _mm256_set1_ps(xmin), _CMP_LE_OQ);
        inf_mask = _mm256_cmp_ps(x, inf, _CMP_EQ_OQ);
        ninf_mask = _mm256_cmp_ps(x, ninf, _CMP_EQ_OQ);
        overflow_mask = _mm256_or_ps(overflow_mask,
                                    _mm256_xor_ps(xmax_mask, inf_mask));
        underflow_mask = _mm256_or_ps(underflow_mask,
                                    _mm256_xor_ps(xmin_mask, ninf_mask));

        x = fma_set_masked_lanes_ps(x, zeros_f, _mm256_or_ps(
                                    _mm256_or_ps(nan_mask, xmin_mask), xmax_mask));

        quadrant = _mm256_mul_ps(x, log2e);

        quadrant = _mm256_add_ps(quadrant, cvt_magic);
        quadrant = _mm256_sub_ps(quadrant, cvt_magic);

        x = simd_range_reduction(x, quadrant, codyw_c1, codyw_c2, zeros_f);

        num_poly = _mm256_fmadd_ps(exp_p5, x, exp_p4);
        num_poly = _mm256_fmadd_ps(num_poly, x, exp_p3);
        num_poly = _mm256_fmadd_ps(num_poly, x, exp_p2);
        num_poly = _mm256_fmadd_ps(num_poly, x, exp_p1);
        num_poly = _mm256_fmadd_ps(num_poly, x, exp_p0);
        denom_poly = _mm256_fmadd_ps(exp_q2, x, exp_q1);
        denom_poly = _mm256_fmadd_ps(denom_poly, x, exp_q0);
        poly = _mm256_div_ps(num_poly, denom_poly);

        poly = fma_scalef_ps(poly, quadrant);

        poly = fma_set_masked_lanes_ps(poly, _mm256_set1_ps(NPY_NANF), nan_mask);
        poly = fma_set_masked_lanes_ps(poly, inf, xmax_mask);
        poly = fma_set_masked_lanes_ps(poly, zeros_f, xmin_mask);

        _mm256_maskstore_ps(op, _mm256_cvtps_epi32(load_mask), poly);

        ip += num_lanes*stride;
        op += num_lanes;
        num_remaining_elements -= num_lanes;
    }

    if (_mm256_movemask_ps(overflow_mask)) {
        npy_set_floatstatus_overflow();
    }

    if (_mm256_movemask_ps(underflow_mask)) {
        npy_set_floatstatus_underflow();
    }
}

#endif // SIMD_AVX2_FMA3

#ifdef SIMD_AVX512F
/*
 * Vectorized implementation of exp using AVX2 and AVX512:
 * 1) if x >= xmax; return INF (overflow)
 * 2) if x <= xmin; return 0.0f (underflow)
 * 3) Range reduction (using Coyd-Waite):
 *      a) y = x - k*ln(2); k = rint(x/ln(2)); y \in [0, ln(2)]
 * 4) Compute exp(y) = P/Q, ratio of 2 polynomials P and Q
 *      b) P = 5th order and Q = 2nd order polynomials obtained from Remez's
 *      algorithm (mini-max polynomial approximation)
 * 5) Compute exp(x) = exp(y) * 2^k
 * 6) Max ULP error measured across all 32-bit FP's = 2.52 (x = 0xc2781e37)
 * 7) Max relative error measured across all 32-bit FP's= 2.1264E-07 (for the
 * same x = 0xc2781e37)
 */
static void
simd_exp_FLOAT(npy_float * op,
                npy_float * ip,
                const npy_intp array_size,
                const npy_intp steps)
{
    const npy_intp stride = steps/(npy_intp)sizeof(npy_float);
    const npy_int num_lanes = 64/(npy_intp)sizeof(npy_float);
    npy_float xmax = 88.72283935546875f;
    npy_float xmin = -103.97208404541015625f;

    npy_int32 indexarr[16];
    for (npy_int32 ii = 0; ii < 16; ii++) {
        indexarr[ii] = ii*stride;
    }

    __m512 codyw_c1 = _mm512_set1_ps(NPY_CODY_WAITE_LOGE_2_HIGHf);
    __m512 codyw_c2 = _mm512_set1_ps(NPY_CODY_WAITE_LOGE_2_LOWf);
    __m512 exp_p0 = _mm512_set1_ps(NPY_COEFF_P0_EXPf);
    __m512 exp_p1 = _mm512_set1_ps(NPY_COEFF_P1_EXPf);
    __m512 exp_p2 = _mm512_set1_ps(NPY_COEFF_P2_EXPf);
    __m512 exp_p3 = _mm512_set1_ps(NPY_COEFF_P3_EXPf);
    __m512 exp_p4 = _mm512_set1_ps(NPY_COEFF_P4_EXPf);
    __m512 exp_p5 = _mm512_set1_ps(NPY_COEFF_P5_EXPf);
    __m512 exp_q0 = _mm512_set1_ps(NPY_COEFF_Q0_EXPf);
    __m512 exp_q1 = _mm512_set1_ps(NPY_COEFF_Q1_EXPf);
    __m512 exp_q2 = _mm512_set1_ps(NPY_COEFF_Q2_EXPf);
    __m512 cvt_magic = _mm512_set1_ps(NPY_RINT_CVT_MAGICf);
    __m512 log2e = _mm512_set1_ps(NPY_LOG2Ef);
    __m512 inf = _mm512_set1_ps(NPY_INFINITYF);
    __m512 ninf = _mm512_set1_ps(-1*NPY_INFINITYF);
    __m512 zeros_f = _mm512_set1_ps(0.0f);
    __m512 poly, num_poly, denom_poly, quadrant;
    __m512i vindex = _mm512_loadu_si512((__m512i*)&indexarr[0]);

    __mmask16 xmax_mask, xmin_mask, nan_mask, inf_mask, ninf_mask;
    __mmask16 overflow_mask = avx512_get_partial_load_mask_ps(0, num_lanes);
    __mmask16 underflow_mask = avx512_get_partial_load_mask_ps(0, num_lanes);
    __mmask16 load_mask = avx512_get_full_load_mask_ps();
    npy_intp num_remaining_elements = array_size;

    while (num_remaining_elements > 0) {

        if (num_remaining_elements < num_lanes) {
            load_mask = avx512_get_partial_load_mask_ps(num_remaining_elements,
                                                       num_lanes);
        }

        __m512 x;
        if (stride == 1) {
            x = avx512_masked_load_ps(load_mask, ip);
        }
        else {
            x = avx512_masked_gather_ps(zeros_f, ip, vindex, load_mask);
        }

        nan_mask = _mm512_cmp_ps_mask(x, x, _CMP_NEQ_UQ);
        x = avx512_set_masked_lanes_ps(x, zeros_f, nan_mask);

        xmax_mask = _mm512_cmp_ps_mask(x, _mm512_set1_ps(xmax), _CMP_GE_OQ);
        xmin_mask = _mm512_cmp_ps_mask(x, _mm512_set1_ps(xmin), _CMP_LE_OQ);
        inf_mask = _mm512_cmp_ps_mask(x, inf, _CMP_EQ_OQ);
        ninf_mask = _mm512_cmp_ps_mask(x, ninf, _CMP_EQ_OQ);
        overflow_mask = _mm512_kor(overflow_mask,
                                    _mm512_kxor(xmax_mask, inf_mask));
        underflow_mask = _mm512_kor(underflow_mask,
                                    _mm512_kxor(xmin_mask, ninf_mask));

        x = avx512_set_masked_lanes_ps(x, zeros_f, _mm512_kor(
                                    _mm512_kor(nan_mask, xmin_mask), xmax_mask));

        quadrant = _mm512_mul_ps(x, log2e);

        quadrant = _mm512_add_ps(quadrant, cvt_magic);
        quadrant = _mm512_sub_ps(quadrant, cvt_magic);

        x = simd_range_reduction(x, quadrant, codyw_c1, codyw_c2, zeros_f);

        num_poly = _mm512_fmadd_ps(exp_p5, x, exp_p4);
        num_poly = _mm512_fmadd_ps(num_poly, x, exp_p3);
        num_poly = _mm512_fmadd_ps(num_poly, x, exp_p2);
        num_poly = _mm512_fmadd_ps(num_poly, x, exp_p1);
        num_poly = _mm512_fmadd_ps(num_poly, x, exp_p0);
        denom_poly = _mm512_fmadd_ps(exp_q2, x, exp_q1);
        denom_poly = _mm512_fmadd_ps(denom_poly, x, exp_q0);
        poly = _mm512_div_ps(num_poly, denom_poly);

        poly = avx512_scalef_ps(poly, quadrant);

        poly = avx512_set_masked_lanes_ps(poly, _mm512_set1_ps(NPY_NANF), nan_mask);
        poly = avx512_set_masked_lanes_ps(poly, inf, xmax_mask);
        poly = avx512_set_masked_lanes_ps(poly, zeros_f, xmin_mask);

        _mm512_mask_storeu_ps(op, load_mask, poly);

        ip += num_lanes*stride;
        op += num_lanes;
        num_remaining_elements -= num_lanes;
    }

    if (npyv_tobits_b32(overflow_mask)) {
        npy_set_floatstatus_overflow();
    }

    if (npyv_tobits_b32(underflow_mask)) {
        npy_set_floatstatus_underflow();
    }
}

#endif // SIMD_AVX512F


#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
static void
simd_exp_f64(const npyv_lanetype_f64 *src, npy_intp ssrc,
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
        npyv_f64 out = __svml_exp8_ha(x);
        if (sdst == 1) {
            npyv_store_till_f64(dst, len, out);
        } else {
            npyv_storen_till_f64(dst, sdst, len, out);
        }
    }
    npyv_cleanup();
}

#else
#ifdef SIMD_AVX512F_NOCLANG_BUG

/* LLVM has a bug where AVX-512F intrinsic `_mm512_mask_mul_pd` emits an
 * unmasked operation with a masked store.  This can cause FP exceptions to
 * occur for the lanes that are suppose to have been masked.
 *
 * See https://bugs.llvm.org/show_bug.cgi?id=51988
 *
 * Note, this affects LLVM based compilers like Apple Clang, Clang, and Intel's
 * ICX.
 */
#if defined(__clang__)
    #if defined(__apple_build_version__)
        #if __apple_build_version__ > 11000000
        #define WORKAROUND_LLVM__mm512_mask_mul_pd
        #endif
    #else
        #if __clang_major__ > 9
        #define WORKAROUND_LLVM__mm512_mask_mul_pd
        #endif
    #endif
#endif

/*
 * Vectorized implementation of exp double using AVX512
 * Reference: Tang, P.T.P., "Table-driven implementation of the
 *  exponential function in IEEE floating-point
 *  arithmetic," ACM Transactions on Mathematical
 *  Software, vol. 15, pp. 144-157, 1989.
 * 1) if x > mTH_max or x is INF; return INF (overflow)
 * 2) if x < mTH_min; return 0.0f (underflow)
 * 3) if abs(x) < mTH_nearzero; return 1.0f + x
 * 4) if x is Nan; return Nan
 * 5) Range reduction:
 *    x = (32m + j)ln2 / 32 + r; r in [-ln2/64, ln2/64]
 * 6) exp(r) - 1 is approximated by a polynomial function p(r)
 *    exp(x) = 2^m(2^(j/32) + 2^(j/32)p(r));
 */
static void
AVX512F_exp_DOUBLE(npy_double * op,
                npy_double * ip,
                const npy_intp array_size,
                const npy_intp steps)
{
    npy_intp num_remaining_elements = array_size;
    const npy_intp stride = steps / (npy_intp)sizeof(npy_double);
    const npy_int num_lanes = 64 / (npy_intp)sizeof(npy_double);
    npy_int32 indexarr[8];
    for (npy_int32 ii = 0; ii < 8; ii++) {
        indexarr[ii] = ii*stride;
    }

    __m512d InvLn2N = _mm512_set1_pd(NPY_INV_LN2_MUL_32);
    __m512d mShift = _mm512_set1_pd(NPY_RINT_CVT_MAGIC);
    __m512d mNegL1 = _mm512_set1_pd(NPY_TANG_NEG_L1);
    __m512d mNegL2 = _mm512_set1_pd(NPY_TANG_NEG_L2);
    __m512i mMod = _mm512_set1_epi64(0x1f);
    __m512d mA1 = _mm512_set1_pd(NPY_TANG_A1);
    __m512d mA2 = _mm512_set1_pd(NPY_TANG_A2);
    __m512d mA3 = _mm512_set1_pd(NPY_TANG_A3);
    __m512d mA4 = _mm512_set1_pd(NPY_TANG_A4);
    __m512d mA5 = _mm512_set1_pd(NPY_TANG_A5);
    __m512d mTH_nearzero = _mm512_set1_pd(0x1p-54);
    __m512d mTH_max = _mm512_set1_pd(0x1.62e42fefa39efp+9);
    __m512d mTH_min = _mm512_set1_pd(-0x1.74910d52d3053p+9);
    __m512d mTH_inf = _mm512_set1_pd(NPY_INFINITY);
    __m512d mTH_ninf = _mm512_set1_pd(-NPY_INFINITY);
    __m512d zeros_d = _mm512_set1_pd(0.0);
    __m512d ones_d = _mm512_set1_pd(1.0);
    __m256i vindex = _mm256_loadu_si256((__m256i*)&indexarr[0]);

    __m512d mTable_top_0 = _mm512_loadu_pd(&(EXP_Table_top[8*0]));
    __m512d mTable_top_1 = _mm512_loadu_pd(&(EXP_Table_top[8*1]));
    __m512d mTable_top_2 = _mm512_loadu_pd(&(EXP_Table_top[8*2]));
    __m512d mTable_top_3 = _mm512_loadu_pd(&(EXP_Table_top[8*3]));
    __m512d mTable_tail_0 = _mm512_loadu_pd(&(EXP_Table_tail[8*0]));
    __m512d mTable_tail_1 = _mm512_loadu_pd(&(EXP_Table_tail[8*1]));
    __m512d mTable_tail_2 = _mm512_loadu_pd(&(EXP_Table_tail[8*2]));
    __m512d mTable_tail_3 = _mm512_loadu_pd(&(EXP_Table_tail[8*3]));

    __mmask8 overflow_mask = avx512_get_partial_load_mask_pd(0, num_lanes);
    __mmask8 underflow_mask = avx512_get_partial_load_mask_pd(0, num_lanes);
    __mmask8 load_mask = avx512_get_full_load_mask_pd();
    __mmask8 xmin_mask, xmax_mask, inf_mask, ninf_mask, nan_mask, nearzero_mask;

    while (num_remaining_elements > 0) {
        if (num_remaining_elements < num_lanes) {
            load_mask = avx512_get_partial_load_mask_pd(num_remaining_elements,
                                                      num_lanes);
        }

        __m512d x;
        if (1 == stride) {
            x = avx512_masked_load_pd(load_mask, ip);
        }
        else {
            x = avx512_masked_gather_pd(zeros_d, ip, vindex, load_mask);
        }

        nan_mask = _mm512_cmp_pd_mask(x, x, _CMP_NEQ_UQ);
        x = avx512_set_masked_lanes_pd(x, zeros_d, nan_mask);
        xmax_mask = _mm512_cmp_pd_mask(x, mTH_max, _CMP_GT_OQ);
        xmin_mask = _mm512_cmp_pd_mask(x, mTH_min, _CMP_LT_OQ);
        inf_mask = _mm512_cmp_pd_mask(x, mTH_inf, _CMP_EQ_OQ);
        ninf_mask = _mm512_cmp_pd_mask(x, mTH_ninf, _CMP_EQ_OQ);
        __m512i x_abs = _mm512_and_epi64(_mm512_castpd_si512(x),
                                _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF));
        nearzero_mask = _mm512_cmp_pd_mask(_mm512_castsi512_pd(x_abs),
                                    mTH_nearzero, _CMP_LT_OQ);
        nearzero_mask = _mm512_kxor(nearzero_mask, nan_mask);
        overflow_mask = _mm512_kor(overflow_mask,
                                _mm512_kxor(xmax_mask, inf_mask));
        underflow_mask = _mm512_kor(underflow_mask,
                                _mm512_kxor(xmin_mask, ninf_mask));
        x = avx512_set_masked_lanes_pd(x, zeros_d,
                        _mm512_kor(_mm512_kor(nan_mask, xmin_mask),
                            _mm512_kor(xmax_mask, nearzero_mask)));

        __m512d z = _mm512_mul_pd(x, InvLn2N);

        __m512d kd = _mm512_add_pd(z, mShift);
        __m512i ki = _mm512_castpd_si512(kd);
        kd = _mm512_sub_pd(kd, mShift);

        __m512d r1 = _mm512_fmadd_pd(kd, mNegL1, x);
        __m512d r2 = _mm512_mul_pd(kd, mNegL2);
        __m512d r = _mm512_add_pd(r1,r2);

        __m512d q = _mm512_fmadd_pd(mA5, r, mA4);
        q = _mm512_fmadd_pd(q, r, mA3);
        q = _mm512_fmadd_pd(q, r, mA2);
        q = _mm512_fmadd_pd(q, r, mA1);
        q = _mm512_mul_pd(q, r);
        __m512d p = _mm512_fmadd_pd(r, q, r2);
        p = _mm512_add_pd(r1, p);

        __m512i j = _mm512_and_epi64(ki, mMod);
        __m512d top = avx512_permute_x4var_pd(mTable_top_0, mTable_top_1,
                                  mTable_top_2, mTable_top_3, j);
        __m512d tail = avx512_permute_x4var_pd(mTable_tail_0, mTable_tail_1,
                                  mTable_tail_2, mTable_tail_3, j);

        __m512d s = _mm512_add_pd(top, tail);
        __m512d res = _mm512_fmadd_pd(s, p, tail);
        res = _mm512_add_pd(res, top);
        res = _mm512_scalef_pd(res, _mm512_div_pd(kd, _mm512_set1_pd(32)));

        res = avx512_set_masked_lanes_pd(res, _mm512_add_pd(x, ones_d),
                                        nearzero_mask);
        res = avx512_set_masked_lanes_pd(res, _mm512_set1_pd(NPY_NAN),
                                        nan_mask);
        res = avx512_set_masked_lanes_pd(res, mTH_inf, xmax_mask);
        res = avx512_set_masked_lanes_pd(res, zeros_d, xmin_mask);

        _mm512_mask_storeu_pd(op, load_mask, res);

        ip += num_lanes * stride;
        op += num_lanes;
        num_remaining_elements -= num_lanes;
    }
    if (npyv_tobits_b64(overflow_mask)) {
        npy_set_floatstatus_overflow();
    }

    if (npyv_tobits_b64(underflow_mask)) {
        npy_set_floatstatus_underflow();
    }
}

#endif // SIMD_AVX512F_NOCLANG_BUG
#endif // NPY_CAN_LINK_SVML


#if SIMD_ARM
#include "simd/simd.hpp"
#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;

/* SVE dispatch target: raw-ACLE FEXPA f64 kernel; NEON/ASIMD fall back to
 * the portable Highway advsimd kernel below. */
#if defined(NPY_HAVE_SVE)
#define NPY_EXPLOG_SVE_FEXPA 1
#else
#define NPY_EXPLOG_SVE_FEXPA 0
#endif

/* ---- Highway (Google Highway) AOR advsimd exp kernels --------------------
 * Algorithm ported from ARM optimized-routines math/aarch64/advsimd/expf.c
 * and exp.c, expressed with Highway
 * scalable vectors following the official numpy SIMD mode (ScalableTag +
 * HWY_ATTR, LoadN/StoreN/GatherIndexN tails, GatherIndex table lookup).
 *
 * numpy contract adaptations (see PR #200 / commit c73289f):
 *  - NaN sign+payload passthrough; +-inf mapped to constants. NaN/inf lanes
 *    are sanitized to 0 before the reduction so inf - inf never raises a
 *    spurious invalid, and the s1*s1 overflow term is sanitized so lanes
 *    where it is not selected never flag.
 *  - the s1*s2 bias-split special path keeps OF/UF raised naturally by the
 *    split arithmetic (no libm, no manual flag writes).
 *  - f32 tiny-remainder product flooring (|r|>=2^-21 even powers); f64 tiny
 *    nonzero |x|<2^-30 routed to the special path (1 + x*ln2 / 1 + x).
 */
const hn::ScalableTag<float> exp_f32;
const hn::ScalableTag<uint32_t> exp_u32;
using vec_x_f32 = hn::Vec<decltype(exp_f32)>;
using vec_x_u32 = hn::Vec<decltype(exp_u32)>;

const hn::ScalableTag<double> exp_f64;
const hn::ScalableTag<int64_t> exp_s64;
const hn::ScalableTag<uint64_t> exp_u64;
using vec_x_f64 = hn::Vec<decltype(exp_f64)>;
using vec_x_u64 = hn::Vec<decltype(exp_u64)>;

template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE vtype
exp_load_vector(type_t *src, npy_intp ssrc, npy_intp len)
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
exp_store_vector(vtype vec, type_t *dst, npy_intp sdst, npy_intp len)
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


HWY_ATTR NPY_FINLINE vec_x_f64
exp_lookup_sbits(vec_x_u64 idx)
{
    if constexpr (hn::MaxLanes(exp_f64) == 2) {
        uint64_t i0 = hn::ExtractLane(idx, 0);
        uint64_t i1 = hn::ExtractLane(idx, 1);
        /* table holds bit patterns: insert as u64 then BitCast (a numeric
         * (double) cast would convert the integer, not reinterpret it). */
        vec_x_u64 t = hn::InsertLane(hn::Zero(exp_u64), 0, neon_exp_table[i0]);
        t = hn::InsertLane(t, 1, neon_exp_table[i1]);
        return hn::BitCast(exp_f64, t);
    } else {
        return hn::GatherIndex(exp_f64, (const double *)neon_exp_table,
                               hn::BitCast(exp_s64, idx));
    }
}

#if NPY_EXPLOG_SVE_FEXPA
/* ---- SVE-native FEXPA exp f64 kernel (SVE dispatch target only) ---------
 * ARM optimized-routines math/aarch64/sve/exp.c (svexpa hardware 64-entry
 * 2^(i/64) table, no gather), numpy-contract adapted: quiet integer abs-bits
 * special detection, NaN sign+payload passthrough with sNaN quieting, +-inf
 * constants, NaN/inf/tiny sanitized before the heavy path, nonzero |x|<2^-30
 * via fl(1+x), predicated underflowing product for exact-subnormal UF parity,
 * two-level branch guarding the clamp+offset+FSCALE path. Raw ACLE is
 * confined to this kernel; the Highway loop, load/store helpers, HALF and
 * wrappers are unchanged (hn::Vec<ScalableTag<double>> is svfloat64_t).
 */

 /* AOR SVE ACLE short-name helpers (sv_math.h subset). */
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
#define InfBound 0x1.63p+9
#define ZeroBound -0x1.74910d52d3053p+9

static const struct exp_sve_data
{
    double c0, c2;
    double c1, c3;
    double ln2_hi, ln2_lo;
    double inv_ln2, shift, inf_bound, zero_bound;
} exp_sve_data = {
    .c0 = 0x1.fffffffffdbcdp-2,
    .c2 = 0x1.555573c6a9f7dp-5,
    .c1 = 0x1.555555555444cp-3,
    .c3 = 0x1.1111266d28935p-7,
    .ln2_hi = 0x1.62e42fefa3800p-1,
    .ln2_lo = 0x1.ef35793c76730p-45,
    .inv_ln2 = 0x1.71547652b82fep+0,
    .shift = 0x1.800000000ffc0p+46,
    .inf_bound = InfBound,
    .zero_bound = ZeroBound,

};

static inline svfloat64_t
exp_inline(svfloat64_t x, const svbool_t pg, const struct exp_sve_data *d)
{
    svfloat64_t z = svmla_x(pg, sv_f64(d->shift), x, d->inv_ln2);
    svfloat64_t n = svsub_x(pg, z, d->shift);

    svfloat64_t ln2_hi_lo = svld1rq(svptrue_b64(), &d->ln2_hi);
    svfloat64_t r = x;
    r = svmls_lane(r, n, ln2_hi_lo, 0);
    r = svmls_lane(r, n, ln2_hi_lo, 1);

    svfloat64_t c13 = svld1rq(svptrue_b64(), &d->c1);
    svfloat64_t r2 = svmul_x(svptrue_b64(), r, r);
    svfloat64_t p01 = svmla_lane(sv_f64(d->c0), r, c13, 0);
    svfloat64_t p23 = svmla_lane(sv_f64(d->c2), r, c13, 1);
    svfloat64_t p04 = svmla_x(pg, p01, p23, r2);
    svfloat64_t poly = svmla_x(pg, r, p04, r2);

    svfloat64_t scale = svexpa(svreinterpret_u64(z));

    return svmla_x(pg, scale, scale, poly);
}

static svfloat64_t __attribute__((always_inline)) inline
special_case(svfloat64_t x, const svbool_t pg, const struct exp_sve_data *d)
{
    svbool_t is_negative = svcmplt(pg, x, 0.0);
    svfloat64_t offset = svneg_m(sv_f64(53.0), is_negative, sv_f64(53.0));
    svint64_t scale_adjust = svneg_m(sv_s64(53), is_negative, sv_s64(53));

    x = svmin_x(pg, svmax_x(pg, x, d->zero_bound), d->inf_bound);

    svfloat64_t z = svmla_x(pg, sv_f64(d->shift), x, d->inv_ln2);
    svfloat64_t n = svsub_x(pg, z, d->shift);

    svfloat64_t ln2_hi_lo = svld1rq(svptrue_b64(), &d->ln2_hi);
    svfloat64_t r = x;
    r = svmls_lane(r, n, ln2_hi_lo, 0);
    r = svmls_lane(r, n, ln2_hi_lo, 1);

    svfloat64_t c13 = svld1rq(svptrue_b64(), &d->c1);
    svfloat64_t r2 = svmul_x(svptrue_b64(), r, r);
    svfloat64_t p01 = svmla_lane(sv_f64(d->c0), r, c13, 0);
    svfloat64_t p23 = svmla_lane(sv_f64(d->c2), r, c13, 1);
    svfloat64_t p04 = svmla_x(pg, p01, p23, r2);
    svfloat64_t poly = svmla_x(pg, r, p04, r2);

    z = svsub_x(pg, z, offset);
    svfloat64_t scale = svexpa(svreinterpret_u64(z));

    svfloat64_t y = svmla_x(pg, scale, scale, poly);

    y = svscale_x(pg, y, scale_adjust);
    /* glibc raises UF for every tiny result, even exact ones (exp2(-1074));
       FSCALE/FMUL only flag inexact tininess on HIP12.  Supply UF via an
       inexact underflowing product on lanes whose result is subnormal/zero
       (all such lanes legitimately underflow; masked ops keep other lanes
       quiet).  y + 0 is exact, so the value is unchanged.  */
    svuint64_t yexp
        = svand_x(pg, svreinterpret_u64(y), sv_u64(0x7ff0000000000000ULL));
    svbool_t tiny_out = svcmpeq_u64(pg, yexp, sv_u64(0));
    svfloat64_t uf = svmul_m(tiny_out, sv_f64(0x1p-600), sv_f64(0x1p-600));
    return svadd_m(tiny_out, y, uf);
}

static inline svfloat64_t
exp_f64_sve_fexpa(svfloat64_t x, const svbool_t pg)
{
    const struct exp_sve_data *d = ptr_barrier(&exp_sve_data);
    svuint64_t abs_bits
        = svand_x(pg, svreinterpret_u64(x), sv_u64(0x7fffffffffffffffULL));
     /* quiet integer detection: catches NaN/inf too (abs bits > bound bits) */
    svbool_t special
        = svcmpge_u64(pg, abs_bits, sv_u64(0x4086232bdd7abcd2ULL));
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
        /* Constant special results computed on raw x for every lane: fl(1+x)
           and the hardware-quietize x+0 both see raw sNaN and raise INVALID
           (libm parity). Hoisted so they are shared by the cheap fast path
           and the final patches; qn feeds the nan_m select, so it is live
           whenever a NaN lane exists and cannot be dead-code-eliminated. */
        svfloat64_t tiny_res = svadd_x(pg, sv_f64(1.0), x);
        svfloat64_t qn = svadd_x(pg, x, sv_f64(0.0));
        svfloat64_t inf_res = svmax_x(pg, x, sv_f64(0.0));
        svfloat64_t y;
        if (svptest_any(pg, sp_finite)) {
            /* finite OF/UF lanes present: one heavy compute for all lanes
               (clamp + offset + FEXPA + FSCALE), correct on clean lanes too */
            y = special_case(xs, pg, d);
        } else {
            /* NaN/inf/tiny/zero only (no finite-special). If ALL lanes are
               cheap (NaN/inf/tiny/0), skip FEXPA+polynomial entirely and use
               constant selects; otherwise light clean-form compute on
               sanitized input. */
            svbool_t small
                = svcmplt_u64(pg, abs_bits, sv_u64(0x3e10000000000000ULL));
            svbool_t cheap = svorr_z(pg, svorr_z(pg, nan_m, inf_m), small);
            if (!svptest_any(pg, svnot_b_z(pg, cheap))) {
                y = svsel_f64(nan_m, qn, inf_res);
                y = svsel_f64(small, tiny_res, y);
            } else {
                svfloat64_t xc
                    = svsel_f64(svorr_z(pg, special, tiny), sv_f64(0.0), x);
                y = exp_inline(xc, pg, d);
            }
        }
        y = svsel_f64(tiny, tiny_res, y);
        /* NaN passthrough: x+0 hardware-quietizes sNaN (sign+payload
           preserved) and raises INVALID, exactly as scalar libm; qNaN
           unchanged. */
        y = svsel_f64(nan_m, qn, y);
        y = svsel_f64(inf_m, inf_res, y);
        return y;
    }
    return exp_inline(x, pg, d);
}

HWY_ATTR NPY_FINLINE vec_x_f64
exp_f64_kernel(vec_x_f64 x)
{
    return exp_f64_sve_fexpa(x, svptrue_b64());
}

#else
HWY_ATTR NPY_FINLINE vec_x_f64
exp_f64_kernel(vec_x_f64 x)
{
    const auto D = exp_f64;
    const vec_x_f64 zero = hn::Zero(D);
    const vec_x_f64 one = hn::Set(D, 1.0);
    const vec_x_f64 shift = hn::Set(D, 0x1.8p+52);
    const vec_x_f64 inv_ln2 = hn::Set(D, 0x1.71547652b82fep7);
    const vec_x_f64 ln2_hi = hn::Set(D, -0x1.62e42fefa39efp-8);
    const vec_x_f64 ln2_lo = hn::Set(D, -0x1.abc9e3b39803f3p-63);
    const vec_x_f64 c0 = hn::Set(D, 0x1.ffffffffffd43p-2);
    const vec_x_f64 c1 = hn::Set(D, 0x1.55555c75adbb2p-3);
    const vec_x_f64 c2 = hn::Set(D, 0x1.55555da646206p-5);
    const vec_x_u64 abs_mask = hn::Set(exp_u64, 0x7fffffffffffffffull);
    const vec_x_u64 inf_bits = hn::Set(exp_u64, 0x7ff0000000000000ull);
    const vec_x_u64 bound_bits = hn::Set(exp_u64, 0x4086232bdd76683cull); /*708.396*/
    const vec_x_u64 tiny_bits = hn::Set(exp_u64, 0x3e10000000000000ull); /*2^-30*/
    const vec_x_u64 idx_mask = hn::Set(exp_u64, V_EXP_INDEX_MASK);
    const vec_x_u64 quiet_bit = hn::Set(exp_u64, 0x0008000000000000ull);
    vec_x_u64 bits = hn::BitCast(exp_u64, x);
    vec_x_u64 abs_bits = hn::And(bits, abs_mask);
    auto special_m = hn::Gt(abs_bits, bound_bits);
    auto tiny_m = hn::And(hn::Lt(abs_bits, tiny_bits),
                          hn::Gt(abs_bits, hn::Zero(exp_u64)));  /* nonzero |x|<2^-30 */
    vec_x_f64 result;
    if (__builtin_expect(!hn::AllFalse(exp_u64, hn::Or(special_m, tiny_m)), 0)) {
        auto nan_m = hn::Gt(abs_bits, inf_bits);
        auto inf_m = hn::Eq(abs_bits, inf_bits);
        auto small_m = hn::Lt(abs_bits, tiny_bits);  /* incl 0 */
        /* cheap = NaN/inf/tiny/0 only; finite-large (cmp_f) needs heavy path */
        auto cheap_m = hn::Or(hn::Or(nan_m, inf_m), small_m);
        /* fl(1+x) for tiny/0; the add sees raw sNaN on every lane and raises
         * INVALID (libm parity). NaN quiet-bit OR preserves sign+payload. */
        vec_x_f64 tiny_res = hn::Add(one, x);
        vec_x_f64 nan_res = hn::BitCast(D, hn::Or(bits, quiet_bit));
        vec_x_f64 inf_res = hn::Max(x, zero);
        if (__builtin_expect(hn::AllTrue(exp_u64, cheap_m), 0)) {
            /* All lanes cheap-special: skip polynomial + table + bias split
             * entirely, constant selects only. */
            result = hn::IfThenElse(hn::RebindMask(D, nan_m), nan_res, inf_res);
            result = hn::IfThenElse(hn::RebindMask(D, small_m), tiny_res, result);
            return result;
        }
        /* Mixed path: sanitize + polynomial + bias-split + patch. */
        auto cmp_f = hn::And(special_m, hn::Lt(abs_bits, inf_bits));
        auto care_m = hn::Or(hn::Or(nan_m, inf_m), tiny_m);
        vec_x_f64 xs = hn::IfThenElse(hn::RebindMask(D, care_m), zero, x);
        /* Huge |x| (>1024) breaks range reduction (catastrophic cancellation
         * makes r*r overflow -> spurious OF). Such lanes are cmp2 and take
         * r_ov, so sanitize them to +/-1024 (sign preserved, |n| still >
         * ScaleBound) before the polynomial. */
        auto huge_m = hn::Gt(abs_bits, hn::Set(exp_u64, 0x4090000000000000ull));
        vec_x_u64 xs_huge = hn::Or(
            hn::And(bits, hn::Set(exp_u64, 0x8000000000000000ull)),
            hn::Set(exp_u64, 0x4090000000000000ull));
        xs = hn::IfThenElse(hn::RebindMask(D, huge_m), hn::BitCast(D, xs_huge), xs);
        vec_x_f64 z = hn::MulAdd(xs, inv_ln2, shift);
        vec_x_u64 u = hn::BitCast(exp_u64, z);
        vec_x_f64 n = hn::Sub(z, shift);
        vec_x_f64 r = hn::MulAdd(n, ln2_hi, xs);
        r = hn::MulAdd(n, ln2_lo, r);
        vec_x_u64 e = hn::ShiftLeft<52 - V_EXP_TABLE_BITS>(u);
        vec_x_f64 tab = exp_lookup_sbits(hn::And(u, idx_mask));
        vec_x_f64 scale = hn::BitCast(D, hn::Add(hn::BitCast(exp_u64, tab), e));
        vec_x_f64 r2 = hn::Mul(r, r);
        vec_x_f64 p = hn::MulAdd(r, c1, c0);
        p = hn::MulAdd(r2, c2, p);
        vec_x_f64 poly = hn::MulAdd(p, r2, r);
        vec_x_f64 scale_r = hn::IfThenElse(hn::RebindMask(D, cmp_f), one, scale);
        result = hn::MulAdd(poly, scale_r, scale_r);
        if (__builtin_expect(!hn::AllFalse(exp_u64, cmp_f), 0)) {
            const vec_x_u64 sp_offset = hn::Set(exp_u64, 0x6000000000000000ull);
            const vec_x_u64 sp_bias1 = hn::Set(exp_u64, 0x7000000000000000ull);
            const vec_x_u64 sp_bias2 = hn::Set(exp_u64, 0x3010000000000000ull);
            const vec_x_f64 scale_bound = hn::Set(D, 163840.0);
            vec_x_u64 b = hn::IfThenElse(
                hn::RebindMask(exp_u64, hn::Le(n, zero)),
                sp_offset, hn::Zero(exp_u64));
            vec_x_f64 s1 = hn::BitCast(D, hn::Sub(sp_bias1, b));
            vec_x_f64 s2 = hn::BitCast(D, hn::Add(
                hn::Sub(hn::BitCast(exp_u64, scale), sp_bias2), b));
            auto cmp2 = hn::Gt(hn::Abs(n), scale_bound);
            /* r_ov selected only by cmp2 (extreme OF/UF). Gate s1_ov by cmp2
             * so a discarded r_ov on moderate-negative lanes cannot leak a
             * spurious UF. */
            vec_x_f64 s1_ov = hn::IfThenElse(cmp2, s1, one);
            vec_x_f64 r_ov = hn::Mul(s1_ov, s1_ov);
            /* cmp2 lanes select r_ov; sanitize their s2 (inf for huge |x|) so
             * the discarded r_sp cannot leak a spurious OF. */
            vec_x_f64 s2_safe = hn::IfThenElse(cmp2, one, s2);
            vec_x_f64 r_sp = hn::Mul(hn::MulAdd(poly, s2_safe, s2_safe), s1);
            vec_x_f64 sp_val = hn::IfThenElse(cmp2, r_ov, r_sp);
            result = hn::IfThenElse(hn::RebindMask(D, cmp_f), sp_val, result);
            /* libm raises UF for every subnormal/zero RESULT (even exact).
             * HIP12 FMUL does not flag exact tininess, so supply UF via an
             * inexact underflowing product gated on result exponent==0 (value
             * unchanged: result+0). Non-subnormal lanes multiply 1*1. */
            vec_x_u64 res_exp = hn::And(hn::BitCast(exp_u64, result), inf_bits);
            auto sub_m = hn::Eq(res_exp, hn::Zero(exp_u64));
            vec_x_f64 uf_a = hn::IfThenElse(hn::RebindMask(D, sub_m),
                                            hn::Set(D, 0x1p-600), one);
            vec_x_f64 uf_prod = hn::Mul(uf_a, uf_a);
            vec_x_f64 res_uf = hn::Add(result, uf_prod);
            result = hn::IfThenElse(hn::RebindMask(D, sub_m), res_uf, result);
        }
        /* tiny -> fl(1+x); NaN quiet-bit OR; +-inf -> max(x,0). */
        result = hn::IfThenElse(hn::RebindMask(D, tiny_m), tiny_res, result);
        result = hn::IfThenElse(hn::RebindMask(D, nan_m), nan_res, result);
        result = hn::IfThenElse(hn::RebindMask(D, inf_m), inf_res, result);
    } else {
        vec_x_f64 z = hn::MulAdd(x, inv_ln2, shift);
        vec_x_u64 u = hn::BitCast(exp_u64, z);
        vec_x_f64 n = hn::Sub(z, shift);
        vec_x_f64 r = hn::MulAdd(n, ln2_hi, x);
        r = hn::MulAdd(n, ln2_lo, r);
        vec_x_u64 e = hn::ShiftLeft<52 - V_EXP_TABLE_BITS>(u);
        vec_x_f64 tab = exp_lookup_sbits(hn::And(u, idx_mask));
        vec_x_f64 scale = hn::BitCast(D, hn::Add(hn::BitCast(exp_u64, tab), e));
        vec_x_f64 r2 = hn::Mul(r, r);
        vec_x_f64 p = hn::MulAdd(r, c1, c0);
        p = hn::MulAdd(r2, c2, p);
        vec_x_f64 poly = hn::MulAdd(p, r2, r);
        result = hn::MulAdd(poly, scale, scale);
    }
    return result;
}
#endif /* NPY_EXPLOG_SVE_FEXPA */

HWY_ATTR static void
simd_exp_hwy_DOUBLE(const npy_double *src, npy_intp ssrc,
                    npy_double *dst, npy_intp sdst, npy_intp len)
{
    const int nlanes = hn::Lanes(exp_f64);
    for (; len > 0; len -= nlanes, src += ssrc * nlanes, dst += sdst * nlanes) {
        vec_x_f64 x = exp_load_vector<vec_x_f64>(src, ssrc, len);
        vec_x_f64 out = exp_f64_kernel(x);
        exp_store_vector<vec_x_f64>(out, dst, sdst, len);
    }
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

/********************************************************************************
 ** Defining ufunc inner functions for exp/log
 ********************************************************************************/

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_exp)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
/* exp f32 uses scalar libm for every stride on ARM: the AOR f32 kernel's
 * 1-2 ULP deviation from libm breaks sub-ULP precision assertions
 * (linalg TestDet::test_generalized_sq_cases, decimal=6 tighter than
 * f32 ULP at magnitude >= 16). exp f64/f16 keep the contiguous vector
 * path (f64 has ~9 orders of ULP margin at decimal=6). */
#if defined(SIMD_AVX2_FMA3) || defined(SIMD_AVX512F)
    if (IS_OUTPUT_BLOCKABLE_UNARY(sizeof(npy_float), sizeof(npy_float), 64)) {
        simd_exp_FLOAT((npy_float*)args[1], (npy_float*)args[0], dimensions[0], steps[0]);
    }
    else {
        UNARY_LOOP {
            simd_exp_FLOAT((npy_float *)op1, (npy_float *)ip1, 1, steps[0]);
        }
    }
#else
    UNARY_LOOP {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = npy_expf(in1);
    }
#endif
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_exp)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if SIMD_ARM
    npy_intp len = dimensions[0];
    /* Vectorize only contiguous non-overlapping runs; strided/overlap
     * falls back to scalar libm (policy note in loops_explog.h). */
    if (steps[0] == (npy_intp)sizeof(npy_double) &&
        steps[1] == (npy_intp)sizeof(npy_double) &&
        !is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        simd_exp_hwy_DOUBLE((npy_double *)args[0], 1, (npy_double *)args[1], 1, len);
    } else {
        UNARY_LOOP {
            const npy_double in1 = *(npy_double *)ip1;
            *(npy_double *)op1 = npy_exp(in1);
        }
    }
    npyv_cleanup();
    return;
#elif NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
    const npy_intp len = dimensions[0];

    if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len) &&
            npyv_loadable_stride_f64(steps[0]) &&
            npyv_storable_stride_f64(steps[1])) {
        const npy_double *src = (npy_double*)args[0];
        npy_double *dst = (npy_double*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(src[0]);
        const npy_intp sdst = steps[1] / sizeof(src[0]);

        simd_exp_f64(src, ssrc, dst, sdst, len);
        return;
    }
#else
  #ifdef SIMD_AVX512F_NOCLANG_BUG
      if (IS_OUTPUT_BLOCKABLE_UNARY(sizeof(npy_double), sizeof(npy_double), 64)) {
          AVX512F_exp_DOUBLE((npy_double*)args[1], (npy_double*)args[0], dimensions[0], steps[0]);
          return;
      }
  #endif
#endif
    UNARY_LOOP {
        const npy_double in1 = *(npy_double *)ip1;
        *(npy_double *)op1 = npy_exp(in1);
    }
}

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
    /* Vectorize only contiguous non-overlapping runs; strided/overlap
     * falls back to scalar libm (policy note in loops_explog.h). */
    if (steps[0] == (npy_intp)sizeof(npy_half) &&
        steps[1] == (npy_intp)sizeof(npy_half) &&
        !is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        simd_exp_hwy_HALF((npy_half *)args[0], 1, (npy_half *)args[1], 1, len);
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
