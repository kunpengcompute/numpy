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

/* ---- Highway (Google Highway) AOR advsimd exp kernels --------------------
 * Algorithm ported from ARM optimized-routines math/aarch64/advsimd/expf.c
 * and exp.c (MIT OR Apache-2.0 WITH LLVM-exception), expressed with Highway
 * scalable vectors following the official numpy SIMD mode (ScalableTag +
 * HWY_ATTR, LoadN/StoreN/GatherIndexN tails, GatherIndex table lookup).
 *
 * Copyright (c) 2019-2026, Arm Limited.
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
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
const hn::ScalableTag<int32_t> exp_s32;
const hn::ScalableTag<uint32_t> exp_u32;
using vec_x_f32 = hn::Vec<decltype(exp_f32)>;
using vec_x_s32 = hn::Vec<decltype(exp_s32)>;
using vec_x_u32 = hn::Vec<decltype(exp_u32)>;

const hn::ScalableTag<double> exp_f64;
const hn::ScalableTag<int64_t> exp_s64;
const hn::ScalableTag<uint64_t> exp_u64;
using vec_x_f64 = hn::Vec<decltype(exp_f64)>;
using vec_x_s64 = hn::Vec<decltype(exp_s64)>;
using vec_x_u64 = hn::Vec<decltype(exp_u64)>;

template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE vtype
exp_load_vector(type_t *src, npy_intp ssrc, npy_intp len)
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
exp_store_vector(vtype vec, type_t *dst, npy_intp sdst, npy_intp len)
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
simd_exp_hwy_FLOAT(const npy_float *src, npy_intp ssrc,
                   npy_float *dst, npy_intp sdst, npy_intp len)
{
    const auto D = exp_f32;
    const int nlanes = hn::Lanes(D);
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

    for (; len > 0; len -= nlanes, src += ssrc * nlanes, dst += sdst * nlanes) {
        vec_x_f32 x = exp_load_vector<vec_x_f32>(src, ssrc, len);
        vec_x_u32 abs_bits = hn::And(hn::BitCast(exp_u32, x), abs_mask);
        auto special_m = hn::Ge(abs_bits, bound_bits);
        vec_x_f32 result;
        if (__builtin_expect(!hn::AllFalse(exp_u32, special_m), 0)) {
            auto nan_m = hn::Gt(abs_bits, inf_bits);
            auto inf_m = hn::Eq(abs_bits, inf_bits);
            auto cmp = hn::And(special_m, hn::Lt(abs_bits, inf_bits));
            vec_x_f32 xs = hn::IfThenElse(
                hn::RebindMask(D, hn::Or(nan_m, inf_m)), zero, x);
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
            result = hn::IfThenElse(hn::RebindMask(D, nan_m), x, result);
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
        exp_store_vector<vec_x_f32>(result, dst, sdst, len);
    }
}HWY_ATTR NPY_FINLINE vec_x_f64
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

HWY_ATTR static void
simd_exp_hwy_DOUBLE(const npy_double *src, npy_intp ssrc,
                    npy_double *dst, npy_intp sdst, npy_intp len)
{
    const auto D = exp_f64;
    const int nlanes = hn::Lanes(D);
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

    for (; len > 0; len -= nlanes, src += ssrc * nlanes, dst += sdst * nlanes) {
        vec_x_f64 x = exp_load_vector<vec_x_f64>(src, ssrc, len);
        vec_x_u64 abs_bits = hn::And(hn::BitCast(exp_u64, x), abs_mask);
        auto special_m = hn::Gt(abs_bits, bound_bits);
        auto tiny_m = hn::And(hn::Lt(abs_bits, tiny_bits),
                              hn::Gt(abs_bits, hn::Zero(exp_u64)));  /* nonzero |x|<2^-30 */
        vec_x_f64 result;
        if (__builtin_expect(!hn::AllFalse(exp_u64, hn::Or(special_m, tiny_m)), 0)) {
            auto nan_m = hn::Gt(abs_bits, inf_bits);
            auto inf_m = hn::Eq(abs_bits, inf_bits);
            auto cmp_f = hn::And(special_m, hn::Lt(abs_bits, inf_bits));
            auto care_m = hn::Or(hn::Or(nan_m, inf_m), tiny_m);
            vec_x_f64 xs = hn::IfThenElse(hn::RebindMask(D, care_m), zero, x);
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
                auto cmp_neg = hn::And(hn::RebindMask(D, cmp_f), hn::Le(n, zero));
                auto ov_mask = hn::Or(cmp2, cmp_neg);
                vec_x_f64 s1_ov = hn::IfThenElse(ov_mask, s1, one);
                vec_x_f64 r_ov = hn::Mul(s1_ov, s1_ov);
                vec_x_f64 r_sp = hn::Mul(hn::MulAdd(poly, s2, s2), s1);
                vec_x_f64 sp_val = hn::IfThenElse(cmp2, r_ov, r_sp);
                result = hn::IfThenElse(hn::RebindMask(D, cmp_f), sp_val, result);
            }
            /* exp(x) = fl(1 + x) for |x|<2^-30 (x^2/2 < 2^-61 below half
             * ULP); adding subnormal x to 1 rounds normal, no UF. */
            vec_x_f64 tiny_res = hn::Add(x, one);
            result = hn::IfThenElse(hn::RebindMask(D, tiny_m), tiny_res, result);
            result = hn::IfThenElse(hn::RebindMask(D, nan_m), x, result);
            result = hn::IfThenElse(hn::RebindMask(D, inf_m), hn::Max(x, zero), result);
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
        exp_store_vector<vec_x_f64>(result, dst, sdst, len);
    }
}static void
simd_exp_neon_HALF(const npy_half *src, npy_intp ssrc,
                    npy_half *dst, npy_intp sdst, npy_intp len)
{
  const npy_intp BLOCK_SIZE = 2048;
  const npy_intp VEC_SIZE = 8;
  
  npy_float *tmp_src = (npy_float *)malloc(BLOCK_SIZE * sizeof(npy_float));
  npy_float *tmp_dst = (npy_float *)malloc(BLOCK_SIZE * sizeof(npy_float));
  
  if (!tmp_src || !tmp_dst) {
    if (tmp_src) free(tmp_src);
    if (tmp_dst) free(tmp_dst);
    for (npy_intp i = 0; i < len; i++) {
      npy_float in1 = npy_half_to_float(src[i * ssrc]);
      dst[i * sdst] = npy_float_to_half(npy_expf(in1));
    }
    return;
  }

  for (npy_intp offset = 0; offset < len; offset += BLOCK_SIZE) {
    npy_intp block_len = (len - offset > BLOCK_SIZE) ? BLOCK_SIZE : (len - offset);
    const npy_half *block_src = src + offset * ssrc;
    npy_half *block_dst = dst + offset * sdst;
    
    npy_intp i = 0;
    if (ssrc == 1) {
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
        __builtin_prefetch(block_src + i + 64, 0, 3);
        float16x8_t h0 = vld1q_f16((const float16_t *)(block_src + i));
        float32x4_t f0 = vcvt_f32_f16(vget_low_f16(h0));
        float32x4_t f1 = vcvt_f32_f16(vget_high_f16(h0));
        vst1q_f32(tmp_src + i, f0);
        vst1q_f32(tmp_src + i + 4, f1);
      }
    } else {
      for (; i + 8 <= block_len; i += 8) {
        for (int j = 0; j < 8; j++) {
          tmp_src[i + j] = npy_half_to_float(block_src[(i + j) * ssrc]);
        }
      }
    }
    for (; i < block_len; i++) {
      tmp_src[i] = npy_half_to_float(block_src[i * ssrc]);
    }
    
    simd_exp_hwy_FLOAT(tmp_src, 1, tmp_dst, 1, block_len);
    
    i = 0;
    if (sdst == 1) {
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
        __builtin_prefetch(block_dst + i + 64, 1, 3);
        float32x4_t f0 = vld1q_f32(tmp_dst + i);
        float32x4_t f1 = vld1q_f32(tmp_dst + i + 4);
        float16x4_t h0 = vcvt_f16_f32(f0);
        float16x4_t h1 = vcvt_f16_f32(f1);
        float16x8_t h_combined = vcombine_f16(h0, h1);
        vst1q_f16((float16_t *)(block_dst + i), h_combined);
      }
    } else {
      for (; i + 8 <= block_len; i += 8) {
        for (int j = 0; j < 8; j++) {
          block_dst[(i + j) * sdst] = npy_float_to_half(tmp_dst[i + j]);
        }
      }
    }
    for (; i < block_len; i++) {
      block_dst[i * sdst] = npy_float_to_half(tmp_dst[i]);
    }
  }
  
  free(tmp_src);
  free(tmp_dst);
}

#endif // SIMD_ARM

/********************************************************************************
 ** Defining ufunc inner functions for exp/log
 ********************************************************************************/

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_exp)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_float in1 = *(npy_float *)ip1;
            *(npy_float *)op1 = npy_expf(in1);
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_float *src = (npy_float*)args[0];
        npy_float *dst = (npy_float*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);
        simd_exp_hwy_FLOAT(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            simd_exp_hwy_FLOAT((npy_float *)ip1, 1, (npy_float *)op1, 1, 1);
        }
    }
    return;
#elif defined(SIMD_AVX2_FMA3) || defined(SIMD_AVX512F)
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
#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
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
#elif SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_double in1 = *(npy_double *)ip1;
            *(npy_double *)op1 = npy_exp(in1);
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_double *src = (npy_double*)args[0];
        npy_double *dst = (npy_double*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_double);
        const npy_intp sdst = steps[1] / sizeof(npy_double);
        simd_exp_hwy_DOUBLE(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            const npy_double in1 = *(npy_double *)ip1;
            *(npy_double *)op1 = npy_exp(in1);
        }
    }
    return;
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
#if SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_expf(in1));
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_half *src = (npy_half*)args[0];
        npy_half *dst = (npy_half*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_half);
        const npy_intp sdst = steps[1] / sizeof(npy_half);
        simd_exp_neon_HALF(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_expf(in1));
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
