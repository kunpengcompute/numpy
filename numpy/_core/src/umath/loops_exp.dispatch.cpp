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

static void
simd_exp_neon_DOUBLE(const npy_double *src, npy_intp ssrc,
                       npy_double *dst, npy_intp sdst, npy_intp len)
{
  const float64x2_t pos_inf = vdupq_n_f64(NPY_INFINITY);
  const float64x2_t neg_inf = vdupq_n_f64(-NPY_INFINITY);
  const float64x2_t nan_val = vdupq_n_f64(NPY_NAN);
  const float64x2_t zero_val = vdupq_n_f64(0.0);
  const float64x2_t inv_ln2 = vdupq_n_f64(0x1.71547652b82fep7);
  const float64x2_t shift = vdupq_n_f64(0x1.8p+52);
  const float64x2_t ln2_hi = vdupq_n_f64(0x1.62e42fefa39efp-8);
  const float64x2_t ln2_lo = vdupq_n_f64(0x1.abc9e3b39803fp-63);
  const float64x2_t c0 = vdupq_n_f64(0x1.ffffffffffd43p-2);
  const float64x2_t c1 = vdupq_n_f64(0x1.55555c75adbb2p-3);
  const float64x2_t c2 = vdupq_n_f64(0x1.55555da646206p-5);
  const uint64x2_t all_ones = vdupq_n_u64(0xFFFFFFFFFFFFFFFFULL);
  const float64x2_t exp_max = vdupq_n_f64(709.782712893384);
  const float64x2_t exp_min = vdupq_n_f64(-745.1332191019411);

  const npy_double *infp = src;
  npy_double *outp = dst;
  npy_intp remaining = len;

  feclearexcept(FE_ALL_EXCEPT);

  const npy_intp UNROLL = 8;
  const npy_intp VEC_SIZE = 2;

  while (remaining >= UNROLL * VEC_SIZE) {
    float64x2_t x[UNROLL];

    if (ssrc == 1) {
      __builtin_prefetch(infp + 64, 0, 3);
      for (int i = 0; i < UNROLL; i++)
        x[i] = vld1q_f64(infp + i * VEC_SIZE);
    } else {
      for (int i = 0; i < UNROLL; i++) {
        double vals[2] = {infp[(2*i) * ssrc], infp[(2*i+1) * ssrc]};
        x[i] = vld1q_f64(vals);
      }
    }

    uint64x2_t need_special[UNROLL], is_nan[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      need_special[i] = vceqq_f64(x[i], pos_inf);
      need_special[i] = vorrq_u64(need_special[i], vceqq_f64(x[i], neg_inf));
      is_nan[i] = veorq_u64(vceqq_f64(x[i], x[i]), all_ones);
      need_special[i] = vorrq_u64(need_special[i], is_nan[i]);
      need_special[i] = vorrq_u64(need_special[i], vcgtq_f64(x[i], exp_max));
      need_special[i] = vorrq_u64(need_special[i], vcltq_f64(x[i], exp_min));
    }

    float64x2_t safe_val[UNROLL], z[UNROLL], n[UNROLL], r[UNROLL];
    uint64x2_t u[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      safe_val[i] = vbslq_f64(need_special[i], zero_val, x[i]);
      z[i] = vfmaq_f64(shift, safe_val[i], inv_ln2);
      u[i] = vreinterpretq_u64_f64(z[i]);
      n[i] = vsubq_f64(z[i], shift);
      r[i] = vfmsq_f64(vfmsq_f64(safe_val[i], n[i], ln2_hi), n[i], ln2_lo);
    }

    float64x2_t scale[UNROLL], r2[UNROLL], poly[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      uint64x2_t e = vshlq_n_u64(u[i], 52 - V_EXP_TABLE_BITS);
      uint64x2_t tab = neon_exp_lookup_sbits(u[i]);
      scale[i] = vreinterpretq_f64_u64(vaddq_u64(tab, e));
      r2[i] = vmulq_f64(r[i], r[i]);
      poly[i] = vfmaq_f64(vfmaq_f64(c0, r[i], c1), r2[i], c2);
      poly[i] = vfmaq_f64(r[i], poly[i], r2[i]);
    }

    float64x2_t result[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      result[i] = vfmaq_f64(scale[i], poly[i], scale[i]);
      result[i] = vbslq_f64(is_nan[i], x[i], result[i]);
      result[i] = vbslq_f64(vceqq_f64(x[i], pos_inf), pos_inf, result[i]);
      result[i] = vbslq_f64(vceqq_f64(x[i], neg_inf), zero_val, result[i]);
      result[i] = vbslq_f64(vcgtq_f64(x[i], exp_max), pos_inf, result[i]);
      result[i] = vbslq_f64(vcltq_f64(x[i], exp_min), zero_val, result[i]);
    }

    if (sdst == 1) {
      __builtin_prefetch(outp + 64, 1, 3);
      for (int i = 0; i < UNROLL; i++)
        vst1q_f64(outp + i * VEC_SIZE, result[i]);
    } else {
      for (int i = 0; i < UNROLL; i++) {
        outp[(2*i) * sdst] = vgetq_lane_f64(result[i], 0);
        outp[(2*i+1) * sdst] = vgetq_lane_f64(result[i], 1);
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
      if (current_len == 1) x = vld1q_dup_f64(infp);
      else x = vld1q_f64(infp);
    } else {
      double vals[2] = {infp[0], (remaining > 1) ? infp[ssrc] : infp[0]};
      x = vld1q_f64(vals);
    }

    uint64x2_t is_nan = vceqq_f64(x, x);
    is_nan = veorq_u64(is_nan, all_ones);
    uint64x2_t is_pos_inf = vceqq_f64(x, pos_inf);
    uint64x2_t is_neg_inf = vceqq_f64(x, neg_inf);
    uint64x2_t is_overflow = vcgtq_f64(x, exp_max);
    uint64x2_t is_underflow = vcltq_f64(x, exp_min);
    uint64x2_t special_mask = vorrq_u64(vorrq_u64(is_nan, is_pos_inf), is_neg_inf);
    special_mask = vorrq_u64(special_mask, vorrq_u64(is_overflow, is_underflow));
    float64x2_t safe_x = vbslq_f64(special_mask, zero_val, x);

    float64x2_t z = vfmaq_f64(shift, safe_x, inv_ln2);
    uint64x2_t u = vreinterpretq_u64_f64(z);
    float64x2_t n = vsubq_f64(z, shift);
    float64x2_t r = vfmsq_f64(vfmsq_f64(safe_x, n, ln2_hi), n, ln2_lo);
    uint64x2_t e = vshlq_n_u64(u, 52 - V_EXP_TABLE_BITS);

    u = neon_exp_lookup_sbits(u);
    float64x2_t scale = vreinterpretq_f64_u64(vaddq_u64(u, e));

    float64x2_t r2 = vmulq_f64(r, r);
    float64x2_t poly = vfmaq_f64(vfmaq_f64(c0, r, c1), r2, c2);
    poly = vfmaq_f64(r, poly, r2);
    float64x2_t result = vfmaq_f64(scale, poly, scale);

    result = vbslq_f64(is_nan, nan_val, result);
    result = vbslq_f64(is_pos_inf, pos_inf, result);
    result = vbslq_f64(is_neg_inf, zero_val, result);
    result = vbslq_f64(is_overflow, pos_inf, result);
    result = vbslq_f64(is_underflow, zero_val, result);

    if (sdst == 1) {
      if (current_len == 1) vst1q_lane_f64(outp, result, 0);
      else vst1q_f64(outp, result);
    } else {
      outp[0] = vgetq_lane_f64(result, 0);
      if (current_len > 1) outp[sdst] = vgetq_lane_f64(result, 1);
    }

    infp += ssrc * current_len;
    outp += sdst * current_len;
    remaining -= current_len;
  }
  feclearexcept(FE_ALL_EXCEPT);
}

static void
simd_exp_neon_FLOAT(const npy_float *src, npy_intp ssrc,
                     npy_float *dst, npy_intp sdst, npy_intp len)
{
  const float32x4_t pos_inf = vdupq_n_f32(NPY_INFINITYF);
  const float32x4_t neg_inf = vdupq_n_f32(-NPY_INFINITYF);
  const float32x4_t nan_val = vdupq_n_f32(NPY_NANF);
  const float32x4_t zero_val = vdupq_n_f32(0.0f);
  const float32x4_t inv_ln2 = vdupq_n_f32(0x1.715476p+0f);
  const float32x4_t shift = vdupq_n_f32(0x1.8p+23f);
  const float32x4_t ln2_hi = vdupq_n_f32(0x1.62e4p-1f);
  const float32x4_t ln2_lo = vdupq_n_f32(0x1.7f7d1cp-20f);
  const float32x4_t c0 = vdupq_n_f32(0x1.111112p-7f);
  const float32x4_t c1 = vdupq_n_f32(0x1.555556p-5f);
  const float32x4_t c2 = vdupq_n_f32(0x1.555556p-3f);
  const float32x4_t c3 = vdupq_n_f32(0x1.0p-1f);
  const float32x4_t c4 = vdupq_n_f32(0x1.0p+0f);
  const float32x4_t c5 = vdupq_n_f32(0x1.6c16c16cp-10f);
  const uint32x4_t exponent_bias = vdupq_n_u32(0x3f800000);
  const float32x4_t exp_max = vdupq_n_f32(88.72283935546875f);
  const float32x4_t exp_min = vdupq_n_f32(-103.97208404541015625f);

  const npy_float *infp = src;
  npy_float *outp = dst;
  npy_intp remaining = len;

  feclearexcept(FE_ALL_EXCEPT);

  const npy_intp UNROLL = 8;
  const npy_intp VEC_SIZE = 4;

  while (remaining >= UNROLL * VEC_SIZE) {
    float32x4_t x[UNROLL];

    if (ssrc == 1) {
      __builtin_prefetch(infp + 128, 0, 3);
      for (int i = 0; i < UNROLL; i++)
        x[i] = vld1q_f32(infp + i * VEC_SIZE);
    } else {
      for (int i = 0; i < UNROLL; i++) {
        float vals[4];
        for (int j = 0; j < 4; j++) vals[j] = infp[(i*4+j) * ssrc];
        x[i] = vld1q_f32(vals);
      }
    }

    uint32x4_t need_special[UNROLL], is_nan[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      need_special[i] = vceqq_f32(x[i], pos_inf);
      need_special[i] = vorrq_u32(need_special[i], vceqq_f32(x[i], neg_inf));
      is_nan[i] = vmvnq_u32(vceqq_f32(x[i], x[i]));
      need_special[i] = vorrq_u32(need_special[i], is_nan[i]);
      need_special[i] = vorrq_u32(need_special[i], vcgtq_f32(x[i], exp_max));
      need_special[i] = vorrq_u32(need_special[i], vcltq_f32(x[i], exp_min));
    }

    float32x4_t safe_val[UNROLL], z[UNROLL], n[UNROLL], r[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      safe_val[i] = vbslq_f32(need_special[i], zero_val, x[i]);
      z[i] = vfmaq_f32(shift, safe_val[i], inv_ln2);
      n[i] = vsubq_f32(z[i], shift);
      r[i] = vfmsq_f32(vfmsq_f32(safe_val[i], n[i], ln2_hi), n[i], ln2_lo);
    }

    float32x4_t scale[UNROLL], r2[UNROLL], r4[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      int32x4_t ni = vcvtq_s32_f32(n[i]);
      uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_s32(ni), 23);
      scale[i] = vreinterpretq_f32_u32(vaddq_u32(e, exponent_bias));
      r2[i] = vmulq_f32(r[i], r[i]);
      r4[i] = vmulq_f32(r2[i], r2[i]);
    }

    float32x4_t poly[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      float32x4_t p = vfmaq_f32(c1, r[i], c0);
      float32x4_t q = vfmaq_f32(c3, r[i], c2);
      q = vfmaq_f32(q, p, r2[i]);
      poly[i] = vfmaq_f32(vmulq_f32(c4, r[i]), q, r2[i]);
      float32x4_t r6 = vmulq_f32(r4[i], r2[i]);
      poly[i] = vfmaq_f32(poly[i], c5, r6);
    }

    float32x4_t result[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      result[i] = vfmaq_f32(scale[i], poly[i], scale[i]);
      result[i] = vbslq_f32(is_nan[i], x[i], result[i]);
      result[i] = vbslq_f32(vceqq_f32(x[i], pos_inf), pos_inf, result[i]);
      result[i] = vbslq_f32(vceqq_f32(x[i], neg_inf), zero_val, result[i]);
      result[i] = vbslq_f32(vcgtq_f32(x[i], exp_max), pos_inf, result[i]);
      result[i] = vbslq_f32(vcltq_f32(x[i], exp_min), zero_val, result[i]);
    }

    if (sdst == 1) {
      __builtin_prefetch(outp + 128, 1, 3);
      for (int i = 0; i < UNROLL; i++)
        vst1q_f32(outp + i * VEC_SIZE, result[i]);
    } else {
      for (int i = 0; i < UNROLL; i++) {
        for (int j = 0; j < 4; j++)
          outp[(i*4+j) * sdst] = vgetq_lane_f32(result[i], j);
      }
    }

    infp += ssrc * UNROLL * VEC_SIZE;
    outp += sdst * UNROLL * VEC_SIZE;
    remaining -= UNROLL * VEC_SIZE;
  }

  while (remaining > 0) {
    int current_len = (remaining < VEC_SIZE) ? remaining : VEC_SIZE;
    float32x4_t x;
    if (ssrc == 1) {
      x = vld1q_f32(infp);
    } else {
      float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int i = 0; i < current_len; i++) vals[i] = infp[i * ssrc];
      x = vld1q_f32(vals);
    }

    uint32x4_t is_nan = vceqq_f32(x, x);
    is_nan = vmvnq_u32(is_nan);
    uint32x4_t is_pos_inf = vceqq_f32(x, pos_inf);
    uint32x4_t is_neg_inf = vceqq_f32(x, neg_inf);
    uint32x4_t is_overflow = vcgtq_f32(x, exp_max);
    uint32x4_t is_underflow = vcltq_f32(x, exp_min);
    uint32x4_t special_mask = vorrq_u32(vorrq_u32(is_nan, is_pos_inf), is_neg_inf);
    special_mask = vorrq_u32(special_mask, vorrq_u32(is_overflow, is_underflow));
    float32x4_t safe_x = vbslq_f32(special_mask, zero_val, x);

    float32x4_t z = vfmaq_f32(shift, safe_x, inv_ln2);
    float32x4_t n = vsubq_f32(z, shift);
    float32x4_t r = vfmsq_f32(vfmsq_f32(safe_x, n, ln2_hi), n, ln2_lo);

    int32x4_t ni = vcvtq_s32_f32(n);
    uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_s32(ni), 23);
    float32x4_t scale = vreinterpretq_f32_u32(vaddq_u32(e, exponent_bias));

    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t r4 = vmulq_f32(r2, r2);
    float32x4_t p = vfmaq_f32(c1, r, c0);
    float32x4_t q = vfmaq_f32(c3, r, c2);
    q = vfmaq_f32(q, p, r2);
    float32x4_t poly = vfmaq_f32(vmulq_f32(c4, r), q, r2);
    float32x4_t r6 = vmulq_f32(r4, r2);
    poly = vfmaq_f32(poly, c5, r6);
    float32x4_t result = vfmaq_f32(scale, poly, scale);

    result = vbslq_f32(is_nan, nan_val, result);
    result = vbslq_f32(is_pos_inf, pos_inf, result);
    result = vbslq_f32(is_neg_inf, zero_val, result);
    result = vbslq_f32(is_overflow, pos_inf, result);
    result = vbslq_f32(is_underflow, zero_val, result);

    if (sdst == 1) {
      for (int k = 0; k < current_len; k++)
        vst1q_lane_f32(outp + k, result, k);
    } else {
      float rbuf[4];
      vst1q_f32(rbuf, result);
      for (int k = 0; k < current_len; k++)
        outp[k * sdst] = rbuf[k];
    }

    infp += ssrc * current_len;
    outp += sdst * current_len;
    remaining -= current_len;
  }
  feclearexcept(FE_ALL_EXCEPT);
}


static void
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
    
    simd_exp_neon_FLOAT(tmp_src, 1, tmp_dst, 1, block_len);
    
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
        simd_exp_neon_FLOAT(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            simd_exp_neon_FLOAT((npy_float *)ip1, 1, (npy_float *)op1, 1, 1);
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
        simd_exp_neon_DOUBLE(src, ssrc, dst, sdst, len);
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

