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
/*
 * Vectorized implementation of log using AVX2 and AVX512
 * 1) if x < 0.0f; return -NAN (invalid input)
 * 2) Range reduction: y = x/2^k;
 *      a) y = normalized mantissa, k is the exponent (0.5 <= y < 1)
 * 3) Compute log(y) = P/Q, ratio of 2 polynomials P and Q
 *      b) P = 5th order and Q = 5th order polynomials obtained from Remez's
 *      algorithm (mini-max polynomial approximation)
 * 5) Compute log(x) = log(y) + k*ln(2)
 * 6) Max ULP error measured across all 32-bit FP's = 3.83 (x = 0x3f486945)
 * 7) Max relative error measured across all 32-bit FP's = 2.359E-07 (for same
 * x = 0x3f486945)
 */
static void
simd_log_FLOAT(npy_float * op,
                npy_float * ip,
                const npy_intp array_size,
                const npy_intp steps)
{
    const npy_intp stride = steps/(npy_intp)sizeof(npy_float);
    const npy_int num_lanes = 32/(npy_intp)sizeof(npy_float);

    npy_int32 indexarr[16];
    for (npy_int32 ii = 0; ii < 16; ii++) {
        indexarr[ii] = ii*stride;
    }

    __m256 log_p0 = _mm256_set1_ps(NPY_COEFF_P0_LOGf);
    __m256 log_p1 = _mm256_set1_ps(NPY_COEFF_P1_LOGf);
    __m256 log_p2 = _mm256_set1_ps(NPY_COEFF_P2_LOGf);
    __m256 log_p3 = _mm256_set1_ps(NPY_COEFF_P3_LOGf);
    __m256 log_p4 = _mm256_set1_ps(NPY_COEFF_P4_LOGf);
    __m256 log_p5 = _mm256_set1_ps(NPY_COEFF_P5_LOGf);
    __m256 log_q0 = _mm256_set1_ps(NPY_COEFF_Q0_LOGf);
    __m256 log_q1 = _mm256_set1_ps(NPY_COEFF_Q1_LOGf);
    __m256 log_q2 = _mm256_set1_ps(NPY_COEFF_Q2_LOGf);
    __m256 log_q3 = _mm256_set1_ps(NPY_COEFF_Q3_LOGf);
    __m256 log_q4 = _mm256_set1_ps(NPY_COEFF_Q4_LOGf);
    __m256 log_q5 = _mm256_set1_ps(NPY_COEFF_Q5_LOGf);
    __m256 loge2 = _mm256_set1_ps(NPY_LOGE2f);
    __m256 nan = _mm256_set1_ps(NPY_NANF);
    __m256 neg_nan = _mm256_set1_ps(-NPY_NANF);
    __m256 neg_inf = _mm256_set1_ps(-NPY_INFINITYF);
    __m256 inf = _mm256_set1_ps(NPY_INFINITYF);
    __m256 zeros_f = _mm256_set1_ps(0.0f);
    __m256 ones_f = _mm256_set1_ps(1.0f);
    __m256i vindex = _mm256_loadu_si256((__m256i*)indexarr);
    __m256 poly, num_poly, denom_poly, exponent;

    __m256 inf_mask, nan_mask, sqrt2_mask, zero_mask, negx_mask;
    __m256 invalid_mask = fma_get_partial_load_mask_ps(0, num_lanes);
    __m256 divide_by_zero_mask = invalid_mask;
    __m256 load_mask = fma_get_full_load_mask_ps();
    npy_intp num_remaining_elements = array_size;

    while (num_remaining_elements > 0) {

        if (num_remaining_elements < num_lanes) {
            load_mask = fma_get_partial_load_mask_ps(num_remaining_elements,
                                                       num_lanes);
        }

        __m256 x_in;
        if (stride == 1) {
            x_in = fma_masked_load_ps(load_mask, ip);
        }
        else {
            x_in  = fma_masked_gather_ps(zeros_f, ip, vindex, load_mask);
        }

        negx_mask = _mm256_cmp_ps(x_in, zeros_f, _CMP_LT_OQ);
        zero_mask = _mm256_cmp_ps(x_in, zeros_f, _CMP_EQ_OQ);
        inf_mask = _mm256_cmp_ps(x_in, inf, _CMP_EQ_OQ);
        nan_mask = _mm256_cmp_ps(x_in, x_in, _CMP_NEQ_UQ);
        divide_by_zero_mask = _mm256_or_ps(divide_by_zero_mask,
                                        _mm256_and_ps(zero_mask, load_mask));
        invalid_mask = _mm256_or_ps(invalid_mask, negx_mask);

        __m256 x = fma_set_masked_lanes_ps(x_in, zeros_f, negx_mask);

        exponent = fma_get_exponent(x);
        x = fma_get_mantissa(x);

        sqrt2_mask = _mm256_cmp_ps(x, _mm256_set1_ps(NPY_SQRT1_2f), _CMP_LE_OQ);
        x = fma_blend(x, _mm256_add_ps(x,x), sqrt2_mask);
        exponent = fma_blend(exponent,
                               _mm256_sub_ps(exponent,ones_f), sqrt2_mask);

        x = _mm256_sub_ps(x, ones_f);

        num_poly = _mm256_fmadd_ps(log_p5, x, log_p4);
        num_poly = _mm256_fmadd_ps(num_poly, x, log_p3);
        num_poly = _mm256_fmadd_ps(num_poly, x, log_p2);
        num_poly = _mm256_fmadd_ps(num_poly, x, log_p1);
        num_poly = _mm256_fmadd_ps(num_poly, x, log_p0);
        denom_poly = _mm256_fmadd_ps(log_q5, x, log_q4);
        denom_poly = _mm256_fmadd_ps(denom_poly, x, log_q3);
        denom_poly = _mm256_fmadd_ps(denom_poly, x, log_q2);
        denom_poly = _mm256_fmadd_ps(denom_poly, x, log_q1);
        denom_poly = _mm256_fmadd_ps(denom_poly, x, log_q0);
        poly = _mm256_div_ps(num_poly, denom_poly);
        poly = _mm256_fmadd_ps(exponent, loge2, poly);

        poly = fma_set_masked_lanes_ps(poly, nan, nan_mask);
        poly = fma_set_masked_lanes_ps(poly, neg_nan, negx_mask);
        poly = fma_set_masked_lanes_ps(poly, neg_inf, zero_mask);
        poly = fma_set_masked_lanes_ps(poly, inf, inf_mask);

        _mm256_maskstore_ps(op, _mm256_cvtps_epi32(load_mask), poly);

        ip += num_lanes*stride;
        op += num_lanes;
        num_remaining_elements -= num_lanes;
    }

    if (_mm256_movemask_ps(invalid_mask)) {
        npy_set_floatstatus_invalid();
    }
    if (_mm256_movemask_ps(divide_by_zero_mask)) {
        npy_set_floatstatus_divbyzero();
    }
}
#endif // SIMD_AVX2_FMA3

#ifdef SIMD_AVX512F
/*
 * Vectorized implementation of log using AVX2 and AVX512
 * 1) if x < 0.0f; return -NAN (invalid input)
 * 2) Range reduction: y = x/2^k;
 *      a) y = normalized mantissa, k is the exponent (0.5 <= y < 1)
 * 3) Compute log(y) = P/Q, ratio of 2 polynomials P and Q
 *      b) P = 5th order and Q = 5th order polynomials obtained from Remez's
 *      algorithm (mini-max polynomial approximation)
 * 5) Compute log(x) = log(y) + k*ln(2)
 * 6) Max ULP error measured across all 32-bit FP's = 3.83 (x = 0x3f486945)
 * 7) Max relative error measured across all 32-bit FP's = 2.359E-07 (for same
 * x = 0x3f486945)
 */
static void
simd_log_FLOAT(npy_float * op,
                npy_float * ip,
                const npy_intp array_size,
                const npy_intp steps)
{
    const npy_intp stride = steps/(npy_intp)sizeof(npy_float);
    const npy_int num_lanes = 64/(npy_intp)sizeof(npy_float);

    npy_int32 indexarr[16];
    for (npy_int32 ii = 0; ii < 16; ii++) {
        indexarr[ii] = ii*stride;
    }

    __m512 log_p0 = _mm512_set1_ps(NPY_COEFF_P0_LOGf);
    __m512 log_p1 = _mm512_set1_ps(NPY_COEFF_P1_LOGf);
    __m512 log_p2 = _mm512_set1_ps(NPY_COEFF_P2_LOGf);
    __m512 log_p3 = _mm512_set1_ps(NPY_COEFF_P3_LOGf);
    __m512 log_p4 = _mm512_set1_ps(NPY_COEFF_P4_LOGf);
    __m512 log_p5 = _mm512_set1_ps(NPY_COEFF_P5_LOGf);
    __m512 log_q0 = _mm512_set1_ps(NPY_COEFF_Q0_LOGf);
    __m512 log_q1 = _mm512_set1_ps(NPY_COEFF_Q1_LOGf);
    __m512 log_q2 = _mm512_set1_ps(NPY_COEFF_Q2_LOGf);
    __m512 log_q3 = _mm512_set1_ps(NPY_COEFF_Q3_LOGf);
    __m512 log_q4 = _mm512_set1_ps(NPY_COEFF_Q4_LOGf);
    __m512 log_q5 = _mm512_set1_ps(NPY_COEFF_Q5_LOGf);
    __m512 loge2 = _mm512_set1_ps(NPY_LOGE2f);
    __m512 nan = _mm512_set1_ps(NPY_NANF);
    __m512 neg_nan = _mm512_set1_ps(-NPY_NANF);
    __m512 neg_inf = _mm512_set1_ps(-NPY_INFINITYF);
    __m512 inf = _mm512_set1_ps(NPY_INFINITYF);
    __m512 zeros_f = _mm512_set1_ps(0.0f);
    __m512 ones_f = _mm512_set1_ps(1.0f);
    __m512i vindex = _mm512_loadu_si512((__m512i*)indexarr);
    __m512 poly, num_poly, denom_poly, exponent;

    __mmask16 inf_mask, nan_mask, sqrt2_mask, zero_mask, negx_mask;
    __mmask16 invalid_mask = avx512_get_partial_load_mask_ps(0, num_lanes);
    __mmask16 divide_by_zero_mask = invalid_mask;
    __mmask16 load_mask = avx512_get_full_load_mask_ps();
    npy_intp num_remaining_elements = array_size;

    while (num_remaining_elements > 0) {

        if (num_remaining_elements < num_lanes) {
            load_mask = avx512_get_partial_load_mask_ps(num_remaining_elements,
                                                       num_lanes);
        }

        __m512 x_in;
        if (stride == 1) {
            x_in = avx512_masked_load_ps(load_mask, ip);
        }
        else {
            x_in  = avx512_masked_gather_ps(zeros_f, ip, vindex, load_mask);
        }

        negx_mask = _mm512_cmp_ps_mask(x_in, zeros_f, _CMP_LT_OQ);
        zero_mask = _mm512_cmp_ps_mask(x_in, zeros_f, _CMP_EQ_OQ);
        inf_mask = _mm512_cmp_ps_mask(x_in, inf, _CMP_EQ_OQ);
        nan_mask = _mm512_cmp_ps_mask(x_in, x_in, _CMP_NEQ_UQ);
        divide_by_zero_mask = _mm512_kor(divide_by_zero_mask,
                                        _mm512_kand(zero_mask, load_mask));
        invalid_mask = _mm512_kor(invalid_mask, negx_mask);

        __m512 x = avx512_set_masked_lanes_ps(x_in, zeros_f, negx_mask);

        exponent = avx512_get_exponent(x);
        x = avx512_get_mantissa(x);

        sqrt2_mask = _mm512_cmp_ps_mask(x, _mm512_set1_ps(NPY_SQRT1_2f), _CMP_LE_OQ);
        x = avx512_blend(x, _mm512_add_ps(x,x), sqrt2_mask);
        exponent = avx512_blend(exponent,
                               _mm512_sub_ps(exponent,ones_f), sqrt2_mask);

        x = _mm512_sub_ps(x, ones_f);

        num_poly = _mm512_fmadd_ps(log_p5, x, log_p4);
        num_poly = _mm512_fmadd_ps(num_poly, x, log_p3);
        num_poly = _mm512_fmadd_ps(num_poly, x, log_p2);
        num_poly = _mm512_fmadd_ps(num_poly, x, log_p1);
        num_poly = _mm512_fmadd_ps(num_poly, x, log_p0);
        denom_poly = _mm512_fmadd_ps(log_q5, x, log_q4);
        denom_poly = _mm512_fmadd_ps(denom_poly, x, log_q3);
        denom_poly = _mm512_fmadd_ps(denom_poly, x, log_q2);
        denom_poly = _mm512_fmadd_ps(denom_poly, x, log_q1);
        denom_poly = _mm512_fmadd_ps(denom_poly, x, log_q0);
        poly = _mm512_div_ps(num_poly, denom_poly);
        poly = _mm512_fmadd_ps(exponent, loge2, poly);

        poly = avx512_set_masked_lanes_ps(poly, nan, nan_mask);
        poly = avx512_set_masked_lanes_ps(poly, neg_nan, negx_mask);
        poly = avx512_set_masked_lanes_ps(poly, neg_inf, zero_mask);
        poly = avx512_set_masked_lanes_ps(poly, inf, inf_mask);

        _mm512_mask_storeu_ps(op, load_mask, poly);

        ip += num_lanes*stride;
        op += num_lanes;
        num_remaining_elements -= num_lanes;
    }

    if (npyv_tobits_b32(invalid_mask)) {
        npy_set_floatstatus_invalid();
    }
    if (npyv_tobits_b32(divide_by_zero_mask)) {
        npy_set_floatstatus_divbyzero();
    }
}
#endif // SIMD_AVX512F


#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
static void
simd_log_f64(const npyv_lanetype_f64 *src, npy_intp ssrc,
                      npyv_lanetype_f64 *dst, npy_intp sdst, npy_intp len)
{
    const int vstep = npyv_nlanes_f64;
    for (; len > 0; len -= vstep, src += ssrc*vstep, dst += sdst*vstep) {
        npyv_f64 x;
        if (ssrc == 1) {
            x = npyv_load_till_f64(src, len, 1);
        } else {
            x = npyv_loadn_till_f64(src, ssrc, len, 1);
        }
        npyv_f64 out = __svml_log8_ha(x);
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
 * Vectorized implementation of log double using AVX512
 * Reference:
 * [1] Tang, Ping Tak Peter. Table-lookup algorithms for elementary functions
 *     and their error analysis. No. CONF-9106103-1. Argonne National Lab.,
 *     IL (USA), 1991.
 * [2] Tang, Ping-Tak Peter. "Table-driven implementation of the logarithm
 *     function in IEEE floating-point arithmetic." ACM Transactions on
 *     Mathematical Software (TOMS) 16.4 (1990): 378-400.
 * [3] Muller, Jean-Michel. "Elementary functions: algorithms and
 *     implementation." (2016).
 * 1) if x = 0; return -INF
 * 2) if x < 0; return NAN
 * 3) if x is INF; return INF
 * 4) if x is NAN; return NAN
 * 5) if x on (1.0 - 0x1p-4, 1.0 + 0x1.09p-4), calling npy_log()
 * 6) Range reduction:
 *    log(x) = log(2^m * z)
 *           = mln2 + log(z)
 * 7) log(z) = log(z / c_k) + log(c_k);
 *    where c_k = 1 + k/64, k = 0,1,...,64
 *    s.t. |x - c_k| <= 1/128 when x on[1,2].
 * 8) r = 2(x - c_k)/(x + c_k)
 *    log(x/c_k) = log((1 + r/2) / (1 - r/2))
 *               = p(r)
 *               = 2((r/2) + 1/3*(r/2)^3 + 1/5*(r/2)^5 + ...)
 */

static void
AVX512F_log_DOUBLE(npy_double * op,
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

    __m512d zeros_d = _mm512_set1_pd(0.0);
    __m512d ones_d = _mm512_set1_pd(1.0);
    __m512d mInf = _mm512_set1_pd(NPY_INFINITY);
    __m512d mInv64 = _mm512_castsi512_pd(_mm512_set1_epi64(0x3f90000000000000));
    __m512d mNeg_nan = _mm512_set1_pd(-NPY_NAN);
    __m512d mNan = _mm512_set1_pd(NPY_NAN);
    __m512d mNeg_inf = _mm512_set1_pd(-NPY_INFINITY);
    __m512d mA1 = _mm512_set1_pd(NPY_TANG_LOG_A1);
    __m512d mA2 = _mm512_set1_pd(NPY_TANG_LOG_A2);
    __m512d mA3 = _mm512_set1_pd(NPY_TANG_LOG_A3);
    __m512d mA4 = _mm512_set1_pd(NPY_TANG_LOG_A4);
    __m512d mLN2hi = _mm512_set1_pd(NPY_TANG_LOG_LN2HI);
    __m512d mLN2lo = _mm512_set1_pd(NPY_TANG_LOG_LN2LO);

    __m512d mTo_glibc_min = _mm512_set1_pd(1.0 - 0x1p-4);
    __m512d mTo_glibc_max = _mm512_set1_pd(1.0 + 0x1.09p-4);
    __m256i vindex = _mm256_loadu_si256((__m256i*)&indexarr[0]);

    __m512d mLUT_TOP_0 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*0]));
    __m512d mLUT_TOP_1 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*1]));
    __m512d mLUT_TOP_2 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*2]));
    __m512d mLUT_TOP_3 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*3]));
    __m512d mLUT_TOP_4 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*4]));
    __m512d mLUT_TOP_5 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*5]));
    __m512d mLUT_TOP_6 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*6]));
    __m512d mLUT_TOP_7 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*7]));
    __m512d mLUT_TAIL_0 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*0]));
    __m512d mLUT_TAIL_1 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*1]));
    __m512d mLUT_TAIL_2 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*2]));
    __m512d mLUT_TAIL_3 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*3]));
    __m512d mLUT_TAIL_4 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*4]));
    __m512d mLUT_TAIL_5 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*5]));
    __m512d mLUT_TAIL_6 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*6]));
    __m512d mLUT_TAIL_7 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*7]));

    __mmask8 load_mask = avx512_get_full_load_mask_pd();
    __mmask8 invalid_mask = avx512_get_partial_load_mask_pd(0, num_lanes);
    __mmask8 divide_by_zero_mask = invalid_mask;

    __mmask8 inf_mask, nan_mask, zero_mask, negx_mask, denormal_mask,
             glibc_mask;

    __m512d x_in;
    while (num_remaining_elements > 0) {
        if (num_remaining_elements < num_lanes) {
            load_mask = avx512_get_partial_load_mask_pd(num_remaining_elements,
                                                      num_lanes);
        }

        if (1 == stride) {
            x_in = avx512_masked_load_pd(load_mask, ip);
        }
        else {
            x_in = avx512_masked_gather_pd(zeros_d, ip, vindex, load_mask);
        }

        __mmask8 m1 = _mm512_cmp_pd_mask(x_in, mTo_glibc_max, _CMP_LT_OQ);
        __mmask8 m2 = _mm512_cmp_pd_mask(x_in, mTo_glibc_min, _CMP_GT_OQ);
        glibc_mask =  m1 & m2;

        if (glibc_mask != 0xFF) {
            zero_mask = _mm512_cmp_pd_mask(x_in, zeros_d, _CMP_EQ_OQ);
            inf_mask = _mm512_cmp_pd_mask(x_in, mInf, _CMP_EQ_OQ);
            negx_mask = _mm512_cmp_pd_mask(x_in, zeros_d, _CMP_LT_OQ);
            nan_mask = _mm512_cmp_pd_mask(x_in, x_in, _CMP_NEQ_UQ);

            divide_by_zero_mask = divide_by_zero_mask | (zero_mask & load_mask);
            invalid_mask = invalid_mask | negx_mask;

            __m512d x = avx512_set_masked_lanes_pd(x_in, zeros_d, negx_mask);
            __m512i ix = _mm512_castpd_si512(x);

            __m512i top12 = _mm512_and_epi64(ix,
                                _mm512_set1_epi64(0xfff0000000000000));
            denormal_mask = _mm512_cmp_epi64_mask(top12, _mm512_set1_epi64(0),
                                _CMP_EQ_OQ);
            denormal_mask = (~zero_mask) & denormal_mask;
            __m512d masked_x = x;
            #ifdef WORKAROUND_LLVM__mm512_mask_mul_pd
            masked_x = avx512_set_masked_lanes_pd(masked_x, zeros_d, (~denormal_mask));
            #endif
            ix = _mm512_castpd_si512(_mm512_mask_mul_pd(x, denormal_mask,
                                    masked_x, _mm512_set1_pd(0x1p52)));
            ix = _mm512_mask_sub_epi64(ix, denormal_mask,
                                    ix, _mm512_set1_epi64(52ULL << 52));

            __m512i tmp = _mm512_sub_epi64(ix,
                              _mm512_set1_epi64(0x3ff0000000000000));
            __m512i i = _mm512_and_epi64(_mm512_srai_epi64(tmp, 52 - 6),
                            _mm512_set1_epi64(0x3fULL));
            __m512i ik = _mm512_srai_epi64(tmp, 52);
            __m512d z = _mm512_castsi512_pd(_mm512_sub_epi64(ix, _mm512_and_epi64(tmp,
                            _mm512_set1_epi64(0xfff0000000000000))));
            __m256i i_32 = _mm512_cvtepi64_epi32(i);
            __m512d c = _mm512_fmadd_pd(_mm512_cvtepi32_pd(i_32), mInv64, ones_d);

            __m512d u = _mm512_div_pd(_mm512_sub_pd(z, c), _mm512_add_pd(z, c));
            u = _mm512_mul_pd(_mm512_set1_pd(2.0), u);

            __m512d v = _mm512_mul_pd(u,u);

            __m512d res = _mm512_fmadd_pd(v, mA4, mA3);
            res = _mm512_fmadd_pd(v, res, mA2);
            res = _mm512_fmadd_pd(v, res, mA1);
            res = _mm512_mul_pd(v, res);
            res = _mm512_fmadd_pd(u, res, u);

            __m512d c_hi = avx512_permute_x8var_pd(mLUT_TOP_0, mLUT_TOP_1,
                            mLUT_TOP_2, mLUT_TOP_3, mLUT_TOP_4, mLUT_TOP_5,
                            mLUT_TOP_6, mLUT_TOP_7, i);
            __m512d c_lo = avx512_permute_x8var_pd(mLUT_TAIL_0, mLUT_TAIL_1,
                              mLUT_TAIL_2, mLUT_TAIL_3, mLUT_TAIL_4, mLUT_TAIL_5,
                              mLUT_TAIL_6, mLUT_TAIL_7, i);

            __m256i ik_32 = _mm512_cvtepi64_epi32(ik);
            __m512d k = _mm512_cvtepi32_pd(ik_32);
            __m512d tt = _mm512_fmadd_pd(k, mLN2hi, c_hi);
            __m512d tt2 = _mm512_fmadd_pd(k, mLN2lo, c_lo);
            tt = _mm512_add_pd(tt, tt2);
            res = _mm512_add_pd(tt, res);

            res = avx512_set_masked_lanes_pd(res, mNan, nan_mask);
            res = avx512_set_masked_lanes_pd(res, mNeg_nan, negx_mask);
            res = avx512_set_masked_lanes_pd(res, mNeg_inf, zero_mask);
            res = avx512_set_masked_lanes_pd(res, mInf, inf_mask);

            _mm512_mask_storeu_pd(op, load_mask, res);
        }

        if (glibc_mask != 0) {
            double NPY_DECL_ALIGNED(64) ip_fback[8];
            _mm512_mask_store_pd(ip_fback, avx512_get_full_load_mask_pd(), x_in);

            for (int ii = 0; ii < 8; ++ii, glibc_mask >>= 1) {
                if (glibc_mask & 0x01) {
                    op[ii] = npy_log(ip_fback[ii]);
                }
            }
        }
        ip += num_lanes * stride;
        op += num_lanes;
        num_remaining_elements -= num_lanes;
    }

    if (npyv_tobits_b64(invalid_mask)) {
        npy_set_floatstatus_invalid();
    }
    if (npyv_tobits_b64(divide_by_zero_mask)) {
        npy_set_floatstatus_divbyzero();
    }
}

#undef WORKAROUND_LLVM__mm512_mask_mul_pd

#endif // SIMD_AVX512F_NOCLANG_BUG
#endif // NPY_CAN_LINK_SVML




/********************************************************************************
 ** Defining ufunc inner functions for exp/log
 ********************************************************************************/
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_log)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if defined(SIMD_AVX2_FMA3) || defined(SIMD_AVX512F)
    if (IS_OUTPUT_BLOCKABLE_UNARY(sizeof(npy_float), sizeof(npy_float), 64)) {
        simd_log_FLOAT((npy_float*)args[1], (npy_float*)args[0], dimensions[0], steps[0]);
    }
    else {
        UNARY_LOOP {
            simd_log_FLOAT((npy_float *)op1, (npy_float *)ip1, 1, steps[0]);
        }
    }
#else
    UNARY_LOOP {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = npy_logf(in1);
    }
#endif
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_log)
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

        simd_log_f64(src, ssrc, dst, sdst, len);
        return;
    }
#else
  #ifdef SIMD_AVX512F_NOCLANG_BUG
      if (IS_OUTPUT_BLOCKABLE_UNARY(sizeof(npy_double), sizeof(npy_double), 64)) {
          AVX512F_log_DOUBLE((npy_double*)args[1], (npy_double*)args[0], dimensions[0], steps[0]);
          return;
      }
  #endif
#endif
    UNARY_LOOP {
        const npy_double in1 = *(npy_double *)ip1;
        *(npy_double *)op1 = npy_log(in1);
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

static void avx512_log_f16(const npy_half *src, npy_half *dst, npy_intp len)
{
    avx512_exponent_f16(src, dst, len, __svml_logf16, 0x3c00);
}
#endif


NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_log)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
#if NPY__SVML_IS_ENABLED
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];

    const npy_intp len = dimensions[0];

    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        (steps[0] == sizeof(npy_half)) &&
        (steps[1] == sizeof(npy_half))) {
    #ifdef NPY_HAVE_AVX512_SPR
        __svml_logs32(src, dst, len);
    #else
        avx512_log_f16(src, dst, len);
    #endif
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_logf(in1));
    }
}
