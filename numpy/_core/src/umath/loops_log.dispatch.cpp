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


#if SIMD_ARM

static inline float64x2_t
neon_log_inline_core(uint64x2_t u, uint64x2_t u_off)
{
  const float64x2_t c0 = vdupq_n_f64(-0x1.ffffffffffff7p-2);
  const float64x2_t c1 = vdupq_n_f64(0x1.55555555170d4p-2);
  const float64x2_t c2 = vdupq_n_f64(-0x1.0000000399c27p-2);
  const float64x2_t c3 = vdupq_n_f64(0x1.999b2e90e94cap-3);
  const float64x2_t c4 = vdupq_n_f64(-0x1.554e550bd501ep-3);
  const float64x2_t ln2 = vdupq_n_f64(0x1.62e42fefa39efp-1);
  const uint64x2_t sign_exp_mask = vdupq_n_u64(0xfff0000000000000ULL);

  int64x2_t k = vshrq_n_s64(vreinterpretq_s64_u64(u_off), 52);
  uint64x2_t iz = vsubq_u64(u, vandq_u64(u_off, sign_exp_mask));
  float64x2_t z = vreinterpretq_f64_u64(iz);
  uint64_t i0 = (vgetq_lane_u64(u_off, 0) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;
  uint64_t i1 = (vgetq_lane_u64(u_off, 1) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;

  float64x2_t invc0 = vld1q_dup_f64(&neon_log_table[i0].invc);
  float64x2_t invc1 = vld1q_dup_f64(&neon_log_table[i1].invc);
  float64x2_t invc = vcombine_f64(vget_low_f64(invc0), vget_low_f64(invc1));

  float64x2_t logc0 = vld1q_dup_f64(&neon_log_table[i0].logc);
  float64x2_t logc1 = vld1q_dup_f64(&neon_log_table[i1].logc);
  float64x2_t logc = vcombine_f64(vget_low_f64(logc0), vget_low_f64(logc1));

  float64x2_t r = vfmaq_f64(vdupq_n_f64(-1.0), z, invc);
  float64x2_t kd = vcvtq_f64_s64(k);

  float64x2_t hi = vfmaq_f64(vaddq_f64(logc, r), kd, ln2);
  float64x2_t r2 = vmulq_f64(r, r);
  float64x2_t y = vfmaq_f64(c2, r, c3);
  float64x2_t p = vfmaq_f64(c0, r, c1);
  y = vfmaq_f64(y, r2, c4);
  y = vfmaq_f64(p, r2, y);

  return vfmaq_f64(hi, y, r2);
}

static void
simd_log_neon_DOUBLE(const npy_double *src, npy_intp ssrc,
                       npy_double *dst, npy_intp sdst, npy_intp len)
{
  const float64x2_t c0 = vdupq_n_f64(-0x1.ffffffffffff7p-2);
  const float64x2_t c1 = vdupq_n_f64(0x1.55555555170d4p-2);
  const float64x2_t c2 = vdupq_n_f64(-0x1.0000000399c27p-2);
  const float64x2_t c3 = vdupq_n_f64(0x1.999b2e90e94cap-3);
  const float64x2_t c4 = vdupq_n_f64(-0x1.554e550bd501ep-3);
  const float64x2_t ln2 = vdupq_n_f64(0x1.62e42fefa39efp-1);
  const uint64x2_t sign_exp_mask = vdupq_n_u64(0xfff0000000000000ULL);
  const uint64x2_t off = vdupq_n_u64(0x3fe6900900000000ULL);
  const float64x2_t zeros_f = vdupq_n_f64(0.0);
  const float64x2_t ones_f = vdupq_n_f64(1.0);
  const float64x2_t neg_one = vdupq_n_f64(-1.0);
  const float64x2_t pos_inf = vdupq_n_f64(NPY_INFINITY);
  const float64x2_t neg_inf_val = vdupq_n_f64(-NPY_INFINITY);
  const float64x2_t nan_val = vdupq_n_f64(NPY_NAN);
  const float64x2_t neg_nan = vdupq_n_f64(-NPY_NAN);
  const uint64x2_t all_ones = vdupq_n_u64(0xFFFFFFFFFFFFFFFFULL);

  const npy_double *infp = src;
  npy_double *outp = dst;
  npy_intp remaining = len;

  feclearexcept(FE_ALL_EXCEPT);

  const npy_intp UNROLL = 4;
  const npy_intp VEC_SIZE = 2;

  while (remaining >= UNROLL * VEC_SIZE) {
    float64x2_t x[UNROLL], u_f[UNROLL];
    uint64x2_t u[UNROLL];

    if (ssrc == 1) {
      for (int i = 0; i < UNROLL; i++)
        x[i] = vld1q_f64(infp + i * VEC_SIZE);
    } else {
      for (int i = 0; i < UNROLL; i++) {
        double vals[2] = {infp[(2*i) * ssrc], infp[(2*i+1) * ssrc]};
        x[i] = vld1q_f64(vals);
      }
    }

    uint64x2_t is_nan[UNROLL], is_pos_inf[UNROLL], is_neg_inf[UNROLL], is_zero[UNROLL], is_neg[UNROLL], full_mask[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      is_nan[i] = veorq_u64(vceqq_f64(x[i], x[i]), all_ones);
      is_pos_inf[i] = vceqq_f64(x[i], pos_inf);
      is_neg_inf[i] = vceqq_f64(x[i], neg_inf_val);
      is_zero[i] = vceqq_f64(x[i], zeros_f);
      is_neg[i] = vbicq_u64(vcltq_f64(x[i], zeros_f), vorrq_u64(is_neg_inf[i], is_zero[i]));
      uint64x2_t special_mask = vorrq_u64(vorrq_u64(is_nan[i], is_pos_inf[i]), vorrq_u64(is_neg_inf[i], is_zero[i]));
      full_mask[i] = vorrq_u64(special_mask, is_neg[i]);
      u_f[i] = vbslq_f64(full_mask[i], ones_f, x[i]);
      u[i] = vreinterpretq_u64_f64(u_f[i]);
    }

    float64x2_t r[UNROLL], kd[UNROLL], hi[UNROLL], r2[UNROLL], y[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      uint64x2_t u_off = vsubq_u64(u[i], off);
      int64x2_t k = vshrq_n_s64(vreinterpretq_s64_u64(u_off), 52);
      uint64x2_t iz = vsubq_u64(u[i], vandq_u64(u_off, sign_exp_mask));
      float64x2_t z = vreinterpretq_f64_u64(iz);

      uint64_t idx[2];
      for (int j = 0; j < 2; j++)
        idx[j] = (vgetq_lane_u64(u_off, j) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;

      float64x2_t invc = vcombine_f64(vdup_n_f64(neon_log_table[idx[0]].invc), vdup_n_f64(neon_log_table[idx[1]].invc));
      float64x2_t logc = vcombine_f64(vdup_n_f64(neon_log_table[idx[0]].logc), vdup_n_f64(neon_log_table[idx[1]].logc));

      r[i] = vfmaq_f64(neg_one, z, invc);
      kd[i] = vcvtq_f64_s64(k);
      hi[i] = vfmaq_f64(vaddq_f64(logc, r[i]), kd[i], ln2);
      r2[i] = vmulq_f64(r[i], r[i]);
      float64x2_t p = vfmaq_f64(c0, r[i], c1);
      y[i] = vfmaq_f64(c2, r[i], c3);
      y[i] = vfmaq_f64(y[i], r2[i], c4);
      y[i] = vfmaq_f64(p, r2[i], y[i]);
    }

    float64x2_t result[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      result[i] = vfmaq_f64(hi[i], y[i], r2[i]);
      result[i] = vbslq_f64(is_nan[i], x[i], result[i]);
      result[i] = vbslq_f64(is_pos_inf[i], pos_inf, result[i]);
      result[i] = vbslq_f64(is_neg_inf[i], nan_val, result[i]);
      result[i] = vbslq_f64(is_zero[i], neg_inf_val, result[i]);
      result[i] = vbslq_f64(is_neg[i], neg_nan, result[i]);
    }

    if (sdst == 1) {
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

    uint64x2_t u = vreinterpretq_u64_f64(x);
    uint64x2_t is_nan = vceqq_f64(x, x);
    is_nan = veorq_u64(is_nan, all_ones);
    uint64x2_t is_pos_inf = vceqq_f64(x, pos_inf);
    uint64x2_t is_neg_inf = vceqq_f64(x, neg_inf_val);
    uint64x2_t is_zero = vceqq_f64(x, zeros_f);
    uint64x2_t is_neg = vcltq_f64(x, zeros_f);
    is_neg = vbicq_u64(is_neg, vorrq_u64(is_neg_inf, is_zero));

    uint64x2_t need_scalar = vorrq_u64(is_neg, vorrq_u64(is_zero, vorrq_u64(is_nan, is_neg_inf)));

    if (__builtin_expect(!neon_has_any_lane_u64(need_scalar), 1)) {
      uint64x2_t u_off = vsubq_u64(u, off);
      float64x2_t result = neon_log_inline_core(u, u_off);

      result = vbslq_f64(is_pos_inf, pos_inf, result);

      if (sdst == 1) {
        if (current_len == 1) vst1q_lane_f64(outp, result, 0);
        else vst1q_f64(outp, result);
      } else {
        outp[0] = vgetq_lane_f64(result, 0);
        if (current_len > 1) outp[sdst] = vgetq_lane_f64(result, 1);
      }
    } else {
      feclearexcept(FE_ALL_EXCEPT);
      npy_intp ds = (sdst == 1) ? 1 : sdst;
      uint64_t m_nan[2], m_pinf[2], m_ninf[2], m_neg[2], m_zero[2];
      vst1q_u64(m_nan, is_nan);
      vst1q_u64(m_pinf, is_pos_inf);
      vst1q_u64(m_ninf, is_neg_inf);
      vst1q_u64(m_neg, is_neg);
      vst1q_u64(m_zero, is_zero);
      for (int k = 0; k < current_len; k++) {
        double val = (ssrc == 1) ? infp[k] : infp[k * ssrc];
        double r;
        if (m_nan[k]) r = val;
        else if (m_pinf[k]) r = NPY_INFINITY;
        else if (m_ninf[k]) r = NPY_NAN;
        else if (m_zero[k]) r = -NPY_INFINITY;
        else if (m_neg[k]) r = -NPY_NAN;
        else r = npy_log(val);
        outp[k * ds] = r;
      }
    }

    infp += ssrc * current_len;
    outp += sdst * current_len;
    remaining -= current_len;
  }
  feclearexcept(FE_ALL_EXCEPT);
}

static inline float32x4_t
neon_logf_inline_core(uint32x4_t u_off, float32x4_t n)
{
  const float32x4_t c0 = vdupq_n_f32(-0x1.3e737cp-3f);
  const float32x4_t c1 = vdupq_n_f32(0x1.5a9aa2p-3f);
  const float32x4_t c2 = vdupq_n_f32(-0x1.4f9934p-3f);
  const float32x4_t c3 = vdupq_n_f32(0x1.961348p-3f);
  const float32x4_t c4 = vdupq_n_f32(-0x1.00187cp-2f);
  const float32x4_t c5 = vdupq_n_f32(0x1.555d7cp-2f);
  const float32x4_t c6 = vdupq_n_f32(-0x1.ffffc8p-2f);
  const float32x4_t ln2 = vdupq_n_f32(0x1.62e43p-1f);
  const uint32x4_t off = vdupq_n_u32(0x3f2aaaab);
  const uint32x4_t mantissa_mask = vdupq_n_u32(0x007fffff);

  uint32x4_t u = vaddq_u32(vandq_u32(u_off, mantissa_mask), off);
  float32x4_t r = vsubq_f32(vreinterpretq_f32_u32(u), vdupq_n_f32(1.0f));

  float32x4_t r2 = vmulq_f32(r, r);

  float32x4_t p = vfmaq_f32(c2, r, c1);
  float32x4_t q = vfmaq_f32(c4, r, c3);
  float32x4_t y = vfmaq_f32(c6, r, c5);
  p = vfmaq_f32(p, r2, c0);

  q = vfmaq_f32(q, p, r2);
  y = vfmaq_f32(y, q, r2);
  float32x4_t base = vfmaq_f32(r, n, ln2);

  return vfmaq_f32(base, y, r2);
}

static void
simd_log_neon_FLOAT(const npy_float *src, npy_intp ssrc,
                      npy_float *dst, npy_intp sdst, npy_intp len)
{
  const uint32x4_t off = vdupq_n_u32(0x3f2aaaab);
  const float32x4_t zeros_f = vdupq_n_f32(0.0f);
  const float32x4_t ones_f = vdupq_n_f32(1.0f);
  const float32x4_t pos_inf = vdupq_n_f32(NPY_INFINITYF);
  const float32x4_t neg_inf_val = vdupq_n_f32(-NPY_INFINITYF);
  const float32x4_t nan_val = vdupq_n_f32(NPY_NANF);
  const float32x4_t neg_nan = vdupq_n_f32(-NPY_NANF);

  const npy_float *infp = src;
  npy_float *outp = dst;
  npy_intp remaining = len;

  feclearexcept(FE_ALL_EXCEPT);

  const npy_intp UNROLL = 4;
  const npy_intp VEC_SIZE = 4;

  while (remaining >= UNROLL * VEC_SIZE) {
    float32x4_t x[UNROLL];
    uint32x4_t u[UNROLL];

    if (ssrc == 1) {
      for (int i = 0; i < UNROLL; i++)
        x[i] = vld1q_f32(infp + i * VEC_SIZE);
    } else {
      for (int i = 0; i < UNROLL; i++) {
        float vals[VEC_SIZE];
        for (int j = 0; j < VEC_SIZE; j++)
          vals[j] = infp[(i * VEC_SIZE + j) * ssrc];
        x[i] = vld1q_f32(vals);
      }
    }

    uint32x4_t is_nan[UNROLL], is_pos_inf[UNROLL], is_neg_inf[UNROLL], is_zero[UNROLL], is_neg[UNROLL], full_mask[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      is_nan[i] = vmvnq_u32(vceqq_f32(x[i], x[i]));
      is_pos_inf[i] = vceqq_f32(x[i], pos_inf);
      is_neg_inf[i] = vceqq_f32(x[i], neg_inf_val);
      is_zero[i] = vceqq_f32(x[i], zeros_f);
      is_neg[i] = vbicq_u32(vcltq_f32(x[i], zeros_f), vorrq_u32(is_neg_inf[i], is_zero[i]));
      uint32x4_t special_mask = vorrq_u32(vorrq_u32(is_nan[i], is_pos_inf[i]), vorrq_u32(is_neg_inf[i], is_zero[i]));
      full_mask[i] = vorrq_u32(special_mask, is_neg[i]);
      u[i] = vreinterpretq_u32_f32(vbslq_f32(full_mask[i], ones_f, x[i]));
    }

    float32x4_t result[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      uint32x4_t u_off = vsubq_u32(u[i], off);
      float32x4_t n = vcvtq_f32_s32(vshrq_n_s32(vreinterpretq_s32_u32(u_off), 23));
      result[i] = neon_logf_inline_core(u_off, n);
    }

    for (int i = 0; i < UNROLL; i++) {
      result[i] = vbslq_f32(is_nan[i], x[i], result[i]);
      result[i] = vbslq_f32(is_pos_inf[i], pos_inf, result[i]);
      result[i] = vbslq_f32(is_neg_inf[i], nan_val, result[i]);
      result[i] = vbslq_f32(is_zero[i], neg_inf_val, result[i]);
      result[i] = vbslq_f32(is_neg[i], neg_nan, result[i]);
    }

    if (sdst == 1) {
      for (int i = 0; i < UNROLL; i++)
        vst1q_f32(outp + i * VEC_SIZE, result[i]);
    } else {
      for (int i = 0; i < UNROLL; i++) {
        for (int j = 0; j < VEC_SIZE; j++)
          outp[(i * VEC_SIZE + j) * sdst] = vgetq_lane_f32(result[i], j);
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
      float vals[4] = {1.0f, 1.0f, 1.0f, 1.0f};
      for (int i = 0; i < current_len; i++) vals[i] = infp[i * ssrc];
      x = vld1q_f32(vals);
    }

    uint32x4_t u = vreinterpretq_u32_f32(x);
    uint32x4_t is_nan = vceqq_f32(x, x);
    is_nan = vmvnq_u32(is_nan);
    uint32x4_t is_pos_inf = vceqq_f32(x, pos_inf);
    uint32x4_t is_neg_inf = vceqq_f32(x, neg_inf_val);
    uint32x4_t is_zero = vceqq_f32(x, zeros_f);
    uint32x4_t is_neg = vcltq_f32(x, zeros_f);
    is_neg = vbicq_u32(is_neg, vorrq_u32(is_neg_inf, is_zero));

    uint32x4_t need_scalar = vorrq_u32(is_neg, vorrq_u32(is_zero, vorrq_u32(is_nan, is_neg_inf)));

    if (__builtin_expect(!neon_has_any_lane_u32(need_scalar), 1)) {
      uint32x4_t u_off = vsubq_u32(u, off);
      float32x4_t n = vcvtq_f32_s32(vshrq_n_s32(vreinterpretq_s32_u32(u_off), 23));
      float32x4_t result = neon_logf_inline_core(u_off, n);

      result = vbslq_f32(is_pos_inf, pos_inf, result);

      if (sdst == 1) {
        for (int k = 0; k < current_len; k++)
          vst1q_lane_f32(outp + k, result, k);
      } else {
        float rbuf[4];
        vst1q_f32(rbuf, result);
        for (int k = 0; k < current_len; k++)
          outp[k * sdst] = rbuf[k];
      }
    } else {
      feclearexcept(FE_ALL_EXCEPT);
      npy_intp ds = (sdst == 1) ? 1 : sdst;
      uint32_t m_nan[4], m_pinf[4], m_ninf[4], m_neg[4], m_zero[4];
      vst1q_u32(m_nan, is_nan);
      vst1q_u32(m_pinf, is_pos_inf);
      vst1q_u32(m_ninf, is_neg_inf);
      vst1q_u32(m_neg, is_neg);
      vst1q_u32(m_zero, is_zero);
      for (int k = 0; k < current_len; k++) {
        float val = (ssrc == 1) ? infp[k] : infp[k * ssrc];
        float r;
        if (m_nan[k]) r = val;
        else if (m_pinf[k]) r = NPY_INFINITYF;
        else if (m_ninf[k]) r = NPY_NANF;
        else if (m_zero[k]) r = -NPY_INFINITYF;
        else if (m_neg[k]) r = -NPY_NANF;
        else r = npy_logf(val);
        outp[k * ds] = r;
      }
    }

    infp += ssrc * current_len;
    outp += sdst * current_len;
    remaining -= current_len;
  }
  feclearexcept(FE_ALL_EXCEPT);
}


static void
simd_log_neon_HALF(const npy_half *src, npy_intp ssrc,
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
      dst[i * sdst] = npy_float_to_half(npy_logf(in1));
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

    simd_log_neon_FLOAT(tmp_src, 1, tmp_dst, 1, block_len);

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
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_log)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_float in1 = *(npy_float *)ip1;
            *(npy_float *)op1 = npy_logf(in1);
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_float *src = (npy_float*)args[0];
        npy_float *dst = (npy_float*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);
        simd_log_neon_FLOAT(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            simd_log_neon_FLOAT((npy_float *)ip1, 1, (npy_float *)op1, 1, 1);
        }
    }
    return;
#elif defined(SIMD_AVX2_FMA3) || defined(SIMD_AVX512F)
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
#elif SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_double in1 = *(npy_double *)ip1;
            *(npy_double *)op1 = npy_log(in1);
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_double *src = (npy_double*)args[0];
        npy_double *dst = (npy_double*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_double);
        const npy_intp sdst = steps[1] / sizeof(npy_double);
        simd_log_neon_DOUBLE(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            const npy_double in1 = *(npy_double *)ip1;
            *(npy_double *)op1 = npy_log(in1);
        }
    }
    return;
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
#if SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_logf(in1));
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_half *src = (npy_half*)args[0];
        npy_half *dst = (npy_half*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_half);
        const npy_intp sdst = steps[1] / sizeof(npy_half);
        simd_log_neon_HALF(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_logf(in1));
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
