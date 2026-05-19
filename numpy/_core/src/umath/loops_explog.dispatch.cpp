#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <float.h>
#include <fenv.h>

#include "numpy/npy_math.h"
#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
#include "lowlevel_strided_loops.h"
#include "fast_loop_macros.h"
#include "npy_simd_data.h"
#include "npy_svml.h"

#if defined(__aarch64__) && NPY_SIMD_FMA3
#define SIMD_ARM 1
#endif

#if !defined(_MSC_VER) && defined(NPY_HAVE_AVX512F)
    #define SIMD_AVX512F
#elif defined(NPY_HAVE_AVX2) && defined(NPY_HAVE_FMA3)
    #define SIMD_AVX2_FMA3
#endif
#if defined(SIMD_AVX512F) && !(defined(__clang__) && (__clang_major__ < 10 || \
                              (__clang_major__ == 10 && __clang_minor__ < 1)))
    #define SIMD_AVX512F_NOCLANG_BUG
#endif

#ifdef SIMD_AVX2_FMA3

NPY_FINLINE __m256
fma_get_full_load_mask_ps(void)
{
    return _mm256_set1_ps(-1.0);
}

NPY_FINLINE __m256i
fma_get_full_load_mask_pd(void)
{
    return _mm256_castpd_si256(_mm256_set1_pd(-1.0));
}

NPY_FINLINE __m256
fma_get_partial_load_mask_ps(const npy_int num_elem, const npy_int num_lanes)
{
    float maskint[16] = {-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
                            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    float* addr = maskint + num_lanes - num_elem;
    return _mm256_loadu_ps(addr);
}

NPY_FINLINE __m256i
fma_get_partial_load_mask_pd(const npy_int num_elem, const npy_int num_lanes)
{
    npy_int maskint[16] = {-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1};
    npy_int* addr = maskint + 2*num_lanes - 2*num_elem;
    return _mm256_loadu_si256((__m256i*) addr);
}

NPY_FINLINE __m256
fma_masked_gather_ps(__m256 src,
                     npy_float* addr,
                     __m256i vindex,
                     __m256 mask)
{
    return _mm256_mask_i32gather_ps(src, addr, vindex, mask, 4);
}

NPY_FINLINE __m256d
fma_masked_gather_pd(__m256d src,
                     npy_double* addr,
                     __m128i vindex,
                     __m256d mask)
{
    return _mm256_mask_i32gather_pd(src, addr, vindex, mask, 8);
}

NPY_FINLINE __m256
fma_masked_load_ps(__m256 mask, npy_float* addr)
{
    return _mm256_maskload_ps(addr, _mm256_cvtps_epi32(mask));
}

NPY_FINLINE __m256d
fma_masked_load_pd(__m256i mask, npy_double* addr)
{
    return _mm256_maskload_pd(addr, mask);
}

NPY_FINLINE __m256
fma_set_masked_lanes_ps(__m256 x, __m256 val, __m256 mask)
{
    return _mm256_blendv_ps(x, val, mask);
}

NPY_FINLINE __m256d
fma_set_masked_lanes_pd(__m256d x, __m256d val, __m256d mask)
{
    return _mm256_blendv_pd(x, val, mask);
}

NPY_FINLINE __m256
fma_blend(__m256 x, __m256 y, __m256 ymask)
{
    return _mm256_blendv_ps(x, y, ymask);
}

NPY_FINLINE __m256
fma_get_exponent(__m256 x)
{
    /*
     * Special handling of denormals:
     * 1) Multiply denormal elements with 2**100 (0x71800000)
     * 2) Get the 8 bits of unbiased exponent
     * 3) Subtract 100 from exponent of denormals
     */

    __m256 two_power_100 = _mm256_castsi256_ps(_mm256_set1_epi32(0x71800000));
    __m256 denormal_mask = _mm256_cmp_ps(x, _mm256_set1_ps(FLT_MIN), _CMP_LT_OQ);
    __m256 normal_mask = _mm256_cmp_ps(x, _mm256_set1_ps(FLT_MIN), _CMP_GE_OQ);

    /*
     * The volatile is probably unnecessary now since we compile clang with
     * `-ftrapping-math`: https://github.com/numpy/numpy/issues/18005
     */
    volatile __m256 temp1 = _mm256_blendv_ps(x, _mm256_set1_ps(0.0f), normal_mask);
    __m256 temp = _mm256_mul_ps(temp1, two_power_100);
    x = _mm256_blendv_ps(x, temp, denormal_mask);

    __m256 exp = _mm256_cvtepi32_ps(
                    _mm256_sub_epi32(
                        _mm256_srli_epi32(
                            _mm256_castps_si256(x), 23),_mm256_set1_epi32(0x7E)));

    __m256 denorm_exp = _mm256_sub_ps(exp, _mm256_set1_ps(100.0f));
    return _mm256_blendv_ps(exp, denorm_exp, denormal_mask);
}

NPY_FINLINE __m256
fma_get_mantissa(__m256 x)
{
    /*
     * Special handling of denormals:
     * 1) Multiply denormal elements with 2**100 (0x71800000)
     * 2) Get the 23 bits of mantissa
     * 3) Mantissa for denormals is not affected by the multiplication
     */

    __m256 two_power_100 = _mm256_castsi256_ps(_mm256_set1_epi32(0x71800000));
    __m256 denormal_mask = _mm256_cmp_ps(x, _mm256_set1_ps(FLT_MIN), _CMP_LT_OQ);
    __m256 normal_mask = _mm256_cmp_ps(x, _mm256_set1_ps(FLT_MIN), _CMP_GE_OQ);

    /*
     * The volatile is probably unnecessary now since we compile clang with
     * `-ftrapping-math`: https://github.com/numpy/numpy/issues/18005
     */
    volatile __m256 temp1 = _mm256_blendv_ps(x, _mm256_set1_ps(0.0f), normal_mask);
    __m256 temp = _mm256_mul_ps(temp1, two_power_100);
    x = _mm256_blendv_ps(x, temp, denormal_mask);

    __m256i mantissa_bits = _mm256_set1_epi32(0x7fffff);
    __m256i exp_126_bits  = _mm256_set1_epi32(126 << 23);
    return _mm256_castsi256_ps(
                _mm256_or_si256(
                    _mm256_and_si256(
                        _mm256_castps_si256(x), mantissa_bits), exp_126_bits));
}

NPY_FINLINE __m256
fma_scalef_ps(__m256 poly, __m256 quadrant)
{
    /*
     * Handle denormals (which occur when quadrant <= -125):
     * 1) This function computes poly*(2^quad) by adding the exponent of
     poly to quad
     * 2) When quad <= -125, the output is a denormal and the above logic
     breaks down
     * 3) To handle such cases, we split quadrant: -125 + (quadrant + 125)
     * 4) poly*(2^-125) is computed the usual way
     * 5) 2^(quad-125) can be computed by: 2 << abs(quad-125)
     * 6) The final div operation generates the denormal
     */
     __m256 minquadrant = _mm256_set1_ps(-125.0f);
     __m256 denormal_mask = _mm256_cmp_ps(quadrant, minquadrant, _CMP_LE_OQ);
     if (_mm256_movemask_ps(denormal_mask) != 0x0000) {
        __m256 quad_diff = _mm256_sub_ps(quadrant, minquadrant);
        quad_diff = _mm256_sub_ps(_mm256_setzero_ps(), quad_diff);
        quad_diff = _mm256_blendv_ps(_mm256_setzero_ps(), quad_diff, denormal_mask);
        __m256i two_power_diff = _mm256_sllv_epi32(
                                   _mm256_set1_epi32(1), _mm256_cvtps_epi32(quad_diff));
        quadrant = _mm256_max_ps(quadrant, minquadrant);
        __m256i exponent = _mm256_slli_epi32(_mm256_cvtps_epi32(quadrant), 23);
        poly = _mm256_castsi256_ps(
                   _mm256_add_epi32(
                       _mm256_castps_si256(poly), exponent));
        __m256 denorm_poly = _mm256_div_ps(poly, _mm256_cvtepi32_ps(two_power_diff));
        return _mm256_blendv_ps(poly, denorm_poly, denormal_mask);
     }
     else {
        __m256i exponent = _mm256_slli_epi32(_mm256_cvtps_epi32(quadrant), 23);
        poly = _mm256_castsi256_ps(
                   _mm256_add_epi32(
                       _mm256_castps_si256(poly), exponent));
        return poly;
     }
}

#endif // SIMD_AVX2_FMA3

#ifdef SIMD_AVX512F

NPY_FINLINE __mmask16
avx512_get_full_load_mask_ps(void)
{
    return 0xFFFF;
}

NPY_FINLINE __mmask8
avx512_get_full_load_mask_pd(void)
{
    return 0xFF;
}

NPY_FINLINE __mmask16
avx512_get_partial_load_mask_ps(const npy_int num_elem, const npy_int total_elem)
{
    return (0x0001 << num_elem) - 0x0001;
}

NPY_FINLINE __mmask8
avx512_get_partial_load_mask_pd(const npy_int num_elem, const npy_int total_elem)
{
    return (0x01 << num_elem) - 0x01;
}

NPY_FINLINE __m512
avx512_masked_gather_ps(__m512 src,
                        npy_float* addr,
                        __m512i vindex,
                        __mmask16 kmask)
{
    return _mm512_mask_i32gather_ps(src, kmask, vindex, addr, 4);
}

NPY_FINLINE __m512d
avx512_masked_gather_pd(__m512d src,
                        npy_double* addr,
                        __m256i vindex,
                        __mmask8 kmask)
{
    return _mm512_mask_i32gather_pd(src, kmask, vindex, addr, 8);
}

NPY_FINLINE __m512
avx512_masked_load_ps(__mmask16 mask, npy_float* addr)
{
    return _mm512_maskz_loadu_ps(mask, (__m512 *)addr);
}

NPY_FINLINE __m512d
avx512_masked_load_pd(__mmask8 mask, npy_double* addr)
{
    return _mm512_maskz_loadu_pd(mask, (__m512d *)addr);
}

NPY_FINLINE __m512
avx512_set_masked_lanes_ps(__m512 x, __m512 val, __mmask16 mask)
{
    return _mm512_mask_blend_ps(mask, x, val);
}

NPY_FINLINE __m512d
avx512_set_masked_lanes_pd(__m512d x, __m512d val, __mmask8 mask)
{
    return _mm512_mask_blend_pd(mask, x, val);
}

NPY_FINLINE __m512
avx512_blend(__m512 x, __m512 y, __mmask16 ymask)
{
    return _mm512_mask_mov_ps(x, ymask, y);
}

NPY_FINLINE __m512
avx512_get_exponent(__m512 x)
{
    return _mm512_add_ps(_mm512_getexp_ps(x), _mm512_set1_ps(1.0f));
}

NPY_FINLINE __m512
avx512_get_mantissa(__m512 x)
{
    return _mm512_getmant_ps(x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
}

NPY_FINLINE __m512
avx512_scalef_ps(__m512 poly, __m512 quadrant)
{
    return _mm512_scalef_ps(poly, quadrant);
}

NPY_FINLINE __m512d
avx512_permute_x4var_pd(__m512d t0,
                        __m512d t1,
                        __m512d t2,
                        __m512d t3,
                        __m512i index)
{
    __mmask8 lut_mask = _mm512_cmp_epi64_mask(
                          _mm512_and_epi64(_mm512_set1_epi64(0x10ULL), index),
                          _mm512_set1_epi64(0), _MM_CMPINT_GT);
    __m512d res1 = _mm512_permutex2var_pd(t0, index, t1);
    __m512d res2 = _mm512_permutex2var_pd(t2, index, t3);
    return _mm512_mask_blend_pd(lut_mask, res1, res2);
}

NPY_FINLINE __m512d
avx512_permute_x8var_pd(__m512d t0, __m512d t1, __m512d t2, __m512d t3,
                        __m512d t4, __m512d t5, __m512d t6, __m512d t7,
                        __m512i index)
{
    __mmask8 lut_mask = _mm512_cmp_epi64_mask(
                          _mm512_and_epi64(_mm512_set1_epi64(0x20ULL), index),
                          _mm512_set1_epi64(0), _MM_CMPINT_GT);
    __m512d res1 = avx512_permute_x4var_pd(t0, t1, t2, t3, index);
    __m512d res2 = avx512_permute_x4var_pd(t4, t5, t6, t7, index);
    return _mm512_mask_blend_pd(lut_mask, res1, res2);
}

#endif // SIMD_AVX512F

#ifdef SIMD_AVX2_FMA3
/*
 * Vectorized Cody-Waite range reduction technique
 * Performs the reduction step x* = x - y*C in three steps:
 * 1) x* = x - y*c1
 * 2) x* = x - y*c2
 * 3) x* = x - y*c3
 * c1, c2 are exact floating points, c3 = C - c1 - c2 simulates higher precision
 */
NPY_FINLINE __m256
simd_range_reduction(__m256 x, __m256 y, __m256 c1, __m256 c2, __m256 c3)
{
    __m256 reduced_x = _mm256_fmadd_ps(y, c1, x);
    reduced_x = _mm256_fmadd_ps(y, c2, reduced_x);
    reduced_x = _mm256_fmadd_ps(y, c3, reduced_x);
    return reduced_x;
}
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
 * Vectorized Cody-Waite range reduction technique
 * Performs the reduction step x* = x - y*C in three steps:
 * 1) x* = x - y*c1
 * 2) x* = x - y*c2
 * 3) x* = x - y*c3
 * c1, c2 are exact floating points, c3 = C - c1 - c2 simulates higher precision
 */
NPY_FINLINE __m512
simd_range_reduction(__m512 x, __m512 y, __m512 c1, __m512 c2, __m512 c3)
{
    __m512 reduced_x = _mm512_fmadd_ps(y, c1, x);
    reduced_x = _mm512_fmadd_ps(y, c2, reduced_x);
    reduced_x = _mm512_fmadd_ps(y, c3, reduced_x);
    return reduced_x;
}
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

#if SIMD_ARM
#include <arm_neon.h>

#define V_EXP_TABLE_BITS 7
#define V_EXP_N (1 << V_EXP_TABLE_BITS)
#define V_EXP_INDEX_MASK (V_EXP_N - 1)

static const uint64_t neon_exp_table[V_EXP_N] = {
  0x3ff0000000000000, 0x3feff63da9fb3335, 0x3fefec9a3e778061,
  0x3fefe315e86e7f85, 0x3fefd9b0d3158574, 0x3fefd06b29ddf6de,
  0x3fefc74518759bc8, 0x3fefbe3ecac6f383, 0x3fefb5586cf9890f,
  0x3fefac922b7247f7, 0x3fefa3ec32d3d1a2, 0x3fef9b66affed31b,
  0x3fef9301d0125b51, 0x3fef8abdc06c31cc, 0x3fef829aaea92de0,
  0x3fef7a98c8a58e51, 0x3fef72b83c7d517b, 0x3fef6af9388c8dea,
  0x3fef635beb6fcb75, 0x3fef5be084045cd4, 0x3fef54873168b9aa,
  0x3fef4d5022fcd91d, 0x3fef463b88628cd6, 0x3fef3f49917ddc96,
  0x3fef387a6e756238, 0x3fef31ce4fb2a63f, 0x3fef2b4565e27cdd,
  0x3fef24dfe1f56381, 0x3fef1e9df51fdee1, 0x3fef187fd0dad990,
  0x3fef1285a6e4030b, 0x3fef0cafa93e2f56, 0x3fef06fe0a31b715,
  0x3fef0170fc4cd831, 0x3feefc08b26416ff, 0x3feef6c55f929ff1,
  0x3feef1a7373aa9cb, 0x3feeecae6d05d866, 0x3feee7db34e59ff7,
  0x3feee32dc313a8e5, 0x3feedea64c123422, 0x3feeda4504ac801c,
  0x3feed60a21f72e2a, 0x3feed1f5d950a897, 0x3feece086061892d,
  0x3feeca41ed1d0057, 0x3feec6a2b5c13cd0, 0x3feec32af0d7d3de,
  0x3feebfdad5362a27, 0x3feebcb299fddd0d, 0x3feeb9b2769d2ca7,
  0x3feeb6daa2cf6642, 0x3feeb42b569d4f82, 0x3feeb1a4ca5d920f,
  0x3feeaf4736b527da, 0x3feead12d497c7fd, 0x3feeab07dd485429,
  0x3feea9268a5946b7, 0x3feea76f15ad2148, 0x3feea5e1b976dc09,
  0x3feea47eb03a5585, 0x3feea34634ccc320, 0x3feea23882552225,
  0x3feea155d44ca973, 0x3feea09e667f3bcd, 0x3feea012750bdabf,
  0x3fee9fb23c651a2f, 0x3fee9f7df9519484, 0x3fee9f75e8ec5f74,
  0x3fee9f9a48a58174, 0x3fee9feb564267c9, 0x3feea0694fde5d3f,
  0x3feea11473eb0187, 0x3feea1ed0130c132, 0x3feea2f336cf4e62,
  0x3feea427543e1a12, 0x3feea589994cce13, 0x3feea71a4623c7ad,
  0x3feea8d99b4492ed, 0x3feeaac7d98a6699, 0x3feeace5422aa0db,
  0x3feeaf3216b5448c, 0x3feeb1ae99157736, 0x3feeb45b0b91ffc6,
  0x3feeb737b0cdc5e5, 0x3feeba44cbc8520f, 0x3feebd829fde4e50,
  0x3feec0f170ca07ba, 0x3feec49182a3f090, 0x3feec86319e32323,
  0x3feecc667b5de565, 0x3feed09bec4a2d33, 0x3feed503b23e255d,
  0x3feed99e1330b358, 0x3feede6b5579fdbf, 0x3feee36bbfd3f37a,
  0x3feee89f995ad3ad, 0x3feeee07298db666, 0x3feef3a2b84f15fb,
  0x3feef9728de5593a, 0x3feeff76f2fb5e47, 0x3fef05b030a1064a,
  0x3fef0c1e904bc1d2, 0x3fef12c25bd71e09, 0x3fef199bdd85529c,
  0x3fef20ab5fffd07a, 0x3fef27f12e57d14b, 0x3fef2f6d9406e7b5,
  0x3fef3720dcef9069, 0x3fef3f0b555dc3fa, 0x3fef472d4a07897c,
  0x3fef4f87080d89f2, 0x3fef5818dcfba487, 0x3fef60e316c98398,
  0x3fef69e603db3285, 0x3fef7321f301b460, 0x3fef7c97337b9b5f,
  0x3fef864614f5a129, 0x3fef902ee78b3ff6, 0x3fef9a51fbc74c83,
  0x3fefa4afa2a490da, 0x3fefaf482d8e67f1, 0x3fefba1bee615a27,
  0x3fefc52b376bba97, 0x3fefd0765b6e4540, 0x3fefdbfdad9cbe14,
  0x3fefe7c1819e90d8, 0x3feff3c22b8f71f1,
};

#define V_LOG_TABLE_BITS 7
#define V_LOG_N (1 << V_LOG_TABLE_BITS)
#define V_LOG_INDEX_MASK (V_LOG_N - 1)

struct neon_log_table_entry {
  double invc;
  double logc;
};

static const struct neon_log_table_entry neon_log_table[V_LOG_N] = {
  { 0x1.6a133d0dec120p+0, -0x1.62fe995eb963ap-2 },
  { 0x1.6815f2f3e42edp+0, -0x1.5d5a48dad6b67p-2 },
  { 0x1.661e39be1ac9ep+0, -0x1.57bde257d2769p-2 },
  { 0x1.642bfa30ac371p+0, -0x1.52294fbf2af55p-2 },
  { 0x1.623f1d916f323p+0, -0x1.4c9c7b598aa38p-2 },
  { 0x1.60578da220f65p+0, -0x1.47174fc5ff560p-2 },
  { 0x1.5e75349dea571p+0, -0x1.4199b7fa7b5cap-2 },
  { 0x1.5c97fd387a75ap+0, -0x1.3c239f48cfb99p-2 },
  { 0x1.5abfd2981f200p+0, -0x1.36b4f154d2aebp-2 },
  { 0x1.58eca051dc99cp+0, -0x1.314d9a0ff32fbp-2 },
  { 0x1.571e526d9df12p+0, -0x1.2bed85cca3cffp-2 },
  { 0x1.5554d555b3fcbp+0, -0x1.2694a11421af9p-2 },
  { 0x1.539015e2a20cdp+0, -0x1.2142d8d014fb2p-2 },
  { 0x1.51d0014ee0164p+0, -0x1.1bf81a2c77776p-2 },
  { 0x1.50148538cd9eep+0, -0x1.16b452a39c6a4p-2 },
  { 0x1.4e5d8f9f698a1p+0, -0x1.11776ffa6c67ep-2 },
  { 0x1.4cab0edca66bep+0, -0x1.0c416035020e0p-2 },
  { 0x1.4afcf1a9db874p+0, -0x1.071211aa10fdap-2 },
  { 0x1.495327136e16fp+0, -0x1.01e972e293b1bp-2 },
  { 0x1.47ad9e84af28fp+0, -0x1.f98ee587fd434p-3 },
  { 0x1.460c47b39ae15p+0, -0x1.ef5800ad716fbp-3 },
  { 0x1.446f12b278001p+0, -0x1.e52e160484698p-3 },
  { 0x1.42d5efdd720ecp+0, -0x1.db1104b19352ep-3 },
  { 0x1.4140cfe001a0fp+0, -0x1.d100ac59e0bd6p-3 },
  { 0x1.3fafa3b421f69p+0, -0x1.c6fced287c3bdp-3 },
  { 0x1.3e225c9c8ece5p+0, -0x1.bd05a7b317c29p-3 },
  { 0x1.3c98ec29a211ap+0, -0x1.b31abc229164fp-3 },
  { 0x1.3b13442a413fep+0, -0x1.a93c0edadb0a3p-3 },
  { 0x1.399156baa3c54p+0, -0x1.9f697ee30d7ddp-3 },
  { 0x1.38131639b4cdbp+0, -0x1.95a2efa9aa40ap-3 },
  { 0x1.36987540fbf53p+0, -0x1.8be843d796044p-3 },
  { 0x1.352166b648f61p+0, -0x1.82395ecc477edp-3 },
  { 0x1.33adddb3eb575p+0, -0x1.7896240966422p-3 },
  { 0x1.323dcd99fc1d3p+0, -0x1.6efe77aca8c55p-3 },
  { 0x1.30d129fefc7d2p+0, -0x1.65723e117ec5cp-3 },
  { 0x1.2f67e6b72fe7dp+0, -0x1.5bf15c0955706p-3 },
  { 0x1.2e01f7cf8b187p+0, -0x1.527bb6c111da1p-3 },
  { 0x1.2c9f518ddc86ep+0, -0x1.491133c939f8fp-3 },
  { 0x1.2b3fe86e5f413p+0, -0x1.3fb1b90c7fc58p-3 },
  { 0x1.29e3b1211b25cp+0, -0x1.365d2cc485f8dp-3 },
  { 0x1.288aa08b373cfp+0, -0x1.2d13758970de7p-3 },
  { 0x1.2734abcaa8467p+0, -0x1.23d47a721fd47p-3 },
  { 0x1.25e1c82459b81p+0, -0x1.1aa0229f25ec2p-3 },
  { 0x1.2491eb1ad59c5p+0, -0x1.117655ddebc3bp-3 },
  { 0x1.23450a54048b5p+0, -0x1.0856fbf83ab6bp-3 },
  { 0x1.21fb1bb09e578p+0, -0x1.fe83fabbaa106p-4 },
  { 0x1.20b415346d8f7p+0, -0x1.ec6e8507a56cdp-4 },
  { 0x1.1f6fed179a1acp+0, -0x1.da6d68c7cc2eap-4 },
  { 0x1.1e2e99b93c7b3p+0, -0x1.c88078462be0cp-4 },
  { 0x1.1cf011a7a882ap+0, -0x1.b6a786a423565p-4 },
  { 0x1.1bb44b97dba5ap+0, -0x1.a4e2676ac7f85p-4 },
  { 0x1.1a7b3e66cdd4fp+0, -0x1.9330eea777e76p-4 },
  { 0x1.1944e11dc56cdp+0, -0x1.8192f134d5ad9p-4 },
  { 0x1.18112aebb1a6ep+0, -0x1.70084464f0538p-4 },
  { 0x1.16e013231b7e9p+0, -0x1.5e90bdec5cb1fp-4 },
  { 0x1.15b1913f156cfp+0, -0x1.4d2c3433c5536p-4 },
  { 0x1.14859cdedde13p+0, -0x1.3bda7e219879ap-4 },
  { 0x1.135c2dc68cfa4p+0, -0x1.2a9b732d27194p-4 },
  { 0x1.12353bdb01684p+0, -0x1.196eeb2b10807p-4 },
  { 0x1.1110bf25b85b4p+0, -0x1.0854be8ef8a7ep-4 },
  { 0x1.0feeafd2f8577p+0, -0x1.ee998cb277432p-5 },
  { 0x1.0ecf062c51c3bp+0, -0x1.ccadb79919fb9p-5 },
  { 0x1.0db1baa076c8bp+0, -0x1.aae5b1d8618b0p-5 },
  { 0x1.0c96c5bb3048ep+0, -0x1.89413015d7442p-5 },
  { 0x1.0b7e20263e070p+0, -0x1.67bfe7bf158dep-5 },
  { 0x1.0a67c2acd0ce3p+0, -0x1.46618f83941bep-5 },
  { 0x1.0953a6391e982p+0, -0x1.2525df1b0618ap-5 },
  { 0x1.0841c3caea380p+0, -0x1.040c8e2f77c6ap-5 },
  { 0x1.07321489b13eap+0, -0x1.c62aad39f738ap-6 },
  { 0x1.062491aee9904p+0, -0x1.847fe3bdead9cp-6 },
  { 0x1.05193497a7cc5p+0, -0x1.43183683400acp-6 },
  { 0x1.040ff6b5f5e9fp+0, -0x1.01f31c4e1d544p-6 },
  { 0x1.0308d19aa6127p+0, -0x1.82201d1e6b69ap-7 },
  { 0x1.0203beedb0c67p+0, -0x1.00dd0f3e1bfd6p-7 },
  { 0x1.010037d38bcc2p+0, -0x1.ff6fe1feb4e53p-9 },
  { 1.0, 0.0 },
  { 0x1.fc06d493cca10p-1, 0x1.fe91885ec8e20p-8 },
  { 0x1.f81e6ac3b918fp-1, 0x1.fc516f716296dp-7 },
  { 0x1.f44546ef18996p-1, 0x1.7bb4dd70a015bp-6 },
  { 0x1.f07b10382c84bp-1, 0x1.f84c99b34b674p-6 },
  { 0x1.ecbf7070e59d4p-1, 0x1.39f9ce4fb2d71p-5 },
  { 0x1.e91213f715939p-1, 0x1.7756c0fd22e78p-5 },
  { 0x1.e572a9a75f7b7p-1, 0x1.b43ee82db8f3ap-5 },
  { 0x1.e1e0e2c530207p-1, 0x1.f0b3fced60034p-5 },
  { 0x1.de5c72d8a8be3p-1, 0x1.165bd78d4878ep-4 },
  { 0x1.dae50fa5658ccp-1, 0x1.3425d2715ebe6p-4 },
  { 0x1.d77a71145a2dap-1, 0x1.51b8bd91b7915p-4 },
  { 0x1.d41c51166623ep-1, 0x1.6f15632c76a47p-4 },
  { 0x1.d0ca6ba0bb29fp-1, 0x1.8c3c88ecbe503p-4 },
  { 0x1.cd847e8e59681p-1, 0x1.a92ef077625dap-4 },
  { 0x1.ca4a499693e00p-1, 0x1.c5ed5745fa006p-4 },
  { 0x1.c71b8e399e821p-1, 0x1.e27876de1c993p-4 },
  { 0x1.c3f80faf19077p-1, 0x1.fed104fce4cdcp-4 },
  { 0x1.c0df92dc2b0ecp-1, 0x1.0d7bd9c17d78bp-3 },
  { 0x1.bdd1de3cbb542p-1, 0x1.1b76986cef97bp-3 },
  { 0x1.baceb9e1007a3p-1, 0x1.295913d24f750p-3 },
  { 0x1.b7d5ef543e55ep-1, 0x1.37239fa295d17p-3 },
  { 0x1.b4e749977d953p-1, 0x1.44d68dd78714bp-3 },
  { 0x1.b20295155478ep-1, 0x1.52722ebe5d780p-3 },
  { 0x1.af279f8e82be2p-1, 0x1.5ff6d12671f98p-3 },
  { 0x1.ac5638197fdf3p-1, 0x1.6d64c2389484bp-3 },
  { 0x1.a98e2f102e087p-1, 0x1.7abc4da40fddap-3 },
  { 0x1.a6cf5606d05c1p-1, 0x1.87fdbda1e8452p-3 },
  { 0x1.a4197fc04d746p-1, 0x1.95295b06a5f37p-3 },
  { 0x1.a16c80293dc01p-1, 0x1.a23f6d34abbc5p-3 },
  { 0x1.9ec82c4dc5bc9p-1, 0x1.af403a28e04f2p-3 },
  { 0x1.9c2c5a491f534p-1, 0x1.bc2c06a85721ap-3 },
  { 0x1.9998e1480b618p-1, 0x1.c903161240163p-3 },
  { 0x1.970d9977c6c2dp-1, 0x1.d5c5aa93287ebp-3 },
  { 0x1.948a5c023d212p-1, 0x1.e274051823fa9p-3 },
  { 0x1.920f0303d6809p-1, 0x1.ef0e656300c16p-3 },
  { 0x1.8f9b698a98b45p-1, 0x1.fb9509f05aa2ap-3 },
  { 0x1.8d2f6b81726f6p-1, 0x1.04041821f37afp-2 },
  { 0x1.8acae5bb55badp-1, 0x1.0a340a49b3029p-2 },
  { 0x1.886db5d9275b8p-1, 0x1.105a7918a126dp-2 },
  { 0x1.8617ba567c13cp-1, 0x1.1677819812b84p-2 },
  { 0x1.83c8d27487800p-1, 0x1.1c8b405b40c0ep-2 },
  { 0x1.8180de3c5dbe7p-1, 0x1.2295d16cfa6b1p-2 },
  { 0x1.7f3fbe71cdb71p-1, 0x1.28975066318a2p-2 },
  { 0x1.7d055498071c1p-1, 0x1.2e8fd855d86fcp-2 },
  { 0x1.7ad182e54f65ap-1, 0x1.347f83d605e59p-2 },
  { 0x1.78a42c3c90125p-1, 0x1.3a666d1244588p-2 },
  { 0x1.767d342f76944p-1, 0x1.4044adb6f8ec4p-2 },
  { 0x1.745c7ef26b00ap-1, 0x1.461a5f077558cp-2 },
  { 0x1.7241f15769d0fp-1, 0x1.4be799e20b9c8p-2 },
  { 0x1.702d70d396e41p-1, 0x1.51ac76a6b79dfp-2 },
  { 0x1.6e1ee3700cd11p-1, 0x1.57690d5744a45p-2 },
  { 0x1.6c162fc9cbe02p-1, 0x1.5d1d758e45217p-2 },
};

static inline uint64x2_t
neon_exp_lookup_sbits(uint64x2_t i)
{
  uint64_t i0 = vgetq_lane_u64(i, 0) & V_EXP_INDEX_MASK;
  uint64_t i1 = vgetq_lane_u64(i, 1) & V_EXP_INDEX_MASK;
  return vcombine_u64(vcreate_u64(neon_exp_table[i0]), vcreate_u64(neon_exp_table[i1]));
}

static inline int neon_has_any_lane_u32(uint32x4_t mask)
{
  return vmaxvq_u32(mask) != 0;
}

static inline int neon_has_any_lane_u64(uint64x2_t mask)
{
  return (vgetq_lane_u64(mask, 0) | vgetq_lane_u64(mask, 1)) != 0;
}

static inline int neon_has_all_lanes_u32(uint32x4_t mask)
{
  return vminvq_u32(mask) != 0;
}

static inline int neon_has_all_lanes_u64(uint64x2_t mask)
{
  return (vgetq_lane_u64(mask, 0) & vgetq_lane_u64(mask, 1)) != 0;
}

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
    float64x2_t x0, x1, x2, x3, x4, x5, x6, x7;

    if (ssrc == 1) {
      __builtin_prefetch(infp + 64, 0, 3);
      x0 = vld1q_f64(infp);
      x1 = vld1q_f64(infp + 2);
      x2 = vld1q_f64(infp + 4);
      x3 = vld1q_f64(infp + 6);
      x4 = vld1q_f64(infp + 8);
      x5 = vld1q_f64(infp + 10);
      x6 = vld1q_f64(infp + 12);
      x7 = vld1q_f64(infp + 14);
    } else {
      double vals0[2] = {infp[0], infp[ssrc]};
      double vals1[2] = {infp[2 * ssrc], infp[3 * ssrc]};
      double vals2[2] = {infp[4 * ssrc], infp[5 * ssrc]};
      double vals3[2] = {infp[6 * ssrc], infp[7 * ssrc]};
      double vals4[2] = {infp[8 * ssrc], infp[9 * ssrc]};
      double vals5[2] = {infp[10 * ssrc], infp[11 * ssrc]};
      double vals6[2] = {infp[12 * ssrc], infp[13 * ssrc]};
      double vals7[2] = {infp[14 * ssrc], infp[15 * ssrc]};
      x0 = vld1q_f64(vals0);
      x1 = vld1q_f64(vals1);
      x2 = vld1q_f64(vals2);
      x3 = vld1q_f64(vals3);
      x4 = vld1q_f64(vals4);
      x5 = vld1q_f64(vals5);
      x6 = vld1q_f64(vals6);
      x7 = vld1q_f64(vals7);
    }

    uint64x2_t need_special0 = vceqq_f64(x0, pos_inf);
    uint64x2_t need_special1 = vceqq_f64(x1, pos_inf);
    uint64x2_t need_special2 = vceqq_f64(x2, pos_inf);
    uint64x2_t need_special3 = vceqq_f64(x3, pos_inf);
    uint64x2_t need_special4 = vceqq_f64(x4, pos_inf);
    uint64x2_t need_special5 = vceqq_f64(x5, pos_inf);
    uint64x2_t need_special6 = vceqq_f64(x6, pos_inf);
    uint64x2_t need_special7 = vceqq_f64(x7, pos_inf);
    need_special0 = vorrq_u64(need_special0, vceqq_f64(x0, neg_inf));
    need_special1 = vorrq_u64(need_special1, vceqq_f64(x1, neg_inf));
    need_special2 = vorrq_u64(need_special2, vceqq_f64(x2, neg_inf));
    need_special3 = vorrq_u64(need_special3, vceqq_f64(x3, neg_inf));
    need_special4 = vorrq_u64(need_special4, vceqq_f64(x4, neg_inf));
    need_special5 = vorrq_u64(need_special5, vceqq_f64(x5, neg_inf));
    need_special6 = vorrq_u64(need_special6, vceqq_f64(x6, neg_inf));
    need_special7 = vorrq_u64(need_special7, vceqq_f64(x7, neg_inf));
    uint64x2_t is_nan0 = vceqq_f64(x0, x0);
    uint64x2_t is_nan1 = vceqq_f64(x1, x1);
    uint64x2_t is_nan2 = vceqq_f64(x2, x2);
    uint64x2_t is_nan3 = vceqq_f64(x3, x3);
    uint64x2_t is_nan4 = vceqq_f64(x4, x4);
    uint64x2_t is_nan5 = vceqq_f64(x5, x5);
    uint64x2_t is_nan6 = vceqq_f64(x6, x6);
    uint64x2_t is_nan7 = vceqq_f64(x7, x7);
    is_nan0 = veorq_u64(is_nan0, all_ones);
    is_nan1 = veorq_u64(is_nan1, all_ones);
    is_nan2 = veorq_u64(is_nan2, all_ones);
    is_nan3 = veorq_u64(is_nan3, all_ones);
    is_nan4 = veorq_u64(is_nan4, all_ones);
    is_nan5 = veorq_u64(is_nan5, all_ones);
    is_nan6 = veorq_u64(is_nan6, all_ones);
    is_nan7 = veorq_u64(is_nan7, all_ones);
    need_special0 = vorrq_u64(need_special0, is_nan0);
    need_special1 = vorrq_u64(need_special1, is_nan1);
    need_special2 = vorrq_u64(need_special2, is_nan2);
    need_special3 = vorrq_u64(need_special3, is_nan3);
    need_special4 = vorrq_u64(need_special4, is_nan4);
    need_special5 = vorrq_u64(need_special5, is_nan5);
    need_special6 = vorrq_u64(need_special6, is_nan6);
    need_special7 = vorrq_u64(need_special7, is_nan7);
    need_special0 = vorrq_u64(need_special0, vcgtq_f64(x0, exp_max));
    need_special1 = vorrq_u64(need_special1, vcgtq_f64(x1, exp_max));
    need_special2 = vorrq_u64(need_special2, vcgtq_f64(x2, exp_max));
    need_special3 = vorrq_u64(need_special3, vcgtq_f64(x3, exp_max));
    need_special4 = vorrq_u64(need_special4, vcgtq_f64(x4, exp_max));
    need_special5 = vorrq_u64(need_special5, vcgtq_f64(x5, exp_max));
    need_special6 = vorrq_u64(need_special6, vcgtq_f64(x6, exp_max));
    need_special7 = vorrq_u64(need_special7, vcgtq_f64(x7, exp_max));
    need_special0 = vorrq_u64(need_special0, vcltq_f64(x0, exp_min));
    need_special1 = vorrq_u64(need_special1, vcltq_f64(x1, exp_min));
    need_special2 = vorrq_u64(need_special2, vcltq_f64(x2, exp_min));
    need_special3 = vorrq_u64(need_special3, vcltq_f64(x3, exp_min));
    need_special4 = vorrq_u64(need_special4, vcltq_f64(x4, exp_min));
    need_special5 = vorrq_u64(need_special5, vcltq_f64(x5, exp_min));
    need_special6 = vorrq_u64(need_special6, vcltq_f64(x6, exp_min));
    need_special7 = vorrq_u64(need_special7, vcltq_f64(x7, exp_min));

    float64x2_t safe0 = vbslq_f64(need_special0, zero_val, x0);
    float64x2_t safe1 = vbslq_f64(need_special1, zero_val, x1);
    float64x2_t safe2 = vbslq_f64(need_special2, zero_val, x2);
    float64x2_t safe3 = vbslq_f64(need_special3, zero_val, x3);
    float64x2_t safe4 = vbslq_f64(need_special4, zero_val, x4);
    float64x2_t safe5 = vbslq_f64(need_special5, zero_val, x5);
    float64x2_t safe6 = vbslq_f64(need_special6, zero_val, x6);
    float64x2_t safe7 = vbslq_f64(need_special7, zero_val, x7);

    float64x2_t z0 = vfmaq_f64(shift, safe0, inv_ln2);
    float64x2_t z1 = vfmaq_f64(shift, safe1, inv_ln2);
    float64x2_t z2 = vfmaq_f64(shift, safe2, inv_ln2);
    float64x2_t z3 = vfmaq_f64(shift, safe3, inv_ln2);
    float64x2_t z4 = vfmaq_f64(shift, safe4, inv_ln2);
    float64x2_t z5 = vfmaq_f64(shift, safe5, inv_ln2);
    float64x2_t z6 = vfmaq_f64(shift, safe6, inv_ln2);
    float64x2_t z7 = vfmaq_f64(shift, safe7, inv_ln2);
    uint64x2_t u0 = vreinterpretq_u64_f64(z0);
    uint64x2_t u1 = vreinterpretq_u64_f64(z1);
    uint64x2_t u2 = vreinterpretq_u64_f64(z2);
    uint64x2_t u3 = vreinterpretq_u64_f64(z3);
    uint64x2_t u4 = vreinterpretq_u64_f64(z4);
    uint64x2_t u5 = vreinterpretq_u64_f64(z5);
    uint64x2_t u6 = vreinterpretq_u64_f64(z6);
    uint64x2_t u7 = vreinterpretq_u64_f64(z7);
    float64x2_t n0 = vsubq_f64(z0, shift);
    float64x2_t n1 = vsubq_f64(z1, shift);
    float64x2_t n2 = vsubq_f64(z2, shift);
    float64x2_t n3 = vsubq_f64(z3, shift);
    float64x2_t n4 = vsubq_f64(z4, shift);
    float64x2_t n5 = vsubq_f64(z5, shift);
    float64x2_t n6 = vsubq_f64(z6, shift);
    float64x2_t n7 = vsubq_f64(z7, shift);

    float64x2_t r0 = vfmsq_f64(vfmsq_f64(safe0, n0, ln2_hi), n0, ln2_lo);
    float64x2_t r1 = vfmsq_f64(vfmsq_f64(safe1, n1, ln2_hi), n1, ln2_lo);
    float64x2_t r2 = vfmsq_f64(vfmsq_f64(safe2, n2, ln2_hi), n2, ln2_lo);
    float64x2_t r3 = vfmsq_f64(vfmsq_f64(safe3, n3, ln2_hi), n3, ln2_lo);
    float64x2_t r4 = vfmsq_f64(vfmsq_f64(safe4, n4, ln2_hi), n4, ln2_lo);
    float64x2_t r5 = vfmsq_f64(vfmsq_f64(safe5, n5, ln2_hi), n5, ln2_lo);
    float64x2_t r6 = vfmsq_f64(vfmsq_f64(safe6, n6, ln2_hi), n6, ln2_lo);
    float64x2_t r7 = vfmsq_f64(vfmsq_f64(safe7, n7, ln2_hi), n7, ln2_lo);

    uint64x2_t e0 = vshlq_n_u64(u0, 52 - V_EXP_TABLE_BITS);
    uint64x2_t e1 = vshlq_n_u64(u1, 52 - V_EXP_TABLE_BITS);
    uint64x2_t e2 = vshlq_n_u64(u2, 52 - V_EXP_TABLE_BITS);
    uint64x2_t e3 = vshlq_n_u64(u3, 52 - V_EXP_TABLE_BITS);
    uint64x2_t e4 = vshlq_n_u64(u4, 52 - V_EXP_TABLE_BITS);
    uint64x2_t e5 = vshlq_n_u64(u5, 52 - V_EXP_TABLE_BITS);
    uint64x2_t e6 = vshlq_n_u64(u6, 52 - V_EXP_TABLE_BITS);
    uint64x2_t e7 = vshlq_n_u64(u7, 52 - V_EXP_TABLE_BITS);

    uint64x2_t tab0 = neon_exp_lookup_sbits(u0);
    uint64x2_t tab1 = neon_exp_lookup_sbits(u1);
    uint64x2_t tab2 = neon_exp_lookup_sbits(u2);
    uint64x2_t tab3 = neon_exp_lookup_sbits(u3);
    uint64x2_t tab4 = neon_exp_lookup_sbits(u4);
    uint64x2_t tab5 = neon_exp_lookup_sbits(u5);
    uint64x2_t tab6 = neon_exp_lookup_sbits(u6);
    uint64x2_t tab7 = neon_exp_lookup_sbits(u7);
    float64x2_t scale0 = vreinterpretq_f64_u64(vaddq_u64(tab0, e0));
    float64x2_t scale1 = vreinterpretq_f64_u64(vaddq_u64(tab1, e1));
    float64x2_t scale2 = vreinterpretq_f64_u64(vaddq_u64(tab2, e2));
    float64x2_t scale3 = vreinterpretq_f64_u64(vaddq_u64(tab3, e3));
    float64x2_t scale4 = vreinterpretq_f64_u64(vaddq_u64(tab4, e4));
    float64x2_t scale5 = vreinterpretq_f64_u64(vaddq_u64(tab5, e5));
    float64x2_t scale6 = vreinterpretq_f64_u64(vaddq_u64(tab6, e6));
    float64x2_t scale7 = vreinterpretq_f64_u64(vaddq_u64(tab7, e7));

    float64x2_t r20 = vmulq_f64(r0, r0);
    float64x2_t r21 = vmulq_f64(r1, r1);
    float64x2_t r22 = vmulq_f64(r2, r2);
    float64x2_t r23 = vmulq_f64(r3, r3);
    float64x2_t r24 = vmulq_f64(r4, r4);
    float64x2_t r25 = vmulq_f64(r5, r5);
    float64x2_t r26 = vmulq_f64(r6, r6);
    float64x2_t r27 = vmulq_f64(r7, r7);
    float64x2_t poly0 = vfmaq_f64(vfmaq_f64(c0, r0, c1), r20, c2);
    float64x2_t poly1 = vfmaq_f64(vfmaq_f64(c0, r1, c1), r21, c2);
    float64x2_t poly2 = vfmaq_f64(vfmaq_f64(c0, r2, c1), r22, c2);
    float64x2_t poly3 = vfmaq_f64(vfmaq_f64(c0, r3, c1), r23, c2);
    float64x2_t poly4 = vfmaq_f64(vfmaq_f64(c0, r4, c1), r24, c2);
    float64x2_t poly5 = vfmaq_f64(vfmaq_f64(c0, r5, c1), r25, c2);
    float64x2_t poly6 = vfmaq_f64(vfmaq_f64(c0, r6, c1), r26, c2);
    float64x2_t poly7 = vfmaq_f64(vfmaq_f64(c0, r7, c1), r27, c2);
    poly0 = vfmaq_f64(r0, poly0, r20);
    poly1 = vfmaq_f64(r1, poly1, r21);
    poly2 = vfmaq_f64(r2, poly2, r22);
    poly3 = vfmaq_f64(r3, poly3, r23);
    poly4 = vfmaq_f64(r4, poly4, r24);
    poly5 = vfmaq_f64(r5, poly5, r25);
    poly6 = vfmaq_f64(r6, poly6, r26);
    poly7 = vfmaq_f64(r7, poly7, r27);

    float64x2_t result0 = vfmaq_f64(scale0, poly0, scale0);
    float64x2_t result1 = vfmaq_f64(scale1, poly1, scale1);
    float64x2_t result2 = vfmaq_f64(scale2, poly2, scale2);
    float64x2_t result3 = vfmaq_f64(scale3, poly3, scale3);
    float64x2_t result4 = vfmaq_f64(scale4, poly4, scale4);
    float64x2_t result5 = vfmaq_f64(scale5, poly5, scale5);
    float64x2_t result6 = vfmaq_f64(scale6, poly6, scale6);
    float64x2_t result7 = vfmaq_f64(scale7, poly7, scale7);

    uint64x2_t is_overflow0 = vcgtq_f64(x0, exp_max);
    uint64x2_t is_overflow1 = vcgtq_f64(x1, exp_max);
    uint64x2_t is_overflow2 = vcgtq_f64(x2, exp_max);
    uint64x2_t is_overflow3 = vcgtq_f64(x3, exp_max);
    uint64x2_t is_overflow4 = vcgtq_f64(x4, exp_max);
    uint64x2_t is_overflow5 = vcgtq_f64(x5, exp_max);
    uint64x2_t is_overflow6 = vcgtq_f64(x6, exp_max);
    uint64x2_t is_overflow7 = vcgtq_f64(x7, exp_max);
    uint64x2_t is_underflow0 = vcltq_f64(x0, exp_min);
    uint64x2_t is_underflow1 = vcltq_f64(x1, exp_min);
    uint64x2_t is_underflow2 = vcltq_f64(x2, exp_min);
    uint64x2_t is_underflow3 = vcltq_f64(x3, exp_min);
    uint64x2_t is_underflow4 = vcltq_f64(x4, exp_min);
    uint64x2_t is_underflow5 = vcltq_f64(x5, exp_min);
    uint64x2_t is_underflow6 = vcltq_f64(x6, exp_min);
    uint64x2_t is_underflow7 = vcltq_f64(x7, exp_min);

    result0 = vbslq_f64(is_nan0, x0, result0);
    result1 = vbslq_f64(is_nan1, x1, result1);
    result2 = vbslq_f64(is_nan2, x2, result2);
    result3 = vbslq_f64(is_nan3, x3, result3);
    result4 = vbslq_f64(is_nan4, x4, result4);
    result5 = vbslq_f64(is_nan5, x5, result5);
    result6 = vbslq_f64(is_nan6, x6, result6);
    result7 = vbslq_f64(is_nan7, x7, result7);
    result0 = vbslq_f64(vceqq_f64(x0, pos_inf), pos_inf, result0);
    result1 = vbslq_f64(vceqq_f64(x1, pos_inf), pos_inf, result1);
    result2 = vbslq_f64(vceqq_f64(x2, pos_inf), pos_inf, result2);
    result3 = vbslq_f64(vceqq_f64(x3, pos_inf), pos_inf, result3);
    result4 = vbslq_f64(vceqq_f64(x4, pos_inf), pos_inf, result4);
    result5 = vbslq_f64(vceqq_f64(x5, pos_inf), pos_inf, result5);
    result6 = vbslq_f64(vceqq_f64(x6, pos_inf), pos_inf, result6);
    result7 = vbslq_f64(vceqq_f64(x7, pos_inf), pos_inf, result7);
    result0 = vbslq_f64(vceqq_f64(x0, neg_inf), zero_val, result0);
    result1 = vbslq_f64(vceqq_f64(x1, neg_inf), zero_val, result1);
    result2 = vbslq_f64(vceqq_f64(x2, neg_inf), zero_val, result2);
    result3 = vbslq_f64(vceqq_f64(x3, neg_inf), zero_val, result3);
    result4 = vbslq_f64(vceqq_f64(x4, neg_inf), zero_val, result4);
    result5 = vbslq_f64(vceqq_f64(x5, neg_inf), zero_val, result5);
    result6 = vbslq_f64(vceqq_f64(x6, neg_inf), zero_val, result6);
    result7 = vbslq_f64(vceqq_f64(x7, neg_inf), zero_val, result7);
    result0 = vbslq_f64(is_overflow0, pos_inf, result0);
    result1 = vbslq_f64(is_overflow1, pos_inf, result1);
    result2 = vbslq_f64(is_overflow2, pos_inf, result2);
    result3 = vbslq_f64(is_overflow3, pos_inf, result3);
    result4 = vbslq_f64(is_overflow4, pos_inf, result4);
    result5 = vbslq_f64(is_overflow5, pos_inf, result5);
    result6 = vbslq_f64(is_overflow6, pos_inf, result6);
    result7 = vbslq_f64(is_overflow7, pos_inf, result7);
    result0 = vbslq_f64(is_underflow0, zero_val, result0);
    result1 = vbslq_f64(is_underflow1, zero_val, result1);
    result2 = vbslq_f64(is_underflow2, zero_val, result2);
    result3 = vbslq_f64(is_underflow3, zero_val, result3);
    result4 = vbslq_f64(is_underflow4, zero_val, result4);
    result5 = vbslq_f64(is_underflow5, zero_val, result5);
    result6 = vbslq_f64(is_underflow6, zero_val, result6);
    result7 = vbslq_f64(is_underflow7, zero_val, result7);

    if (sdst == 1) {
      __builtin_prefetch(outp + 64, 1, 3);
      vst1q_f64(outp, result0);
      vst1q_f64(outp + 2, result1);
      vst1q_f64(outp + 4, result2);
      vst1q_f64(outp + 6, result3);
      vst1q_f64(outp + 8, result4);
      vst1q_f64(outp + 10, result5);
      vst1q_f64(outp + 12, result6);
      vst1q_f64(outp + 14, result7);
    } else {
      for (int i = 0; i < 2; i++) {
        outp[i * sdst] = vgetq_lane_f64(result0, i);
        outp[(2 + i) * sdst] = vgetq_lane_f64(result1, i);
        outp[(4 + i) * sdst] = vgetq_lane_f64(result2, i);
        outp[(6 + i) * sdst] = vgetq_lane_f64(result3, i);
        outp[(8 + i) * sdst] = vgetq_lane_f64(result4, i);
        outp[(10 + i) * sdst] = vgetq_lane_f64(result5, i);
        outp[(12 + i) * sdst] = vgetq_lane_f64(result6, i);
        outp[(14 + i) * sdst] = vgetq_lane_f64(result7, i);
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
    uint64x2_t special_mask = vorrq_u64(vorrq_u64(is_nan, is_pos_inf), is_neg_inf);
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
  const float32x4_t c0 = vdupq_n_f32(0x1.0e4020p-7f);
  const float32x4_t c1 = vdupq_n_f32(0x1.573e2ep-5f);
  const float32x4_t c2 = vdupq_n_f32(0x1.555e66p-3f);
  const float32x4_t c3 = vdupq_n_f32(0x1.fffdb6p-2f);
  const float32x4_t c4 = vdupq_n_f32(0x1.ffffecp-1f);
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
    float32x4_t x0, x1, x2, x3, x4, x5, x6, x7;

    if (ssrc == 1) {
      __builtin_prefetch(infp + 128, 0, 3);
      x0 = vld1q_f32(infp);
      x1 = vld1q_f32(infp + 4);
      x2 = vld1q_f32(infp + 8);
      x3 = vld1q_f32(infp + 12);
      x4 = vld1q_f32(infp + 16);
      x5 = vld1q_f32(infp + 20);
      x6 = vld1q_f32(infp + 24);
      x7 = vld1q_f32(infp + 28);
    } else {
      float vals0[4], vals1[4], vals2[4], vals3[4];
      float vals4[4], vals5[4], vals6[4], vals7[4];
      for (int i = 0; i < 4; i++) {
        vals0[i] = infp[i * ssrc];
        vals1[i] = infp[(4 + i) * ssrc];
        vals2[i] = infp[(8 + i) * ssrc];
        vals3[i] = infp[(12 + i) * ssrc];
        vals4[i] = infp[(16 + i) * ssrc];
        vals5[i] = infp[(20 + i) * ssrc];
        vals6[i] = infp[(24 + i) * ssrc];
        vals7[i] = infp[(28 + i) * ssrc];
      }
      x0 = vld1q_f32(vals0);
      x1 = vld1q_f32(vals1);
      x2 = vld1q_f32(vals2);
      x3 = vld1q_f32(vals3);
      x4 = vld1q_f32(vals4);
      x5 = vld1q_f32(vals5);
      x6 = vld1q_f32(vals6);
      x7 = vld1q_f32(vals7);
    }

    uint32x4_t need_special0 = vceqq_f32(x0, pos_inf);
    uint32x4_t need_special1 = vceqq_f32(x1, pos_inf);
    uint32x4_t need_special2 = vceqq_f32(x2, pos_inf);
    uint32x4_t need_special3 = vceqq_f32(x3, pos_inf);
    uint32x4_t need_special4 = vceqq_f32(x4, pos_inf);
    uint32x4_t need_special5 = vceqq_f32(x5, pos_inf);
    uint32x4_t need_special6 = vceqq_f32(x6, pos_inf);
    uint32x4_t need_special7 = vceqq_f32(x7, pos_inf);
    need_special0 = vorrq_u32(need_special0, vceqq_f32(x0, neg_inf));
    need_special1 = vorrq_u32(need_special1, vceqq_f32(x1, neg_inf));
    need_special2 = vorrq_u32(need_special2, vceqq_f32(x2, neg_inf));
    need_special3 = vorrq_u32(need_special3, vceqq_f32(x3, neg_inf));
    need_special4 = vorrq_u32(need_special4, vceqq_f32(x4, neg_inf));
    need_special5 = vorrq_u32(need_special5, vceqq_f32(x5, neg_inf));
    need_special6 = vorrq_u32(need_special6, vceqq_f32(x6, neg_inf));
    need_special7 = vorrq_u32(need_special7, vceqq_f32(x7, neg_inf));
    uint32x4_t is_nan0 = vceqq_f32(x0, x0);
    uint32x4_t is_nan1 = vceqq_f32(x1, x1);
    uint32x4_t is_nan2 = vceqq_f32(x2, x2);
    uint32x4_t is_nan3 = vceqq_f32(x3, x3);
    uint32x4_t is_nan4 = vceqq_f32(x4, x4);
    uint32x4_t is_nan5 = vceqq_f32(x5, x5);
    uint32x4_t is_nan6 = vceqq_f32(x6, x6);
    uint32x4_t is_nan7 = vceqq_f32(x7, x7);
    is_nan0 = vmvnq_u32(is_nan0);
    is_nan1 = vmvnq_u32(is_nan1);
    is_nan2 = vmvnq_u32(is_nan2);
    is_nan3 = vmvnq_u32(is_nan3);
    is_nan4 = vmvnq_u32(is_nan4);
    is_nan5 = vmvnq_u32(is_nan5);
    is_nan6 = vmvnq_u32(is_nan6);
    is_nan7 = vmvnq_u32(is_nan7);
    need_special0 = vorrq_u32(need_special0, is_nan0);
    need_special1 = vorrq_u32(need_special1, is_nan1);
    need_special2 = vorrq_u32(need_special2, is_nan2);
    need_special3 = vorrq_u32(need_special3, is_nan3);
    need_special4 = vorrq_u32(need_special4, is_nan4);
    need_special5 = vorrq_u32(need_special5, is_nan5);
    need_special6 = vorrq_u32(need_special6, is_nan6);
    need_special7 = vorrq_u32(need_special7, is_nan7);
    need_special0 = vorrq_u32(need_special0, vcgtq_f32(x0, exp_max));
    need_special1 = vorrq_u32(need_special1, vcgtq_f32(x1, exp_max));
    need_special2 = vorrq_u32(need_special2, vcgtq_f32(x2, exp_max));
    need_special3 = vorrq_u32(need_special3, vcgtq_f32(x3, exp_max));
    need_special4 = vorrq_u32(need_special4, vcgtq_f32(x4, exp_max));
    need_special5 = vorrq_u32(need_special5, vcgtq_f32(x5, exp_max));
    need_special6 = vorrq_u32(need_special6, vcgtq_f32(x6, exp_max));
    need_special7 = vorrq_u32(need_special7, vcgtq_f32(x7, exp_max));
    need_special0 = vorrq_u32(need_special0, vcltq_f32(x0, exp_min));
    need_special1 = vorrq_u32(need_special1, vcltq_f32(x1, exp_min));
    need_special2 = vorrq_u32(need_special2, vcltq_f32(x2, exp_min));
    need_special3 = vorrq_u32(need_special3, vcltq_f32(x3, exp_min));
    need_special4 = vorrq_u32(need_special4, vcltq_f32(x4, exp_min));
    need_special5 = vorrq_u32(need_special5, vcltq_f32(x5, exp_min));
    need_special6 = vorrq_u32(need_special6, vcltq_f32(x6, exp_min));
    need_special7 = vorrq_u32(need_special7, vcltq_f32(x7, exp_min));

    float32x4_t safe0 = vbslq_f32(need_special0, zero_val, x0);
    float32x4_t safe1 = vbslq_f32(need_special1, zero_val, x1);
    float32x4_t safe2 = vbslq_f32(need_special2, zero_val, x2);
    float32x4_t safe3 = vbslq_f32(need_special3, zero_val, x3);
    float32x4_t safe4 = vbslq_f32(need_special4, zero_val, x4);
    float32x4_t safe5 = vbslq_f32(need_special5, zero_val, x5);
    float32x4_t safe6 = vbslq_f32(need_special6, zero_val, x6);
    float32x4_t safe7 = vbslq_f32(need_special7, zero_val, x7);

    float32x4_t z0 = vfmaq_f32(shift, safe0, inv_ln2);
    float32x4_t z1 = vfmaq_f32(shift, safe1, inv_ln2);
    float32x4_t z2 = vfmaq_f32(shift, safe2, inv_ln2);
    float32x4_t z3 = vfmaq_f32(shift, safe3, inv_ln2);
    float32x4_t z4 = vfmaq_f32(shift, safe4, inv_ln2);
    float32x4_t z5 = vfmaq_f32(shift, safe5, inv_ln2);
    float32x4_t z6 = vfmaq_f32(shift, safe6, inv_ln2);
    float32x4_t z7 = vfmaq_f32(shift, safe7, inv_ln2);
    float32x4_t n0 = vsubq_f32(z0, shift);
    float32x4_t n1 = vsubq_f32(z1, shift);
    float32x4_t n2 = vsubq_f32(z2, shift);
    float32x4_t n3 = vsubq_f32(z3, shift);
    float32x4_t n4 = vsubq_f32(z4, shift);
    float32x4_t n5 = vsubq_f32(z5, shift);
    float32x4_t n6 = vsubq_f32(z6, shift);
    float32x4_t n7 = vsubq_f32(z7, shift);

    float32x4_t r0 = vfmsq_f32(vfmsq_f32(safe0, n0, ln2_hi), n0, ln2_lo);
    float32x4_t r1 = vfmsq_f32(vfmsq_f32(safe1, n1, ln2_hi), n1, ln2_lo);
    float32x4_t r2 = vfmsq_f32(vfmsq_f32(safe2, n2, ln2_hi), n2, ln2_lo);
    float32x4_t r3 = vfmsq_f32(vfmsq_f32(safe3, n3, ln2_hi), n3, ln2_lo);
    float32x4_t r4 = vfmsq_f32(vfmsq_f32(safe4, n4, ln2_hi), n4, ln2_lo);
    float32x4_t r5 = vfmsq_f32(vfmsq_f32(safe5, n5, ln2_hi), n5, ln2_lo);
    float32x4_t r6 = vfmsq_f32(vfmsq_f32(safe6, n6, ln2_hi), n6, ln2_lo);
    float32x4_t r7 = vfmsq_f32(vfmsq_f32(safe7, n7, ln2_hi), n7, ln2_lo);

    int32x4_t ni0 = vcvtq_s32_f32(n0);
    int32x4_t ni1 = vcvtq_s32_f32(n1);
    int32x4_t ni2 = vcvtq_s32_f32(n2);
    int32x4_t ni3 = vcvtq_s32_f32(n3);
    int32x4_t ni4 = vcvtq_s32_f32(n4);
    int32x4_t ni5 = vcvtq_s32_f32(n5);
    int32x4_t ni6 = vcvtq_s32_f32(n6);
    int32x4_t ni7 = vcvtq_s32_f32(n7);
    uint32x4_t e0 = vshlq_n_u32(vreinterpretq_u32_s32(ni0), 23);
    uint32x4_t e1 = vshlq_n_u32(vreinterpretq_u32_s32(ni1), 23);
    uint32x4_t e2 = vshlq_n_u32(vreinterpretq_u32_s32(ni2), 23);
    uint32x4_t e3 = vshlq_n_u32(vreinterpretq_u32_s32(ni3), 23);
    uint32x4_t e4 = vshlq_n_u32(vreinterpretq_u32_s32(ni4), 23);
    uint32x4_t e5 = vshlq_n_u32(vreinterpretq_u32_s32(ni5), 23);
    uint32x4_t e6 = vshlq_n_u32(vreinterpretq_u32_s32(ni6), 23);
    uint32x4_t e7 = vshlq_n_u32(vreinterpretq_u32_s32(ni7), 23);
    float32x4_t scale0 = vreinterpretq_f32_u32(vaddq_u32(e0, exponent_bias));
    float32x4_t scale1 = vreinterpretq_f32_u32(vaddq_u32(e1, exponent_bias));
    float32x4_t scale2 = vreinterpretq_f32_u32(vaddq_u32(e2, exponent_bias));
    float32x4_t scale3 = vreinterpretq_f32_u32(vaddq_u32(e3, exponent_bias));
    float32x4_t scale4 = vreinterpretq_f32_u32(vaddq_u32(e4, exponent_bias));
    float32x4_t scale5 = vreinterpretq_f32_u32(vaddq_u32(e5, exponent_bias));
    float32x4_t scale6 = vreinterpretq_f32_u32(vaddq_u32(e6, exponent_bias));
    float32x4_t scale7 = vreinterpretq_f32_u32(vaddq_u32(e7, exponent_bias));

    float32x4_t r20 = vmulq_f32(r0, r0);
    float32x4_t r21 = vmulq_f32(r1, r1);
    float32x4_t r22 = vmulq_f32(r2, r2);
    float32x4_t r23 = vmulq_f32(r3, r3);
    float32x4_t r24 = vmulq_f32(r4, r4);
    float32x4_t r25 = vmulq_f32(r5, r5);
    float32x4_t r26 = vmulq_f32(r6, r6);
    float32x4_t r27 = vmulq_f32(r7, r7);
    float32x4_t p0 = vfmaq_f32(c1, r0, c0);
    float32x4_t p1 = vfmaq_f32(c1, r1, c0);
    float32x4_t p2 = vfmaq_f32(c1, r2, c0);
    float32x4_t p3 = vfmaq_f32(c1, r3, c0);
    float32x4_t p4 = vfmaq_f32(c1, r4, c0);
    float32x4_t p5 = vfmaq_f32(c1, r5, c0);
    float32x4_t p6 = vfmaq_f32(c1, r6, c0);
    float32x4_t p7 = vfmaq_f32(c1, r7, c0);
    float32x4_t q0 = vfmaq_f32(c3, r0, c2);
    float32x4_t q1 = vfmaq_f32(c3, r1, c2);
    float32x4_t q2 = vfmaq_f32(c3, r2, c2);
    float32x4_t q3 = vfmaq_f32(c3, r3, c2);
    float32x4_t q4 = vfmaq_f32(c3, r4, c2);
    float32x4_t q5 = vfmaq_f32(c3, r5, c2);
    float32x4_t q6 = vfmaq_f32(c3, r6, c2);
    float32x4_t q7 = vfmaq_f32(c3, r7, c2);
    q0 = vfmaq_f32(q0, p0, r20);
    q1 = vfmaq_f32(q1, p1, r21);
    q2 = vfmaq_f32(q2, p2, r22);
    q3 = vfmaq_f32(q3, p3, r23);
    q4 = vfmaq_f32(q4, p4, r24);
    q5 = vfmaq_f32(q5, p5, r25);
    q6 = vfmaq_f32(q6, p6, r26);
    q7 = vfmaq_f32(q7, p7, r27);
    float32x4_t poly0 = vfmaq_f32(vmulq_f32(c4, r0), q0, r20);
    float32x4_t poly1 = vfmaq_f32(vmulq_f32(c4, r1), q1, r21);
    float32x4_t poly2 = vfmaq_f32(vmulq_f32(c4, r2), q2, r22);
    float32x4_t poly3 = vfmaq_f32(vmulq_f32(c4, r3), q3, r23);
    float32x4_t poly4 = vfmaq_f32(vmulq_f32(c4, r4), q4, r24);
    float32x4_t poly5 = vfmaq_f32(vmulq_f32(c4, r5), q5, r25);
    float32x4_t poly6 = vfmaq_f32(vmulq_f32(c4, r6), q6, r26);
    float32x4_t poly7 = vfmaq_f32(vmulq_f32(c4, r7), q7, r27);
    float32x4_t result0 = vfmaq_f32(scale0, poly0, scale0);
    float32x4_t result1 = vfmaq_f32(scale1, poly1, scale1);
    float32x4_t result2 = vfmaq_f32(scale2, poly2, scale2);
    float32x4_t result3 = vfmaq_f32(scale3, poly3, scale3);
    float32x4_t result4 = vfmaq_f32(scale4, poly4, scale4);
    float32x4_t result5 = vfmaq_f32(scale5, poly5, scale5);
    float32x4_t result6 = vfmaq_f32(scale6, poly6, scale6);
    float32x4_t result7 = vfmaq_f32(scale7, poly7, scale7);

    uint32x4_t is_overflow0 = vcgtq_f32(x0, exp_max);
    uint32x4_t is_overflow1 = vcgtq_f32(x1, exp_max);
    uint32x4_t is_overflow2 = vcgtq_f32(x2, exp_max);
    uint32x4_t is_overflow3 = vcgtq_f32(x3, exp_max);
    uint32x4_t is_overflow4 = vcgtq_f32(x4, exp_max);
    uint32x4_t is_overflow5 = vcgtq_f32(x5, exp_max);
    uint32x4_t is_overflow6 = vcgtq_f32(x6, exp_max);
    uint32x4_t is_overflow7 = vcgtq_f32(x7, exp_max);
    uint32x4_t is_underflow0 = vcltq_f32(x0, exp_min);
    uint32x4_t is_underflow1 = vcltq_f32(x1, exp_min);
    uint32x4_t is_underflow2 = vcltq_f32(x2, exp_min);
    uint32x4_t is_underflow3 = vcltq_f32(x3, exp_min);
    uint32x4_t is_underflow4 = vcltq_f32(x4, exp_min);
    uint32x4_t is_underflow5 = vcltq_f32(x5, exp_min);
    uint32x4_t is_underflow6 = vcltq_f32(x6, exp_min);
    uint32x4_t is_underflow7 = vcltq_f32(x7, exp_min);

    result0 = vbslq_f32(is_nan0, x0, result0);
    result1 = vbslq_f32(is_nan1, x1, result1);
    result2 = vbslq_f32(is_nan2, x2, result2);
    result3 = vbslq_f32(is_nan3, x3, result3);
    result4 = vbslq_f32(is_nan4, x4, result4);
    result5 = vbslq_f32(is_nan5, x5, result5);
    result6 = vbslq_f32(is_nan6, x6, result6);
    result7 = vbslq_f32(is_nan7, x7, result7);
    result0 = vbslq_f32(vceqq_f32(x0, pos_inf), pos_inf, result0);
    result1 = vbslq_f32(vceqq_f32(x1, pos_inf), pos_inf, result1);
    result2 = vbslq_f32(vceqq_f32(x2, pos_inf), pos_inf, result2);
    result3 = vbslq_f32(vceqq_f32(x3, pos_inf), pos_inf, result3);
    result4 = vbslq_f32(vceqq_f32(x4, pos_inf), pos_inf, result4);
    result5 = vbslq_f32(vceqq_f32(x5, pos_inf), pos_inf, result5);
    result6 = vbslq_f32(vceqq_f32(x6, pos_inf), pos_inf, result6);
    result7 = vbslq_f32(vceqq_f32(x7, pos_inf), pos_inf, result7);
    result0 = vbslq_f32(vceqq_f32(x0, neg_inf), zero_val, result0);
    result1 = vbslq_f32(vceqq_f32(x1, neg_inf), zero_val, result1);
    result2 = vbslq_f32(vceqq_f32(x2, neg_inf), zero_val, result2);
    result3 = vbslq_f32(vceqq_f32(x3, neg_inf), zero_val, result3);
    result4 = vbslq_f32(vceqq_f32(x4, neg_inf), zero_val, result4);
    result5 = vbslq_f32(vceqq_f32(x5, neg_inf), zero_val, result5);
    result6 = vbslq_f32(vceqq_f32(x6, neg_inf), zero_val, result6);
    result7 = vbslq_f32(vceqq_f32(x7, neg_inf), zero_val, result7);
    result0 = vbslq_f32(is_overflow0, pos_inf, result0);
    result1 = vbslq_f32(is_overflow1, pos_inf, result1);
    result2 = vbslq_f32(is_overflow2, pos_inf, result2);
    result3 = vbslq_f32(is_overflow3, pos_inf, result3);
    result4 = vbslq_f32(is_overflow4, pos_inf, result4);
    result5 = vbslq_f32(is_overflow5, pos_inf, result5);
    result6 = vbslq_f32(is_overflow6, pos_inf, result6);
    result7 = vbslq_f32(is_overflow7, pos_inf, result7);
    result0 = vbslq_f32(is_underflow0, zero_val, result0);
    result1 = vbslq_f32(is_underflow1, zero_val, result1);
    result2 = vbslq_f32(is_underflow2, zero_val, result2);
    result3 = vbslq_f32(is_underflow3, zero_val, result3);
    result4 = vbslq_f32(is_underflow4, zero_val, result4);
    result5 = vbslq_f32(is_underflow5, zero_val, result5);
    result6 = vbslq_f32(is_underflow6, zero_val, result6);
    result7 = vbslq_f32(is_underflow7, zero_val, result7);

    if (sdst == 1) {
      __builtin_prefetch(outp + 128, 1, 3);
      vst1q_f32(outp, result0);
      vst1q_f32(outp + 4, result1);
      vst1q_f32(outp + 8, result2);
      vst1q_f32(outp + 12, result3);
      vst1q_f32(outp + 16, result4);
      vst1q_f32(outp + 20, result5);
      vst1q_f32(outp + 24, result6);
      vst1q_f32(outp + 28, result7);
    } else {
      for (int i = 0; i < 4; i++) {
        outp[i * sdst] = vgetq_lane_f32(result0, i);
        outp[(4 + i) * sdst] = vgetq_lane_f32(result1, i);
        outp[(8 + i) * sdst] = vgetq_lane_f32(result2, i);
        outp[(12 + i) * sdst] = vgetq_lane_f32(result3, i);
        outp[(16 + i) * sdst] = vgetq_lane_f32(result4, i);
        outp[(20 + i) * sdst] = vgetq_lane_f32(result5, i);
        outp[(24 + i) * sdst] = vgetq_lane_f32(result6, i);
        outp[(28 + i) * sdst] = vgetq_lane_f32(result7, i);
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
    uint32x4_t special_mask = vorrq_u32(vorrq_u32(is_nan, is_pos_inf), is_neg_inf);
    float32x4_t safe_x = vbslq_f32(special_mask, zero_val, x);

    float32x4_t z = vfmaq_f32(shift, safe_x, inv_ln2);
    float32x4_t n = vsubq_f32(z, shift);
    float32x4_t r = vfmsq_f32(vfmsq_f32(safe_x, n, ln2_hi), n, ln2_lo);

    int32x4_t ni = vcvtq_s32_f32(n);
    uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_s32(ni), 23);
    float32x4_t scale = vreinterpretq_f32_u32(vaddq_u32(e, exponent_bias));

    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t p = vfmaq_f32(c1, r, c0);
    float32x4_t q = vfmaq_f32(c3, r, c2);
    q = vfmaq_f32(q, p, r2);
    float32x4_t poly = vfmaq_f32(vmulq_f32(c4, r), q, r2);
    float32x4_t result = vfmaq_f32(scale, poly, scale);

    result = vbslq_f32(is_nan, nan_val, result);
    result = vbslq_f32(is_pos_inf, pos_inf, result);
    result = vbslq_f32(is_neg_inf, zero_val, result);

    if (sdst == 1) {
      if (current_len == 1) vst1q_lane_f32(outp, result, 0);
      else if (current_len == 2) vst1_f32(outp, vget_low_f32(result));
      else if (current_len == 3) {
        vst1q_lane_f32(outp, result, 0);
        vst1q_lane_f32(outp + 1, result, 1);
        vst1q_lane_f32(outp + 2, result, 2);
      } else vst1q_f32(outp, result);
    } else {
      for (int i = 0; i < current_len; i++) outp[i * sdst] = vgetq_lane_f32(result, i);
    }

    infp += ssrc * current_len;
    outp += sdst * current_len;
    remaining -= current_len;
  }
  feclearexcept(FE_ALL_EXCEPT);
}

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
    float64x2_t x0, x1, x2, x3;
    uint64x2_t u0, u1, u2, u3;

    if (ssrc == 1) {
      x0 = vld1q_f64(infp);
      x1 = vld1q_f64(infp + 2);
      x2 = vld1q_f64(infp + 4);
      x3 = vld1q_f64(infp + 6);
    } else {
      double vals0[2] = {infp[0], infp[ssrc]};
      double vals1[2] = {infp[2 * ssrc], infp[3 * ssrc]};
      double vals2[2] = {infp[4 * ssrc], infp[5 * ssrc]};
      double vals3[2] = {infp[6 * ssrc], infp[7 * ssrc]};
      x0 = vld1q_f64(vals0);
      x1 = vld1q_f64(vals1);
      x2 = vld1q_f64(vals2);
      x3 = vld1q_f64(vals3);
    }

    u0 = vreinterpretq_u64_f64(x0);
    u1 = vreinterpretq_u64_f64(x1);
    u2 = vreinterpretq_u64_f64(x2);
    u3 = vreinterpretq_u64_f64(x3);

    uint64x2_t is_nan0 = vceqq_f64(x0, x0);
    uint64x2_t is_nan1 = vceqq_f64(x1, x1);
    uint64x2_t is_nan2 = vceqq_f64(x2, x2);
    uint64x2_t is_nan3 = vceqq_f64(x3, x3);
    is_nan0 = veorq_u64(is_nan0, all_ones);
    is_nan1 = veorq_u64(is_nan1, all_ones);
    is_nan2 = veorq_u64(is_nan2, all_ones);
    is_nan3 = veorq_u64(is_nan3, all_ones);

    uint64x2_t is_pos_inf0 = vceqq_f64(x0, pos_inf);
    uint64x2_t is_pos_inf1 = vceqq_f64(x1, pos_inf);
    uint64x2_t is_pos_inf2 = vceqq_f64(x2, pos_inf);
    uint64x2_t is_pos_inf3 = vceqq_f64(x3, pos_inf);
    uint64x2_t is_neg_inf0 = vceqq_f64(x0, neg_inf_val);
    uint64x2_t is_neg_inf1 = vceqq_f64(x1, neg_inf_val);
    uint64x2_t is_neg_inf2 = vceqq_f64(x2, neg_inf_val);
    uint64x2_t is_neg_inf3 = vceqq_f64(x3, neg_inf_val);

    uint64x2_t is_zero0 = vceqq_f64(x0, zeros_f);
    uint64x2_t is_zero1 = vceqq_f64(x1, zeros_f);
    uint64x2_t is_zero2 = vceqq_f64(x2, zeros_f);
    uint64x2_t is_zero3 = vceqq_f64(x3, zeros_f);

    uint64x2_t is_neg0 = vcltq_f64(x0, zeros_f);
    uint64x2_t is_neg1 = vcltq_f64(x1, zeros_f);
    uint64x2_t is_neg2 = vcltq_f64(x2, zeros_f);
    uint64x2_t is_neg3 = vcltq_f64(x3, zeros_f);
    is_neg0 = vbicq_u64(is_neg0, vorrq_u64(is_neg_inf0, is_zero0));
    is_neg1 = vbicq_u64(is_neg1, vorrq_u64(is_neg_inf1, is_zero1));
    is_neg2 = vbicq_u64(is_neg2, vorrq_u64(is_neg_inf2, is_zero2));
    is_neg3 = vbicq_u64(is_neg3, vorrq_u64(is_neg_inf3, is_zero3));

    uint64x2_t special_mask0 = vorrq_u64(vorrq_u64(is_nan0, is_pos_inf0), vorrq_u64(is_neg_inf0, is_zero0));
    uint64x2_t special_mask1 = vorrq_u64(vorrq_u64(is_nan1, is_pos_inf1), vorrq_u64(is_neg_inf1, is_zero1));
    uint64x2_t special_mask2 = vorrq_u64(vorrq_u64(is_nan2, is_pos_inf2), vorrq_u64(is_neg_inf2, is_zero2));
    uint64x2_t special_mask3 = vorrq_u64(vorrq_u64(is_nan3, is_pos_inf3), vorrq_u64(is_neg_inf3, is_zero3));
    uint64x2_t full_mask0 = vorrq_u64(special_mask0, is_neg0);
    uint64x2_t full_mask1 = vorrq_u64(special_mask1, is_neg1);
    uint64x2_t full_mask2 = vorrq_u64(special_mask2, is_neg2);
    uint64x2_t full_mask3 = vorrq_u64(special_mask3, is_neg3);
    float64x2_t safe0 = vbslq_f64(full_mask0, ones_f, x0);
    float64x2_t safe1 = vbslq_f64(full_mask1, ones_f, x1);
    float64x2_t safe2 = vbslq_f64(full_mask2, ones_f, x2);
    float64x2_t safe3 = vbslq_f64(full_mask3, ones_f, x3);
    u0 = vreinterpretq_u64_f64(safe0);
    u1 = vreinterpretq_u64_f64(safe1);
    u2 = vreinterpretq_u64_f64(safe2);
    u3 = vreinterpretq_u64_f64(safe3);

    uint64x2_t u_off0 = vsubq_u64(u0, off);
    uint64x2_t u_off1 = vsubq_u64(u1, off);
    uint64x2_t u_off2 = vsubq_u64(u2, off);
    uint64x2_t u_off3 = vsubq_u64(u3, off);
    int64x2_t k0 = vshrq_n_s64(vreinterpretq_s64_u64(u_off0), 52);
    int64x2_t k1 = vshrq_n_s64(vreinterpretq_s64_u64(u_off1), 52);
    int64x2_t k2 = vshrq_n_s64(vreinterpretq_s64_u64(u_off2), 52);
    int64x2_t k3 = vshrq_n_s64(vreinterpretq_s64_u64(u_off3), 52);
    uint64x2_t iz0 = vsubq_u64(u0, vandq_u64(u_off0, sign_exp_mask));
    uint64x2_t iz1 = vsubq_u64(u1, vandq_u64(u_off1, sign_exp_mask));
    uint64x2_t iz2 = vsubq_u64(u2, vandq_u64(u_off2, sign_exp_mask));
    uint64x2_t iz3 = vsubq_u64(u3, vandq_u64(u_off3, sign_exp_mask));
    float64x2_t z0 = vreinterpretq_f64_u64(iz0);
    float64x2_t z1 = vreinterpretq_f64_u64(iz1);
    float64x2_t z2 = vreinterpretq_f64_u64(iz2);
    float64x2_t z3 = vreinterpretq_f64_u64(iz3);

    uint64_t i00 = (vgetq_lane_u64(u_off0, 0) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;
    uint64_t i01 = (vgetq_lane_u64(u_off0, 1) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;
    uint64_t i10 = (vgetq_lane_u64(u_off1, 0) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;
    uint64_t i11 = (vgetq_lane_u64(u_off1, 1) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;
    uint64_t i20 = (vgetq_lane_u64(u_off2, 0) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;
    uint64_t i21 = (vgetq_lane_u64(u_off2, 1) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;
    uint64_t i30 = (vgetq_lane_u64(u_off3, 0) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;
    uint64_t i31 = (vgetq_lane_u64(u_off3, 1) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;

    float64x2_t invc0 = vcombine_f64(vdup_n_f64(neon_log_table[i00].invc), vdup_n_f64(neon_log_table[i01].invc));
    float64x2_t invc1 = vcombine_f64(vdup_n_f64(neon_log_table[i10].invc), vdup_n_f64(neon_log_table[i11].invc));
    float64x2_t invc2 = vcombine_f64(vdup_n_f64(neon_log_table[i20].invc), vdup_n_f64(neon_log_table[i21].invc));
    float64x2_t invc3 = vcombine_f64(vdup_n_f64(neon_log_table[i30].invc), vdup_n_f64(neon_log_table[i31].invc));
    float64x2_t logc0 = vcombine_f64(vdup_n_f64(neon_log_table[i00].logc), vdup_n_f64(neon_log_table[i01].logc));
    float64x2_t logc1 = vcombine_f64(vdup_n_f64(neon_log_table[i10].logc), vdup_n_f64(neon_log_table[i11].logc));
    float64x2_t logc2 = vcombine_f64(vdup_n_f64(neon_log_table[i20].logc), vdup_n_f64(neon_log_table[i21].logc));
    float64x2_t logc3 = vcombine_f64(vdup_n_f64(neon_log_table[i30].logc), vdup_n_f64(neon_log_table[i31].logc));

    float64x2_t r0 = vfmaq_f64(neg_one, z0, invc0);
    float64x2_t r1 = vfmaq_f64(neg_one, z1, invc1);
    float64x2_t r2 = vfmaq_f64(neg_one, z2, invc2);
    float64x2_t r3 = vfmaq_f64(neg_one, z3, invc3);
    float64x2_t kd0 = vcvtq_f64_s64(k0);
    float64x2_t kd1 = vcvtq_f64_s64(k1);
    float64x2_t kd2 = vcvtq_f64_s64(k2);
    float64x2_t kd3 = vcvtq_f64_s64(k3);

    float64x2_t hi0 = vfmaq_f64(vaddq_f64(logc0, r0), kd0, ln2);
    float64x2_t hi1 = vfmaq_f64(vaddq_f64(logc1, r1), kd1, ln2);
    float64x2_t hi2 = vfmaq_f64(vaddq_f64(logc2, r2), kd2, ln2);
    float64x2_t hi3 = vfmaq_f64(vaddq_f64(logc3, r3), kd3, ln2);
    float64x2_t r20 = vmulq_f64(r0, r0);
    float64x2_t r21 = vmulq_f64(r1, r1);
    float64x2_t r22 = vmulq_f64(r2, r2);
    float64x2_t r23 = vmulq_f64(r3, r3);
    float64x2_t y0 = vfmaq_f64(c2, r0, c3);
    float64x2_t y1 = vfmaq_f64(c2, r1, c3);
    float64x2_t y2 = vfmaq_f64(c2, r2, c3);
    float64x2_t y3 = vfmaq_f64(c2, r3, c3);
    float64x2_t p0 = vfmaq_f64(c0, r0, c1);
    float64x2_t p1 = vfmaq_f64(c0, r1, c1);
    float64x2_t p2 = vfmaq_f64(c0, r2, c1);
    float64x2_t p3 = vfmaq_f64(c0, r3, c1);
    y0 = vfmaq_f64(y0, r20, c4);
    y1 = vfmaq_f64(y1, r21, c4);
    y2 = vfmaq_f64(y2, r22, c4);
    y3 = vfmaq_f64(y3, r23, c4);
    y0 = vfmaq_f64(p0, r20, y0);
    y1 = vfmaq_f64(p1, r21, y1);
    y2 = vfmaq_f64(p2, r22, y2);
    y3 = vfmaq_f64(p3, r23, y3);

    float64x2_t result0 = vfmaq_f64(hi0, y0, r20);
    float64x2_t result1 = vfmaq_f64(hi1, y1, r21);
    float64x2_t result2 = vfmaq_f64(hi2, y2, r22);
    float64x2_t result3 = vfmaq_f64(hi3, y3, r23);

    result0 = vbslq_f64(is_nan0, x0, result0);
    result1 = vbslq_f64(is_nan1, x1, result1);
    result2 = vbslq_f64(is_nan2, x2, result2);
    result3 = vbslq_f64(is_nan3, x3, result3);
    result0 = vbslq_f64(is_pos_inf0, pos_inf, result0);
    result1 = vbslq_f64(is_pos_inf1, pos_inf, result1);
    result2 = vbslq_f64(is_pos_inf2, pos_inf, result2);
    result3 = vbslq_f64(is_pos_inf3, pos_inf, result3);
    result0 = vbslq_f64(is_neg_inf0, nan_val, result0);
    result1 = vbslq_f64(is_neg_inf1, nan_val, result1);
    result2 = vbslq_f64(is_neg_inf2, nan_val, result2);
    result3 = vbslq_f64(is_neg_inf3, nan_val, result3);
    result0 = vbslq_f64(is_zero0, neg_inf_val, result0);
    result1 = vbslq_f64(is_zero1, neg_inf_val, result1);
    result2 = vbslq_f64(is_zero2, neg_inf_val, result2);
    result3 = vbslq_f64(is_zero3, neg_inf_val, result3);
    result0 = vbslq_f64(is_neg0, neg_nan, result0);
    result1 = vbslq_f64(is_neg1, neg_nan, result1);
    result2 = vbslq_f64(is_neg2, neg_nan, result2);
    result3 = vbslq_f64(is_neg3, neg_nan, result3);

    if (sdst == 1) {
      vst1q_f64(outp, result0);
      vst1q_f64(outp + 2, result1);
      vst1q_f64(outp + 4, result2);
      vst1q_f64(outp + 6, result3);
    } else {
      for (int i = 0; i < 2; i++) {
        outp[i * sdst] = vgetq_lane_f64(result0, i);
        outp[(2 + i) * sdst] = vgetq_lane_f64(result1, i);
        outp[(4 + i) * sdst] = vgetq_lane_f64(result2, i);
        outp[(6 + i) * sdst] = vgetq_lane_f64(result3, i);
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
      for (int i = 0; i < current_len; i++) {
        double val = (ssrc == 1) ? infp[i] : infp[i * ssrc];
        uint64_t lane_nan = vgetq_lane_u64(is_nan, i);
        uint64_t lane_pinf = vgetq_lane_u64(is_pos_inf, i);
        uint64_t lane_ninf = vgetq_lane_u64(is_neg_inf, i);
        uint64_t lane_neg = vgetq_lane_u64(is_neg, i);
        uint64_t lane_zero = vgetq_lane_u64(is_zero, i);
        if (lane_nan) {
          if (sdst == 1) outp[i] = val;
          else outp[i * sdst] = val;
        } else if (lane_pinf) {
          if (sdst == 1) outp[i] = NPY_INFINITY;
          else outp[i * sdst] = NPY_INFINITY;
        } else if (lane_ninf) {
          if (sdst == 1) outp[i] = NPY_NAN;
          else outp[i * sdst] = NPY_NAN;
        } else if (lane_zero) {
          if (sdst == 1) outp[i] = -NPY_INFINITY;
          else outp[i * sdst] = -NPY_INFINITY;
        } else if (lane_neg) {
          if (sdst == 1) outp[i] = -NPY_NAN;
          else outp[i * sdst] = -NPY_NAN;
        } else {
          double result = npy_log(val);
          if (sdst == 1) outp[i] = result;
          else outp[i * sdst] = result;
        }
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
  const float32x4_t zeros_f = vdupq_n_f32(0.0f);
  const float32x4_t ones_f = vdupq_n_f32(1.0f);
  const float32x4_t pos_inf = vdupq_n_f32(NPY_INFINITYF);
  const float32x4_t neg_inf_val = vdupq_n_f32(-NPY_INFINITYF);
  const float32x4_t nan_val = vdupq_n_f32(NPY_NANF);

  const npy_float *infp = src;
  npy_float *outp = dst;
  npy_intp remaining = len;

  feclearexcept(FE_ALL_EXCEPT);

  const npy_intp UNROLL = 4;
  const npy_intp VEC_SIZE = 4;

  while (remaining >= UNROLL * VEC_SIZE) {
    float32x4_t x0, x1, x2, x3;
    uint32x4_t u0, u1, u2, u3;

    if (ssrc == 1) {
      x0 = vld1q_f32(infp);
      x1 = vld1q_f32(infp + 4);
      x2 = vld1q_f32(infp + 8);
      x3 = vld1q_f32(infp + 12);
    } else {
      float vals0[4], vals1[4], vals2[4], vals3[4];
      for (int i = 0; i < 4; i++) {
        vals0[i] = infp[i * ssrc];
        vals1[i] = infp[(4 + i) * ssrc];
        vals2[i] = infp[(8 + i) * ssrc];
        vals3[i] = infp[(12 + i) * ssrc];
      }
      x0 = vld1q_f32(vals0);
      x1 = vld1q_f32(vals1);
      x2 = vld1q_f32(vals2);
      x3 = vld1q_f32(vals3);
    }

    u0 = vreinterpretq_u32_f32(x0);
    u1 = vreinterpretq_u32_f32(x1);
    u2 = vreinterpretq_u32_f32(x2);
    u3 = vreinterpretq_u32_f32(x3);

    uint32x4_t is_nan0 = vceqq_f32(x0, x0);
    uint32x4_t is_nan1 = vceqq_f32(x1, x1);
    uint32x4_t is_nan2 = vceqq_f32(x2, x2);
    uint32x4_t is_nan3 = vceqq_f32(x3, x3);
    is_nan0 = vmvnq_u32(is_nan0);
    is_nan1 = vmvnq_u32(is_nan1);
    is_nan2 = vmvnq_u32(is_nan2);
    is_nan3 = vmvnq_u32(is_nan3);

    uint32x4_t is_pos_inf0 = vceqq_f32(x0, pos_inf);
    uint32x4_t is_pos_inf1 = vceqq_f32(x1, pos_inf);
    uint32x4_t is_pos_inf2 = vceqq_f32(x2, pos_inf);
    uint32x4_t is_pos_inf3 = vceqq_f32(x3, pos_inf);
    uint32x4_t is_neg_inf0 = vceqq_f32(x0, neg_inf_val);
    uint32x4_t is_neg_inf1 = vceqq_f32(x1, neg_inf_val);
    uint32x4_t is_neg_inf2 = vceqq_f32(x2, neg_inf_val);
    uint32x4_t is_neg_inf3 = vceqq_f32(x3, neg_inf_val);

    uint32x4_t is_zero0 = vceqq_f32(x0, zeros_f);
    uint32x4_t is_zero1 = vceqq_f32(x1, zeros_f);
    uint32x4_t is_zero2 = vceqq_f32(x2, zeros_f);
    uint32x4_t is_zero3 = vceqq_f32(x3, zeros_f);

    uint32x4_t is_neg0 = vcltq_f32(x0, zeros_f);
    uint32x4_t is_neg1 = vcltq_f32(x1, zeros_f);
    uint32x4_t is_neg2 = vcltq_f32(x2, zeros_f);
    uint32x4_t is_neg3 = vcltq_f32(x3, zeros_f);
    is_neg0 = vbicq_u32(is_neg0, vorrq_u32(is_neg_inf0, is_zero0));
    is_neg1 = vbicq_u32(is_neg1, vorrq_u32(is_neg_inf1, is_zero1));
    is_neg2 = vbicq_u32(is_neg2, vorrq_u32(is_neg_inf2, is_zero2));
    is_neg3 = vbicq_u32(is_neg3, vorrq_u32(is_neg_inf3, is_zero3));

    uint32x4_t special_mask0 = vorrq_u32(vorrq_u32(is_nan0, is_pos_inf0), vorrq_u32(is_neg_inf0, is_zero0));
    uint32x4_t special_mask1 = vorrq_u32(vorrq_u32(is_nan1, is_pos_inf1), vorrq_u32(is_neg_inf1, is_zero1));
    uint32x4_t special_mask2 = vorrq_u32(vorrq_u32(is_nan2, is_pos_inf2), vorrq_u32(is_neg_inf2, is_zero2));
    uint32x4_t special_mask3 = vorrq_u32(vorrq_u32(is_nan3, is_pos_inf3), vorrq_u32(is_neg_inf3, is_zero3));
    uint32x4_t full_mask0 = vorrq_u32(special_mask0, is_neg0);
    uint32x4_t full_mask1 = vorrq_u32(special_mask1, is_neg1);
    uint32x4_t full_mask2 = vorrq_u32(special_mask2, is_neg2);
    uint32x4_t full_mask3 = vorrq_u32(special_mask3, is_neg3);
    float32x4_t safe0 = vbslq_f32(full_mask0, ones_f, x0);
    float32x4_t safe1 = vbslq_f32(full_mask1, ones_f, x1);
    float32x4_t safe2 = vbslq_f32(full_mask2, ones_f, x2);
    float32x4_t safe3 = vbslq_f32(full_mask3, ones_f, x3);
    u0 = vreinterpretq_u32_f32(safe0);
    u1 = vreinterpretq_u32_f32(safe1);
    u2 = vreinterpretq_u32_f32(safe2);
    u3 = vreinterpretq_u32_f32(safe3);

    uint32x4_t u_off0 = vsubq_u32(u0, off);
    uint32x4_t u_off1 = vsubq_u32(u1, off);
    uint32x4_t u_off2 = vsubq_u32(u2, off);
    uint32x4_t u_off3 = vsubq_u32(u3, off);
    float32x4_t n0 = vcvtq_f32_s32(vshrq_n_s32(vreinterpretq_s32_u32(u_off0), 23));
    float32x4_t n1 = vcvtq_f32_s32(vshrq_n_s32(vreinterpretq_s32_u32(u_off1), 23));
    float32x4_t n2 = vcvtq_f32_s32(vshrq_n_s32(vreinterpretq_s32_u32(u_off2), 23));
    float32x4_t n3 = vcvtq_f32_s32(vshrq_n_s32(vreinterpretq_s32_u32(u_off3), 23));

    uint32x4_t um0 = vaddq_u32(vandq_u32(u_off0, mantissa_mask), off);
    uint32x4_t um1 = vaddq_u32(vandq_u32(u_off1, mantissa_mask), off);
    uint32x4_t um2 = vaddq_u32(vandq_u32(u_off2, mantissa_mask), off);
    uint32x4_t um3 = vaddq_u32(vandq_u32(u_off3, mantissa_mask), off);
    float32x4_t r0 = vsubq_f32(vreinterpretq_f32_u32(um0), ones_f);
    float32x4_t r1 = vsubq_f32(vreinterpretq_f32_u32(um1), ones_f);
    float32x4_t r2 = vsubq_f32(vreinterpretq_f32_u32(um2), ones_f);
    float32x4_t r3 = vsubq_f32(vreinterpretq_f32_u32(um3), ones_f);

    float32x4_t r20 = vmulq_f32(r0, r0);
    float32x4_t r21 = vmulq_f32(r1, r1);
    float32x4_t r22 = vmulq_f32(r2, r2);
    float32x4_t r23 = vmulq_f32(r3, r3);

    float32x4_t p0 = vfmaq_f32(c2, r0, c1);
    float32x4_t p1 = vfmaq_f32(c2, r1, c1);
    float32x4_t p2 = vfmaq_f32(c2, r2, c1);
    float32x4_t p3 = vfmaq_f32(c2, r3, c1);
    float32x4_t q0 = vfmaq_f32(c4, r0, c3);
    float32x4_t q1 = vfmaq_f32(c4, r1, c3);
    float32x4_t q2 = vfmaq_f32(c4, r2, c3);
    float32x4_t q3 = vfmaq_f32(c4, r3, c3);
    float32x4_t y0 = vfmaq_f32(c6, r0, c5);
    float32x4_t y1 = vfmaq_f32(c6, r1, c5);
    float32x4_t y2 = vfmaq_f32(c6, r2, c5);
    float32x4_t y3 = vfmaq_f32(c6, r3, c5);
    p0 = vfmaq_f32(p0, r20, c0);
    p1 = vfmaq_f32(p1, r21, c0);
    p2 = vfmaq_f32(p2, r22, c0);
    p3 = vfmaq_f32(p3, r23, c0);

    q0 = vfmaq_f32(q0, p0, r20);
    q1 = vfmaq_f32(q1, p1, r21);
    q2 = vfmaq_f32(q2, p2, r22);
    q3 = vfmaq_f32(q3, p3, r23);
    y0 = vfmaq_f32(y0, q0, r20);
    y1 = vfmaq_f32(y1, q1, r21);
    y2 = vfmaq_f32(y2, q2, r22);
    y3 = vfmaq_f32(y3, q3, r23);
    float32x4_t base0 = vfmaq_f32(r0, n0, ln2);
    float32x4_t base1 = vfmaq_f32(r1, n1, ln2);
    float32x4_t base2 = vfmaq_f32(r2, n2, ln2);
    float32x4_t base3 = vfmaq_f32(r3, n3, ln2);

    float32x4_t result0 = vfmaq_f32(base0, y0, r20);
    float32x4_t result1 = vfmaq_f32(base1, y1, r21);
    float32x4_t result2 = vfmaq_f32(base2, y2, r22);
    float32x4_t result3 = vfmaq_f32(base3, y3, r23);

    result0 = vbslq_f32(is_nan0, x0, result0);
    result1 = vbslq_f32(is_nan1, x1, result1);
    result2 = vbslq_f32(is_nan2, x2, result2);
    result3 = vbslq_f32(is_nan3, x3, result3);
    result0 = vbslq_f32(is_pos_inf0, pos_inf, result0);
    result1 = vbslq_f32(is_pos_inf1, pos_inf, result1);
    result2 = vbslq_f32(is_pos_inf2, pos_inf, result2);
    result3 = vbslq_f32(is_pos_inf3, pos_inf, result3);
    result0 = vbslq_f32(is_neg_inf0, nan_val, result0);
    result1 = vbslq_f32(is_neg_inf1, nan_val, result1);
    result2 = vbslq_f32(is_neg_inf2, nan_val, result2);
    result3 = vbslq_f32(is_neg_inf3, nan_val, result3);
    result0 = vbslq_f32(is_zero0, neg_inf_val, result0);
    result1 = vbslq_f32(is_zero1, neg_inf_val, result1);
    result2 = vbslq_f32(is_zero2, neg_inf_val, result2);
    result3 = vbslq_f32(is_zero3, neg_inf_val, result3);
    result0 = vbslq_f32(is_neg0, nan_val, result0);
    result1 = vbslq_f32(is_neg1, nan_val, result1);
    result2 = vbslq_f32(is_neg2, nan_val, result2);
    result3 = vbslq_f32(is_neg3, nan_val, result3);

    if (sdst == 1) {
      vst1q_f32(outp, result0);
      vst1q_f32(outp + 4, result1);
      vst1q_f32(outp + 8, result2);
      vst1q_f32(outp + 12, result3);
    } else {
      for (int i = 0; i < 4; i++) {
        outp[i * sdst] = vgetq_lane_f32(result0, i);
        outp[(4 + i) * sdst] = vgetq_lane_f32(result1, i);
        outp[(8 + i) * sdst] = vgetq_lane_f32(result2, i);
        outp[(12 + i) * sdst] = vgetq_lane_f32(result3, i);
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
        if (current_len == 1) vst1q_lane_f32(outp, result, 0);
        else if (current_len == 2) vst1_f32(outp, vget_low_f32(result));
        else if (current_len == 3) {
          vst1q_lane_f32(outp, result, 0);
          vst1q_lane_f32(outp + 1, result, 1);
          vst1q_lane_f32(outp + 2, result, 2);
        } else vst1q_f32(outp, result);
      } else {
        for (int i = 0; i < current_len; i++) outp[i * sdst] = vgetq_lane_f32(result, i);
      }
    } else {
      feclearexcept(FE_ALL_EXCEPT);
      if (current_len == 1) {
        float val = (ssrc == 1) ? infp[0] : infp[ssrc];
        uint32_t lane_nan = vgetq_lane_u32(is_nan, 0);
        uint32_t lane_pinf = vgetq_lane_u32(is_pos_inf, 0);
        uint32_t lane_ninf = vgetq_lane_u32(is_neg_inf, 0);
        uint32_t lane_neg = vgetq_lane_u32(is_neg, 0);
        uint32_t lane_zero = vgetq_lane_u32(is_zero, 0);
        if (lane_nan) {
          if (sdst == 1) outp[0] = val;
          else outp[sdst] = val;
        } else if (lane_pinf) {
          if (sdst == 1) outp[0] = NPY_INFINITYF;
          else outp[sdst] = NPY_INFINITYF;
        } else if (lane_ninf) {
          if (sdst == 1) outp[0] = NPY_NANF;
          else outp[sdst] = NPY_NANF;
        } else if (lane_zero) {
          if (sdst == 1) outp[0] = -NPY_INFINITYF;
          else outp[sdst] = -NPY_INFINITYF;
        } else if (lane_neg) {
          if (sdst == 1) outp[0] = NPY_NANF;
          else outp[sdst] = NPY_NANF;
        } else {
          float result = npy_logf(val);
          if (sdst == 1) outp[0] = result;
          else outp[sdst] = result;
        }
      } else if (current_len == 2) {
        for (int k = 0; k < 2; k++) {
          float val = (ssrc == 1) ? infp[k] : infp[k * ssrc];
          uint32_t lane_nan = (k == 0) ? vgetq_lane_u32(is_nan, 0) : vgetq_lane_u32(is_nan, 1);
          uint32_t lane_pinf = (k == 0) ? vgetq_lane_u32(is_pos_inf, 0) : vgetq_lane_u32(is_pos_inf, 1);
          uint32_t lane_ninf = (k == 0) ? vgetq_lane_u32(is_neg_inf, 0) : vgetq_lane_u32(is_neg_inf, 1);
          uint32_t lane_neg = (k == 0) ? vgetq_lane_u32(is_neg, 0) : vgetq_lane_u32(is_neg, 1);
          uint32_t lane_zero = (k == 0) ? vgetq_lane_u32(is_zero, 0) : vgetq_lane_u32(is_zero, 1);
          if (lane_nan) {
            if (sdst == 1) outp[k] = val;
            else outp[k * sdst] = val;
          } else if (lane_pinf) {
            if (sdst == 1) outp[k] = NPY_INFINITYF;
            else outp[k * sdst] = NPY_INFINITYF;
          } else if (lane_ninf) {
            if (sdst == 1) outp[k] = NPY_NANF;
            else outp[k * sdst] = NPY_NANF;
          } else if (lane_zero) {
            if (sdst == 1) outp[k] = -NPY_INFINITYF;
            else outp[k * sdst] = -NPY_INFINITYF;
          } else if (lane_neg) {
            if (sdst == 1) outp[k] = NPY_NANF;
            else outp[k * sdst] = NPY_NANF;
          } else {
            float result = npy_logf(val);
            if (sdst == 1) outp[k] = result;
            else outp[k * sdst] = result;
          }
        }
      } else if (current_len == 3) {
        for (int k = 0; k < 3; k++) {
          float val = (ssrc == 1) ? infp[k] : infp[k * ssrc];
          uint32_t lane_nan = (k == 0) ? vgetq_lane_u32(is_nan, 0) : (k == 1) ? vgetq_lane_u32(is_nan, 1) : vgetq_lane_u32(is_nan, 2);
          uint32_t lane_pinf = (k == 0) ? vgetq_lane_u32(is_pos_inf, 0) : (k == 1) ? vgetq_lane_u32(is_pos_inf, 1) : vgetq_lane_u32(is_pos_inf, 2);
          uint32_t lane_ninf = (k == 0) ? vgetq_lane_u32(is_neg_inf, 0) : (k == 1) ? vgetq_lane_u32(is_neg_inf, 1) : vgetq_lane_u32(is_neg_inf, 2);
          uint32_t lane_neg = (k == 0) ? vgetq_lane_u32(is_neg, 0) : (k == 1) ? vgetq_lane_u32(is_neg, 1) : vgetq_lane_u32(is_neg, 2);
          uint32_t lane_zero = (k == 0) ? vgetq_lane_u32(is_zero, 0) : (k == 1) ? vgetq_lane_u32(is_zero, 1) : vgetq_lane_u32(is_zero, 2);
          if (lane_nan) {
            if (sdst == 1) outp[k] = val;
            else outp[k * sdst] = val;
          } else if (lane_pinf) {
            if (sdst == 1) outp[k] = NPY_INFINITYF;
            else outp[k * sdst] = NPY_INFINITYF;
          } else if (lane_ninf) {
            if (sdst == 1) outp[k] = NPY_NANF;
            else outp[k * sdst] = NPY_NANF;
          } else if (lane_zero) {
            if (sdst == 1) outp[k] = -NPY_INFINITYF;
            else outp[k * sdst] = -NPY_INFINITYF;
          } else if (lane_neg) {
            if (sdst == 1) outp[k] = NPY_NANF;
            else outp[k * sdst] = NPY_NANF;
          } else {
            float result = npy_logf(val);
            if (sdst == 1) outp[k] = result;
            else outp[k * sdst] = result;
          }
        }
      } else {
        for (int k = 0; k < 4; k++) {
          float val = (ssrc == 1) ? infp[k] : infp[k * ssrc];
          uint32_t lane_nan = (k == 0) ? vgetq_lane_u32(is_nan, 0) : (k == 1) ? vgetq_lane_u32(is_nan, 1) : (k == 2) ? vgetq_lane_u32(is_nan, 2) : vgetq_lane_u32(is_nan, 3);
          uint32_t lane_pinf = (k == 0) ? vgetq_lane_u32(is_pos_inf, 0) : (k == 1) ? vgetq_lane_u32(is_pos_inf, 1) : (k == 2) ? vgetq_lane_u32(is_pos_inf, 2) : vgetq_lane_u32(is_pos_inf, 3);
          uint32_t lane_ninf = (k == 0) ? vgetq_lane_u32(is_neg_inf, 0) : (k == 1) ? vgetq_lane_u32(is_neg_inf, 1) : (k == 2) ? vgetq_lane_u32(is_neg_inf, 2) : vgetq_lane_u32(is_neg_inf, 3);
          uint32_t lane_neg = (k == 0) ? vgetq_lane_u32(is_neg, 0) : (k == 1) ? vgetq_lane_u32(is_neg, 1) : (k == 2) ? vgetq_lane_u32(is_neg, 2) : vgetq_lane_u32(is_neg, 3);
          uint32_t lane_zero = (k == 0) ? vgetq_lane_u32(is_zero, 0) : (k == 1) ? vgetq_lane_u32(is_zero, 1) : (k == 2) ? vgetq_lane_u32(is_zero, 2) : vgetq_lane_u32(is_zero, 3);
          if (lane_nan) {
            if (sdst == 1) outp[k] = val;
            else outp[k * sdst] = val;
          } else if (lane_pinf) {
            if (sdst == 1) outp[k] = NPY_INFINITYF;
            else outp[k * sdst] = NPY_INFINITYF;
          } else if (lane_ninf) {
            if (sdst == 1) outp[k] = NPY_NANF;
            else outp[k * sdst] = NPY_NANF;
          } else if (lane_zero) {
            if (sdst == 1) outp[k] = -NPY_INFINITYF;
            else outp[k * sdst] = -NPY_INFINITYF;
          } else if (lane_neg) {
            if (sdst == 1) outp[k] = NPY_NANF;
            else outp[k * sdst] = NPY_NANF;
          } else {
            float result = npy_logf(val);
            if (sdst == 1) outp[k] = result;
            else outp[k * sdst] = result;
          }
        }
      }
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

#endif // SIMD_ARM_NEON

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

static void avx512_exp_f16(const npy_half *src, npy_half *dst, npy_intp len)
{
    avx512_exponent_f16(src, dst, len, __svml_expf16, 0);
}

static void avx512_log_f16(const npy_half *src, npy_half *dst, npy_intp len)
{
    avx512_exponent_f16(src, dst, len, __svml_logf16, 0x3c00);
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

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_log)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
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