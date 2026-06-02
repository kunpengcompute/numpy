#ifndef _NPY_UMATH_LOOPS_EXPLOG_H_
#define _NPY_UMATH_LOOPS_EXPLOG_H_

#include <float.h>
#include <fenv.h>

#include "numpy/npy_math.h"
#include "simd/simd.h"
#include "loops_utils.h"
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

NPY_FINLINE __m256
fma_get_partial_load_mask_ps(const npy_int num_elem, const npy_int num_lanes)
{
    float maskint[16] = {-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,
                            1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    float* addr = maskint + num_lanes - num_elem;
    return _mm256_loadu_ps(addr);
}

NPY_FINLINE __m256
fma_masked_gather_ps(__m256 src,
                     npy_float* addr,
                     __m256i vindex,
                     __m256 mask)
{
    return _mm256_mask_i32gather_ps(src, addr, vindex, mask, 4);
}

NPY_FINLINE __m256
fma_masked_load_ps(__m256 mask, npy_float* addr)
{
    return _mm256_maskload_ps(addr, _mm256_cvtps_epi32(mask));
}

NPY_FINLINE __m256
fma_set_masked_lanes_ps(__m256 x, __m256 val, __m256 mask)
{
    return _mm256_blendv_ps(x, val, mask);
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

#endif // SIMD_AVX2_FMA3

#ifdef SIMD_AVX512F

NPY_FINLINE __mmask16
avx512_get_full_load_mask_ps(void)
{
    return 0xFFFF;
}

NPY_FINLINE __mmask16
avx512_get_partial_load_mask_ps(const npy_int num_elem, const npy_int total_elem)
{
    return (0x0001 << num_elem) - 0x0001;
}

NPY_FINLINE __m512
avx512_masked_gather_ps(__m512 src,
                        npy_float* addr,
                        __m512i vindex,
                        __mmask16 kmask)
{
    return _mm512_mask_i32gather_ps(src, kmask, vindex, addr, 4);
}

NPY_FINLINE __m512
avx512_masked_load_ps(__mmask16 mask, npy_float* addr)
{
    return _mm512_maskz_loadu_ps(mask, (__m512 *)addr);
}

NPY_FINLINE __m512
avx512_set_masked_lanes_ps(__m512 x, __m512 val, __mmask16 mask)
{
    return _mm512_mask_blend_ps(mask, x, val);
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

#endif // SIMD_AVX512F

#ifdef SIMD_AVX2_FMA3

NPY_FINLINE __m256i
fma_get_full_load_mask_pd(void)
{
    return _mm256_castpd_si256(_mm256_set1_pd(-1.0));
}

NPY_FINLINE __m256i
fma_get_partial_load_mask_pd(const npy_int num_elem, const npy_int num_lanes)
{
    npy_int maskint[16] = {-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1};
    npy_int* addr = maskint + 2*num_lanes - 2*num_elem;
    return _mm256_loadu_si256((__m256i*) addr);
}

NPY_FINLINE __m256d
fma_masked_gather_pd(__m256d src,
                     npy_double* addr,
                     __m128i vindex,
                     __m256d mask)
{
    return _mm256_mask_i32gather_pd(src, addr, vindex, mask, 8);
}

NPY_FINLINE __m256d
fma_masked_load_pd(__m256i mask, npy_double* addr)
{
    return _mm256_maskload_pd(addr, mask);
}

NPY_FINLINE __m256d
fma_set_masked_lanes_pd(__m256d x, __m256d val, __m256d mask)
{
    return _mm256_blendv_pd(x, val, mask);
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

NPY_FINLINE __m256
simd_range_reduction(__m256 x, __m256 y, __m256 c1, __m256 c2, __m256 c3)
{
    __m256 reduced_x = _mm256_fmadd_ps(y, c1, x);
    reduced_x = _mm256_fmadd_ps(y, c2, reduced_x);
    reduced_x = _mm256_fmadd_ps(y, c3, reduced_x);
    return reduced_x;
}

#endif // SIMD_AVX2_FMA3

#ifdef SIMD_AVX512F

NPY_FINLINE __mmask8
avx512_get_full_load_mask_pd(void)
{
    return 0xFF;
}

NPY_FINLINE __mmask8
avx512_get_partial_load_mask_pd(const npy_int num_elem, const npy_int total_elem)
{
    return (0x01 << num_elem) - 0x01;
}

NPY_FINLINE __m512d
avx512_masked_gather_pd(__m512d src,
                        npy_double* addr,
                        __m256i vindex,
                        __mmask8 kmask)
{
    return _mm512_mask_i32gather_pd(src, kmask, vindex, addr, 8);
}

NPY_FINLINE __m512d
avx512_masked_load_pd(__mmask8 mask, npy_double* addr)
{
    return _mm512_maskz_loadu_pd(mask, (__m512d *)addr);
}

NPY_FINLINE __m512d
avx512_set_masked_lanes_pd(__m512d x, __m512d val, __mmask8 mask)
{
    return _mm512_mask_blend_pd(mask, x, val);
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

NPY_FINLINE __m512
simd_range_reduction(__m512 x, __m512 y, __m512 c1, __m512 c2, __m512 c3)
{
    __m512 reduced_x = _mm512_fmadd_ps(y, c1, x);
    reduced_x = _mm512_fmadd_ps(y, c2, reduced_x);
    reduced_x = _mm512_fmadd_ps(y, c3, reduced_x);
    return reduced_x;
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


// Shared ARM NEON tables, macros, and helpers are in loops_explog.h

static inline int neon_has_all_lanes_u32(uint32x4_t mask)
{
  return vminvq_u32(mask) != 0;
}

static inline int neon_has_all_lanes_u64(uint64x2_t mask)
{
  return (vgetq_lane_u64(mask, 0) & vgetq_lane_u64(mask, 1)) != 0;
}



#endif // SIMD_ARM

/*
 * SIMD-dispatched loop function declarations for exp, log, exp2, log2.
 *
 * Implementations live in loops_exp.dispatch.cpp (exp), loops_log.dispatch.cpp (log),
 * and loops_exp2.dispatch.cpp / loops_log2.dispatch.cpp (exp2/log2).
 * Each type (FLOAT, DOUBLE, HALF) has a separate dispatch table generated
 * by Meson's cc_simd mechanism; the auto-generated headers below supply the
 * NPY_CPU_DISPATCH_* macros.
 */
#include "loops_exp.dispatch.h"
#include "loops_log.dispatch.h"
#include "loops_exp2.dispatch.h"
#include "loops_log2.dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void FLOAT_exp,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void FLOAT_log,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void DOUBLE_exp,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void DOUBLE_log,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void HALF_exp,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void HALF_log,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void FLOAT_exp2,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void FLOAT_log2,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void DOUBLE_exp2,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void DOUBLE_log2,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void HALF_exp2,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void HALF_log2,
    (char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)))

#ifdef __cplusplus
}
#endif

#endif // _NPY_UMATH_LOOPS_EXPLOG_H_
