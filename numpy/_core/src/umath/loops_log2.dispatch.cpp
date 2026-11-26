#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "loops_explog.h"
#include "loops.h"

#ifdef SIMD_AVX2_FMA3

static void
simd_log2_FLOAT(npy_float * op,
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
    __m256 log2e = _mm256_set1_ps(NPY_LOG2Ef);
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

        __m256 num_p01 = _mm256_fmadd_ps(log_p1, x, log_p0);
        __m256 num_p23 = _mm256_fmadd_ps(log_p3, x, log_p2);
        __m256 num_p45 = _mm256_fmadd_ps(log_p5, x, log_p4);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 num_lo = _mm256_fmadd_ps(num_p23, x2, num_p01);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        num_poly = _mm256_fmadd_ps(num_p45, x4, num_lo);
        __m256 den_p01 = _mm256_fmadd_ps(log_q1, x, log_q0);
        __m256 den_p23 = _mm256_fmadd_ps(log_q3, x, log_q2);
        __m256 den_p45 = _mm256_fmadd_ps(log_q5, x, log_q4);
        __m256 den_lo = _mm256_fmadd_ps(den_p23, x2, den_p01);
        denom_poly = _mm256_fmadd_ps(den_p45, x4, den_lo);
        poly = _mm256_div_ps(num_poly, denom_poly);
        poly = _mm256_fmadd_ps(poly, log2e, exponent);

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

static void
simd_log2_FLOAT(npy_float * op,
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
    __m512 log2e = _mm512_set1_ps(NPY_LOG2Ef);
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

        __m512 num_p01 = _mm512_fmadd_ps(log_p1, x, log_p0);
        __m512 num_p23 = _mm512_fmadd_ps(log_p3, x, log_p2);
        __m512 num_p45 = _mm512_fmadd_ps(log_p5, x, log_p4);
        __m512 x2 = _mm512_mul_ps(x, x);
        __m512 num_lo = _mm512_fmadd_ps(num_p23, x2, num_p01);
        __m512 x4 = _mm512_mul_ps(x2, x2);
        num_poly = _mm512_fmadd_ps(num_p45, x4, num_lo);
        __m512 den_p01 = _mm512_fmadd_ps(log_q1, x, log_q0);
        __m512 den_p23 = _mm512_fmadd_ps(log_q3, x, log_q2);
        __m512 den_p45 = _mm512_fmadd_ps(log_q5, x, log_q4);
        __m512 den_lo = _mm512_fmadd_ps(den_p23, x2, den_p01);
        denom_poly = _mm512_fmadd_ps(den_p45, x4, den_lo);
        poly = _mm512_div_ps(num_poly, denom_poly);
        poly = _mm512_fmadd_ps(poly, log2e, exponent);

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

/*
 * Highway port of the Arm Optimized Routines (AOR) advsimd log/log2
 * kernels, https://github.com/ARM-software/optimized-routines
 * (math/aarch64/advsimd/log.c, log2.c, logf.c, log2f.c and the
 * v_log_data.c / v_log2_data.c lookup tables).
 *
 * Structured in the SVML style: each lane-parallel AOR evaluation is a
 * short-vector kernel API (log2_f64_kernel / log2_f32_kernel) and the
 * strided loops only do vectorized load -> kernel -> vectorized store.
 *
 * numpy adaptations over the AOR skeleton:
 *  - the AOR constant patches (log2(0) -> -inf, log2(neg) -> NaN) do not
 *    raise FP flags, so each kernel accumulates the special-lane masks
 *    into caller-owned invalid/divide accumulators and the loop writes
 *    divide-by-zero/invalid back at its exit, matching scalar libm;
 *  - NaN lanes pass through bit-exactly (sign + payload);
 *  - subnormals keep AOR's in-band 2*log2(sqrt(x)) promotion;
 *  - inactive tail lanes are padded with 1.0 so the native LoadN/StoreN
 *    zero fill cannot raise spurious divide-by-zero.
 */
#include "simd/simd.hpp"  /* pulls <hwy/highway.h> + defines NPY_HWY_F16 */
namespace hn = hwy::HWY_NAMESPACE;

const hn::ScalableTag<double> f64;
const hn::ScalableTag<int64_t> s64;
const hn::ScalableTag<uint64_t> u64;
using vec_f64 = hn::Vec<decltype(f64)>;
using vec_s64 = hn::Vec<decltype(s64)>;
using vec_u64 = hn::Vec<decltype(u64)>;
const hn::ScalableTag<float> f32;
const hn::ScalableTag<int32_t> s32;
const hn::ScalableTag<uint32_t> u32;
using vec_f32 = hn::Vec<decltype(f32)>;
using vec_u32 = hn::Vec<decltype(u32)>;
using mask_u64 = hn::Mask<hn::ScalableTag<uint64_t>>;
using mask_u32 = hn::Mask<hn::ScalableTag<uint32_t>>;

template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE vtype
load_vector(type_t* src, npy_intp ssrc, npy_intp len)
{
  auto D = hn::DFromV<vtype>();
  using DI = hn::RebindToSigned<decltype(D)>;
  DI di;

  auto indices = hn::Iota(di, 0);
  auto stride = hn::Set(di, ssrc);
  indices = hn::Mul(indices, stride);

  const int nlanes = hn::Lanes(D);
  if (len < nlanes) {
    if (ssrc == 1) {
      return hn::LoadN(D, src, len);
    } else {
      return hn::GatherIndexN(D, src, indices, len);
    }
  } else {
    if (ssrc == 1) {
      return hn::LoadU(D, src);
    } else {
      return hn::GatherIndex(D, src, indices);
    }
  }
}

template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE void
store_vector(vtype vec, type_t* dst, npy_intp sdst, npy_intp len)
{
  auto D = hn::DFromV<vtype>();
  using DI = hn::RebindToSigned<decltype(D)>;
  DI di;

  auto indices = hn::Iota(di, 0);
  auto stride = hn::Set(di, sdst);
  indices = hn::Mul(indices, stride);

  const int nlanes = hn::Lanes(D);
  if (len < nlanes) {
    if (sdst == 1) {
      hn::StoreN(vec, D, dst, len);
    } else {
      hn::ScatterIndexN(vec, D, dst, indices, len);
    }
  } else {
    if (sdst == 1) {
      hn::StoreU(vec, D, dst);
    } else {
      hn::ScatterIndex(vec, D, dst, indices);
    }
  }
}

/* AOR v_log2_data.c table (TABLE_BITS=7, N=128), de-interleaved into
 * per-field arrays for hn::GatherIndex. invc matches v_log_data.c;
 * log2c was scaled by log2(e) in extended precision upstream, so no
 * runtime invln2 rounding of the table is needed. */

static const double hwy_log2_table_invc[128] = {
  0x1.6a133d0dec120p+0, 0x1.6815f2f3e42edp+0, 0x1.661e39be1ac9ep+0, 0x1.642bfa30ac371p+0,
  0x1.623f1d916f323p+0, 0x1.60578da220f65p+0, 0x1.5e75349dea571p+0, 0x1.5c97fd387a75ap+0,
  0x1.5abfd2981f200p+0, 0x1.58eca051dc99cp+0, 0x1.571e526d9df12p+0, 0x1.5554d555b3fcbp+0,
  0x1.539015e2a20cdp+0, 0x1.51d0014ee0164p+0, 0x1.50148538cd9eep+0, 0x1.4e5d8f9f698a1p+0,
  0x1.4cab0edca66bep+0, 0x1.4afcf1a9db874p+0, 0x1.495327136e16fp+0, 0x1.47ad9e84af28fp+0,
  0x1.460c47b39ae15p+0, 0x1.446f12b278001p+0, 0x1.42d5efdd720ecp+0, 0x1.4140cfe001a0fp+0,
  0x1.3fafa3b421f69p+0, 0x1.3e225c9c8ece5p+0, 0x1.3c98ec29a211ap+0, 0x1.3b13442a413fep+0,
  0x1.399156baa3c54p+0, 0x1.38131639b4cdbp+0, 0x1.36987540fbf53p+0, 0x1.352166b648f61p+0,
  0x1.33adddb3eb575p+0, 0x1.323dcd99fc1d3p+0, 0x1.30d129fefc7d2p+0, 0x1.2f67e6b72fe7dp+0,
  0x1.2e01f7cf8b187p+0, 0x1.2c9f518ddc86ep+0, 0x1.2b3fe86e5f413p+0, 0x1.29e3b1211b25cp+0,
  0x1.288aa08b373cfp+0, 0x1.2734abcaa8467p+0, 0x1.25e1c82459b81p+0, 0x1.2491eb1ad59c5p+0,
  0x1.23450a54048b5p+0, 0x1.21fb1bb09e578p+0, 0x1.20b415346d8f7p+0, 0x1.1f6fed179a1acp+0,
  0x1.1e2e99b93c7b3p+0, 0x1.1cf011a7a882ap+0, 0x1.1bb44b97dba5ap+0, 0x1.1a7b3e66cdd4fp+0,
  0x1.1944e11dc56cdp+0, 0x1.18112aebb1a6ep+0, 0x1.16e013231b7e9p+0, 0x1.15b1913f156cfp+0,
  0x1.14859cdedde13p+0, 0x1.135c2dc68cfa4p+0, 0x1.12353bdb01684p+0, 0x1.1110bf25b85b4p+0,
  0x1.0feeafd2f8577p+0, 0x1.0ecf062c51c3bp+0, 0x1.0db1baa076c8bp+0, 0x1.0c96c5bb3048ep+0,
  0x1.0b7e20263e070p+0, 0x1.0a67c2acd0ce3p+0, 0x1.0953a6391e982p+0, 0x1.0841c3caea380p+0,
  0x1.07321489b13eap+0, 0x1.062491aee9904p+0, 0x1.05193497a7cc5p+0, 0x1.040ff6b5f5e9fp+0,
  0x1.0308d19aa6127p+0, 0x1.0203beedb0c67p+0, 0x1.010037d38bcc2p+0, 1.0,
  0x1.fc06d493cca10p-1, 0x1.f81e6ac3b918fp-1, 0x1.f44546ef18996p-1, 0x1.f07b10382c84bp-1,
  0x1.ecbf7070e59d4p-1, 0x1.e91213f715939p-1, 0x1.e572a9a75f7b7p-1, 0x1.e1e0e2c530207p-1,
  0x1.de5c72d8a8be3p-1, 0x1.dae50fa5658ccp-1, 0x1.d77a71145a2dap-1, 0x1.d41c51166623ep-1,
  0x1.d0ca6ba0bb29fp-1, 0x1.cd847e8e59681p-1, 0x1.ca4a499693e00p-1, 0x1.c71b8e399e821p-1,
  0x1.c3f80faf19077p-1, 0x1.c0df92dc2b0ecp-1, 0x1.bdd1de3cbb542p-1, 0x1.baceb9e1007a3p-1,
  0x1.b7d5ef543e55ep-1, 0x1.b4e749977d953p-1, 0x1.b20295155478ep-1, 0x1.af279f8e82be2p-1,
  0x1.ac5638197fdf3p-1, 0x1.a98e2f102e087p-1, 0x1.a6cf5606d05c1p-1, 0x1.a4197fc04d746p-1,
  0x1.a16c80293dc01p-1, 0x1.9ec82c4dc5bc9p-1, 0x1.9c2c5a491f534p-1, 0x1.9998e1480b618p-1,
  0x1.970d9977c6c2dp-1, 0x1.948a5c023d212p-1, 0x1.920f0303d6809p-1, 0x1.8f9b698a98b45p-1,
  0x1.8d2f6b81726f6p-1, 0x1.8acae5bb55badp-1, 0x1.886db5d9275b8p-1, 0x1.8617ba567c13cp-1,
  0x1.83c8d27487800p-1, 0x1.8180de3c5dbe7p-1, 0x1.7f3fbe71cdb71p-1, 0x1.7d055498071c1p-1,
  0x1.7ad182e54f65ap-1, 0x1.78a42c3c90125p-1, 0x1.767d342f76944p-1, 0x1.745c7ef26b00ap-1,
  0x1.7241f15769d0fp-1, 0x1.702d70d396e41p-1, 0x1.6e1ee3700cd11p-1, 0x1.6c162fc9cbe02p-1,
};
static const double hwy_log2_table_log2c[128] = {
  -0x1.00130d57f5fadp-1, -0x1.f802661bd725ep-2, -0x1.efea1c6f73a5bp-2, -0x1.e7dd1dcd06f05p-2,
  -0x1.dfdb4ae024809p-2, -0x1.d7e484d101958p-2, -0x1.cff8ad452f6ep-2, -0x1.c817a666c997fp-2,
  -0x1.c04152d640419p-2, -0x1.b87595a3f64b2p-2, -0x1.b0b4526c44d07p-2, -0x1.a8fd6d1a90f5ep-2,
  -0x1.a150ca2559fc6p-2, -0x1.99ae4e62cca29p-2, -0x1.9215df1a1e842p-2, -0x1.8a8761fe1f0d9p-2,
  -0x1.8302bd1cc9a54p-2, -0x1.7b87d6fb437f6p-2, -0x1.741696673a86dp-2, -0x1.6caee2b3c6fe4p-2,
  -0x1.6550a3666c27ap-2, -0x1.5dfbc08de02a4p-2, -0x1.56b022766c84ap-2, -0x1.4f6db1c955536p-2,
  -0x1.4834579063054p-2, -0x1.4103fd2249a76p-2, -0x1.39dc8c3fe6dabp-2, -0x1.32bdeed4b5c8fp-2,
  -0x1.2ba80f41e20ddp-2, -0x1.249ad8332f4a7p-2, -0x1.1d96347e7f3ebp-2, -0x1.169a0f7d6604ap-2,
  -0x1.0fa654a221909p-2, -0x1.08baefcf8251ap-2, -0x1.01d7cd14deecdp-2, -0x1.f5f9b1ad55495p-3,
  -0x1.e853ff76a77afp-3, -0x1.dabe5d624cba1p-3, -0x1.cd38a5cef4822p-3, -0x1.bfc2b38d315f9p-3,
  -0x1.b25c61f5edd0fp-3, -0x1.a5058d18e9cacp-3, -0x1.97be1113e47a3p-3, -0x1.8a85cafdf5e27p-3,
  -0x1.7d5c97e8fc45bp-3, -0x1.704255d6486e4p-3, -0x1.6336e2cedd7bfp-3, -0x1.563a1d9b0cc6ap-3,
  -0x1.494be541aaa6fp-3, -0x1.3c6c1964dd0f2p-3, -0x1.2f9a99f19a243p-3, -0x1.22d747344446p-3,
  -0x1.1622020d4f7f5p-3, -0x1.097aabb3553f3p-3, -0x1.f9c24b48014c5p-4, -0x1.e0aaa3bdc858ap-4,
  -0x1.c7ae257c952d6p-4, -0x1.aecc960a03e58p-4, -0x1.9605bb724d541p-4, -0x1.7d595ca7147cep-4,
  -0x1.64c74165002d9p-4, -0x1.4c4f31c86d344p-4, -0x1.33f0f70388258p-4, -0x1.1bac5abb3037dp-4,
  -0x1.0381272495f21p-4, -0x1.d6de4eba2de2ap-5, -0x1.a6ec4e8156898p-5, -0x1.772be542e3e1bp-5,
  -0x1.479cadcde852dp-5, -0x1.183e4265faa5p-5, -0x1.d2207fdaa1b85p-6, -0x1.742486cb4a6a2p-6,
  -0x1.1687d77cfc299p-6, -0x1.7293623a6b5dep-7, -0x1.70ec80ec8f25dp-8, 0.0,
  0x1.704c1ca6b6bc9p-7, 0x1.6eac8ba664beap-6, 0x1.11e67d040772dp-5, 0x1.6bc665e2105dep-5,
  0x1.c4f8a9772bf1dp-5, 0x1.0ebff10fbb951p-4, 0x1.3aaf4d7805d11p-4, 0x1.664ba81a4d717p-4,
  0x1.9196387da6de4p-4, 0x1.bc902f2b7796p-4, 0x1.e73ab5f584f28p-4, 0x1.08cb78510d232p-3,
  0x1.1dd2fe2f0dcb5p-3, 0x1.32b4784400df4p-3, 0x1.47706f3d49942p-3, 0x1.5c0768ee4a4dcp-3,
  0x1.7079e86fc7c6dp-3, 0x1.84c86e1183467p-3, 0x1.98f377a34b499p-3, 0x1.acfb803bc924bp-3,
  0x1.c0e10098b025fp-3, 0x1.d4a46efe103efp-3, 0x1.e8463f45b8d0bp-3, 0x1.fbc6e3228997fp-3,
  0x1.079364f2e5aa8p-2, 0x1.1133306010a63p-2, 0x1.1ac309631bd17p-2, 0x1.24432485370c1p-2,
  0x1.2db3b5449132fp-2, 0x1.3714ee1d7a32p-2, 0x1.406700ab52c94p-2, 0x1.49aa1d87522b2p-2,
  0x1.52de746d7ecb2p-2, 0x1.5c0434336b343p-2, 0x1.651b8ad6c90d1p-2, 0x1.6e24a56ab5831p-2,
  0x1.771fb04ec29b1p-2, 0x1.800cd6f19c25ep-2, 0x1.88ec441df11dfp-2, 0x1.91be21b7c93f5p-2,
  0x1.9a8298f8c7454p-2, 0x1.a339d255c04ddp-2, 0x1.abe3f59f43db7p-2, 0x1.b48129deca9efp-2,
  0x1.bd119575364c1p-2, 0x1.c5955e23ebcbcp-2, 0x1.ce0ca8f4e1557p-2, 0x1.d6779a5a75774p-2,
  0x1.ded6563550d27p-2, 0x1.e728ffafd840ep-2, 0x1.ef6fb96c8d739p-2, 0x1.f7aaa57907219p-2,
};

/* AOR inline_log2: x = 2^k * z with z in [Off, 2*Off), split into N=128
 * subintervals; log2(x) = log1p(z/c - 1)/ln2 + log2(c) + k. The reduction
 * reconstructs z as a normal positive value for any input bit pattern,
 * so this core raises no invalid/divide/overflow/underflow flags. */
HWY_ATTR NPY_FINLINE vec_f64
hwy_log2_core(vec_u64 u, vec_u64 u_off)
{
  const vec_f64 c0 = hn::Set(f64, -0x1.71547652b8300p-1);
  const vec_f64 c1 = hn::Set(f64, 0x1.ec709dc340953p-2);
  const vec_f64 c2 = hn::Set(f64, -0x1.71547651c8f35p-2);
  const vec_f64 c3 = hn::Set(f64, 0x1.2777ebe12dda5p-2);
  const vec_f64 c4 = hn::Set(f64, -0x1.ec738d616fe26p-3);
  const vec_f64 invln2 = hn::Set(f64, 0x1.71547652b82fep0);

  vec_s64 k = hn::ShiftRight<52>(hn::BitCast(s64, u_off)); /* arithmetic */
  vec_u64 iz = hn::Sub(u, hn::And(u_off, hn::Set(u64, 0xfff0000000000000ULL)));
  vec_f64 z = hn::BitCast(f64, iz);
  /* per-lane table index: (u_off >> (52 - TABLE_BITS)) & (N - 1) */
  vec_u64 idx = hn::And(hn::ShiftRight<45>(u_off), hn::Set(u64, 127));
  vec_f64 invc = hn::GatherIndex(f64, hwy_log2_table_invc, hn::BitCast(s64, idx));
  vec_f64 log2c = hn::GatherIndex(f64, hwy_log2_table_log2c, hn::BitCast(s64, idx));

  vec_f64 r = hn::MulAdd(z, invc, hn::Set(f64, -1.0));
  vec_f64 kd = hn::ConvertTo(f64, k);
  vec_f64 hi = hn::MulAdd(r, invln2, hn::Add(log2c, kd));
  /* y = r2*(c0 + r*c1 + r2*(c2 + r*c3 + r2*c4)) + hi */
  vec_f64 r2 = hn::Mul(r, r);
  vec_f64 y = hn::MulAdd(r, c3, c2);
  vec_f64 p = hn::MulAdd(r, c1, c0);
  y = hn::MulAdd(r2, c4, y);
  y = hn::MulAdd(r2, y, p);
  return hn::MulAdd(r2, y, hi);
}

/* AOR special_case: subnormal/zero/negative/NaN lanes are promoted via
 * 2*log2(sqrt(x)); inf/nan/negative/zero are then patched to constants.
 * All detection uses integer bit compares (no FP flags). The NaN
 * constant patch is overridden by a bit-exact NaN passthrough, and the
 * invalid/divide masks are accumulated for the exit writeback. */
HWY_ATTR NPY_FINLINE vec_f64
hwy_log2_special(vec_u64 u_off, mask_u64 &invalid_any,
                 mask_u64 &divbyzero_any)
{
  const vec_u64 off = hn::Set(u64, 0x3fe6900900000000ULL);
  const vec_u64 inf_bits = hn::Set(u64, 0x7ff0000000000000ULL);
  vec_u64 u = hn::Add(u_off, off); /* recover x */
  vec_f64 x = hn::BitCast(f64, u);
  auto promote = hn::Ge(
      hn::Sub(u_off, hn::Set(u64, 0x0010000000000000ULL - 0x3fe6900900000000ULL)),
      hn::Set(u64, 0x7ffe000000000000ULL));
  vec_f64 xs = hn::IfThenElse(hn::RebindMask(f64, promote), hn::Sqrt(x), x);
  vec_u64 us = hn::BitCast(u64, xs);
  vec_f64 y = hwy_log2_core(us, hn::Sub(us, off));
  y = hn::IfThenElse(hn::RebindMask(f64, promote), hn::Mul(y, hn::Set(f64, 2.0)), y);

  /* unsigned bits >= +inf covers +-inf, +-NaN and every negative */
  vec_u64 abs_u = hn::And(u, hn::Set(u64, 0x7fffffffffffffffULL));
  auto is_infnan = hn::Ge(u, inf_bits);
  auto is_zero = hn::Eq(abs_u, hn::Zero(u64));
  auto is_nan = hn::Gt(abs_u, inf_bits);
  y = hn::IfThenElse(hn::RebindMask(f64, hn::Or(is_infnan, is_zero)),
                     hn::Set(f64, NPY_NAN), y);
  y = hn::IfThenElse(hn::RebindMask(f64, hn::Eq(u, inf_bits)),
                     hn::Set(f64, NPY_INFINITY), y);
  y = hn::IfThenElse(hn::RebindMask(f64, is_zero),
                     hn::Set(f64, -NPY_INFINITY), y);
  /* contract: NaN sign + payload pass through bit-exactly; quietize a
   * signaling NaN (set bit 51) so the result is a quiet NaN, matching
   * libm, which quietizes sNaN while preserving sign + payload */
  y = hn::IfThenElse(hn::RebindMask(f64, is_nan),
                     hn::BitCast(f64, hn::Or(u, hn::Set(u64, 0x0008000000000000ULL))), y);

  /* invalid: negative finite or -inf (not -0, not NaN); divide: +-0 */
  auto sign_set = hn::Ne(hn::And(u, hn::Set(u64, 0x8000000000000000ULL)),
                         hn::Zero(u64));
  auto is_neg = hn::And(sign_set, hn::Not(hn::Or(is_nan, is_zero)));
  invalid_any = hn::Or(invalid_any, is_neg);
  divbyzero_any = hn::Or(divbyzero_any, is_zero);
  return y;
}

/* Short-vector API: one lane-parallel log2(x) evaluation. The clean /
 * special branch lives here; special lanes accumulate invalid/divide
 * into the caller-owned accumulators for the loop-exit writeback. */
HWY_ATTR NPY_FINLINE vec_f64
log2_f64_kernel(vec_f64 x, mask_u64& inv_acc, mask_u64& div_acc)
{
  const vec_u64 off = hn::Set(u64, 0x3fe6900900000000ULL);
  /* lower bound - offset (wraps): subnormals are detected after the
   * offset has been subtracted, as in AOR */
  const vec_u64 off_lower =
      hn::Set(u64, 0x0010000000000000ULL - 0x3fe6900900000000ULL);
  /* asuint64(inf) - asuint64(0x1p-1022): the fast path covers exactly
   * the positive normals below inf (AOR's narrowed u32 check) */
  const vec_u64 outer_bound = hn::Set(u64, 0x7fe0000000000000ULL);
  vec_u64 u = hn::BitCast(u64, x);
  vec_u64 u_off = hn::Sub(u, off);
  auto special_m = hn::Ge(hn::Sub(u_off, off_lower), outer_bound);
  if (!hn::AllFalse(u64, special_m)) {
    return hwy_log2_special(u_off, inv_acc, div_acc);
  }
  return hwy_log2_core(u, u_off);
}

HWY_ATTR static void
simd_log2_hwy_DOUBLE(const npy_double* src, npy_intp ssrc,
                     npy_double* dst, npy_intp sdst, npy_intp len)
{
  const int nlanes = hn::Lanes(f64);
  auto inv_acc = hn::MaskFalse(u64);
  auto div_acc = hn::MaskFalse(u64);

  for (; len > 0; len -= nlanes, src += ssrc * nlanes, dst += sdst * nlanes) {
    vec_f64 x = load_vector<vec_f64>(src, ssrc, len);
    if (len < nlanes) {
      /* pad inactive tail lanes with 1.0 (log2(1) = 0, no flags, not
       * special): LoadN's zero fill would otherwise raise divide */
      x = hn::IfThenElse(hn::FirstN(f64, len), x, hn::Set(f64, 1.0));
    }
    vec_f64 out = log2_f64_kernel(x, inv_acc, div_acc);
    store_vector<vec_f64>(out, dst, sdst, len);
  }
  /* The constant patches never execute a trapping instruction, so raise
   * the FP flags libm's log2 would have raised for the special inputs
   * (invalid: x < 0 or -inf, divide-by-zero: +-0) explicitly. */
  if (!hn::AllFalse(u64, inv_acc)) {
    npy_set_floatstatus_invalid();
  }
  if (!hn::AllFalse(u64, div_acc)) {
    npy_set_floatstatus_divbyzero();
  }
}

/* AOR inline_log2f: same argument reduction as logf;
 * log2(x) = log2(1+r) + n with a degree-8 polynomial (Remez on
 * log2(1+r)/r, r in [-1/3, 1/3]). Flag-free for any input bit pattern. */
HWY_ATTR NPY_FINLINE vec_f32
hwy_log2f_core(vec_u32 u_off, vec_f32 n)
{
  const vec_f32 c0 = hn::Set(f32, 0x1.715476p0f);
  const vec_f32 c1 = hn::Set(f32, -0x1.715458p-1f);
  const vec_f32 c2 = hn::Set(f32, 0x1.ec701cp-2f);
  const vec_f32 c3 = hn::Set(f32, -0x1.7171a4p-2f);
  const vec_f32 c4 = hn::Set(f32, 0x1.27a0b8p-2f);
  const vec_f32 c5 = hn::Set(f32, -0x1.e5143ep-3f);
  const vec_f32 c6 = hn::Set(f32, 0x1.9d8ecap-3f);
  const vec_f32 c7 = hn::Set(f32, -0x1.c675bp-3f);
  const vec_f32 c8 = hn::Set(f32, 0x1.9e495p-3f);

  vec_u32 u = hn::Add(hn::And(u_off, hn::Set(u32, 0x007fffff)),
                      hn::Set(u32, 0x3f2aaaab));
  vec_f32 r = hn::Sub(hn::BitCast(f32, u), hn::Set(f32, 1.0f));
  vec_f32 r2 = hn::Mul(r, r);
  vec_f32 c01 = hn::MulAdd(r, c1, c0);
  vec_f32 c23 = hn::MulAdd(r, c3, c2);
  vec_f32 c45 = hn::MulAdd(r, c5, c4);
  vec_f32 c67 = hn::MulAdd(r, c7, c6);
  vec_f32 p68 = hn::MulAdd(r2, c8, c67);
  vec_f32 p48 = hn::MulAdd(r2, p68, c45);
  vec_f32 p28 = hn::MulAdd(r2, p48, c23);
  vec_f32 p = hn::MulAdd(r2, p28, c01);
  return hn::MulAdd(p, r, n);
}

/* AOR log2f special_case: per-lane 2*log2(sqrt(x)) promotion for
 * subnormal/zero/negative/NaN; constant patches + NaN passthrough +
 * invalid/divide accumulation (see the f64 kernel). */
HWY_ATTR NPY_FINLINE vec_f32
hwy_log2f_special(vec_u32 u_off, mask_u32 &invalid_any,
                  mask_u32 &divbyzero_any)
{
  const vec_u32 off = hn::Set(u32, 0x3f2aaaab);
  const vec_u32 inf_bits = hn::Set(u32, 0x7f800000);
  vec_u32 u = hn::Add(u_off, off); /* recover x */
  vec_f32 x = hn::BitCast(f32, u);
  auto promote = hn::Ge(hn::Sub(u_off, hn::Set(u32, 0x00800000 - 0x3f2aaaab)),
                        hn::Set(u32, 0x7f000000));
  vec_f32 xs = hn::IfThenElse(hn::RebindMask(f32, promote), hn::Sqrt(x), x);
  vec_u32 us_off = hn::Sub(hn::BitCast(u32, xs), off);
  vec_f32 n = hn::ConvertTo(f32, hn::ShiftRight<23>(hn::BitCast(s32, us_off)));
  vec_f32 y = hwy_log2f_core(us_off, n);
  y = hn::IfThenElse(hn::RebindMask(f32, promote), hn::Mul(y, hn::Set(f32, 2.0f)), y);

  vec_u32 abs_u = hn::And(u, hn::Set(u32, 0x7fffffff));
  auto is_infnan = hn::Ge(u, inf_bits);
  auto is_zero = hn::Eq(abs_u, hn::Zero(u32));
  auto is_nan = hn::Gt(abs_u, inf_bits);
  y = hn::IfThenElse(hn::RebindMask(f32, hn::Or(is_infnan, is_zero)),
                     hn::Set(f32, NPY_NANF), y);
  y = hn::IfThenElse(hn::RebindMask(f32, hn::Eq(u, inf_bits)),
                     hn::Set(f32, NPY_INFINITYF), y);
  y = hn::IfThenElse(hn::RebindMask(f32, is_zero),
                     hn::Set(f32, -NPY_INFINITYF), y);
  /* NaN passthrough; quietize sNaN (set bit 22) to match libm, as f64 */
  y = hn::IfThenElse(hn::RebindMask(f32, is_nan),
                     hn::BitCast(f32, hn::Or(u, hn::Set(u32, 0x00400000))), y);

  auto sign_set = hn::Ne(hn::And(u, hn::Set(u32, 0x80000000)), hn::Zero(u32));
  auto is_neg = hn::And(sign_set, hn::Not(hn::Or(is_nan, is_zero)));
  invalid_any = hn::Or(invalid_any, is_neg);
  divbyzero_any = hn::Or(divbyzero_any, is_zero);
  return y;
}

/* Short-vector API: one lane-parallel log2(x) evaluation (f32). */
HWY_ATTR NPY_FINLINE vec_f32
log2_f32_kernel(vec_f32 x, mask_u32& inv_acc, mask_u32& div_acc)
{
  const vec_u32 off = hn::Set(u32, 0x3f2aaaab);
  const vec_u32 off_lower = hn::Set(u32, 0x00800000 - 0x3f2aaaab);
  const vec_u32 outer_bound = hn::Set(u32, 0x7f000000);
  vec_u32 u = hn::BitCast(u32, x);
  vec_u32 u_off = hn::Sub(u, off);
  auto special_m = hn::Ge(hn::Sub(u_off, off_lower), outer_bound);
  if (!hn::AllFalse(u32, special_m)) {
    return hwy_log2f_special(u_off, inv_acc, div_acc);
  }
  vec_f32 n = hn::ConvertTo(f32, hn::ShiftRight<23>(hn::BitCast(s32, u_off)));
  return hwy_log2f_core(u_off, n);
}

HWY_ATTR static void
simd_log2_hwy_FLOAT(const npy_float* src, npy_intp ssrc,
                    npy_float* dst, npy_intp sdst, npy_intp len)
{
  const int nlanes = hn::Lanes(f32);
  auto inv_acc = hn::MaskFalse(u32);
  auto div_acc = hn::MaskFalse(u32);

  for (; len > 0; len -= nlanes, src += ssrc * nlanes, dst += sdst * nlanes) {
    vec_f32 x = load_vector<vec_f32>(src, ssrc, len);
    if (len < nlanes) {
      x = hn::IfThenElse(hn::FirstN(f32, len), x, hn::Set(f32, 1.0f));
    }
    vec_f32 out = log2_f32_kernel(x, inv_acc, div_acc);
    store_vector<vec_f32>(out, dst, sdst, len);
  }
  if (!hn::AllFalse(u32, inv_acc)) {
    npy_set_floatstatus_invalid();
  }
  if (!hn::AllFalse(u32, div_acc)) {
    npy_set_floatstatus_divbyzero();
  }
}


/* f16 path: highway-vectorized f16 load -> promote to two f32 halves ->
 * log2_f32_kernel (shared accumulators) -> demote -> combine -> f16 store. No
 * malloc temp buffers and no raw NEON. f16 has no SVE gather/scatter, so
 * the strided case uses a small stack gather + LoadN/StoreN; the tail
 * zero-fill is padded with 1.0 before the kernel. The invalid/divide
 * accumulators are owned here and written back at the loop exit. */
#if NPY_HWY_F16
const hn::ScalableTag<::hwy::float16_t> f16;
using vec_f16 = hn::Vec<decltype(f16)>;

HWY_ATTR NPY_FINLINE vec_f16
load_f16(const ::hwy::float16_t* src, npy_intp ssrc, npy_intp len)
{
  const int nlanes = hn::Lanes(f16);
  if (ssrc == 1) {
    if (len < nlanes) {
      return hn::LoadN(f16, src, len);
    }
    return hn::LoadU(f16, src);
  }
  ::hwy::float16_t tmp[hn::MaxLanes(f16)];
  const int n = (len < nlanes) ? static_cast<int>(len) : nlanes;
  for (int j = 0; j < n; ++j) {
    tmp[j] = src[j * ssrc];
  }
  return hn::LoadN(f16, tmp, n);
}

HWY_ATTR NPY_FINLINE void
store_f16(vec_f16 v, ::hwy::float16_t* dst, npy_intp sdst, npy_intp len)
{
  const int nlanes = hn::Lanes(f16);
  if (sdst == 1) {
    if (len < nlanes) {
      hn::StoreN(v, f16, dst, len);
    } else {
      hn::StoreU(v, f16, dst);
    }
    return;
  }
  ::hwy::float16_t tmp[hn::MaxLanes(f16)];
  const int n = (len < nlanes) ? static_cast<int>(len) : nlanes;
  hn::StoreN(v, f16, tmp, n);
  for (int j = 0; j < n; ++j) {
    dst[j * sdst] = tmp[j];
  }
}

HWY_ATTR static void
simd_log2_hwy_HALF(const npy_half* src, npy_intp ssrc,
      npy_half* dst, npy_intp sdst, npy_intp len)
{
  const int nlanes = hn::Lanes(f16);
  const ::hwy::float16_t* src16 = reinterpret_cast<const ::hwy::float16_t*>(src);
  ::hwy::float16_t* dst16 = reinterpret_cast<::hwy::float16_t*>(dst);
  auto inv_acc = hn::MaskFalse(u32);
  auto div_acc = hn::MaskFalse(u32);

  for (; len > 0; len -= nlanes, src16 += ssrc * nlanes, dst16 += sdst * nlanes) {
    vec_f16 h = load_f16(src16, ssrc, len);
    if (len < nlanes) {
      h = hn::IfThenElse(hn::FirstN(f16, len), h,
                         hn::Set(f16, ::hwy::float16_t(1.0f)));
    }
    vec_f32 x0 = hn::PromoteLowerTo(f32, h);
    vec_f32 x1 = hn::PromoteUpperTo(f32, h);
    vec_f32 y0 = log2_f32_kernel(x0, inv_acc, div_acc);
    vec_f32 y1 = log2_f32_kernel(x1, inv_acc, div_acc);
    vec_f16 out = hn::Combine(f16, hn::DemoteTo(f16, y1), hn::DemoteTo(f16, y0));
    store_f16(out, dst16, sdst, len);
  }
  if (!hn::AllFalse(u32, inv_acc)) {
    npy_set_floatstatus_invalid();
  }
  if (!hn::AllFalse(u32, div_acc)) {
    npy_set_floatstatus_divbyzero();
  }
}
#endif  /* NPY_HWY_F16 */

#endif  /* SIMD_ARM */
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

static void avx512_log2_f16(const npy_half *src, npy_half *dst, npy_intp len)
{
    avx512_exponent_f16(src, dst, len, __svml_log2f16, 0x3c00);
}

static void
simd_log2_svml_f32(const npyv_lanetype_f32 *src, npy_intp ssrc,
                   npyv_lanetype_f32 *dst, npy_intp sdst, npy_intp len)
{
    const int vstep = npyv_nlanes_f32;
    for (; len > 0; len -= vstep, src += ssrc*vstep, dst += sdst*vstep) {
        npyv_f32 x;
        if (ssrc == 1) {
            x = npyv_load_till_f32(src, len, 1);
        } else {
            x = npyv_loadn_till_f32(src, ssrc, len, 1);
        }
        npyv_f32 out = __svml_log2f16(x);
        if (sdst == 1) {
            npyv_store_till_f32(dst, len, out);
        } else {
            npyv_storen_till_f32(dst, sdst, len, out);
        }
    }
    npyv_cleanup();
}

static void
simd_log2_svml_f64(const npyv_lanetype_f64 *src, npy_intp ssrc,
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
        npyv_f64 out = __svml_log28_ha(x);
        if (sdst == 1) {
            npyv_store_till_f64(dst, len, out);
        } else {
            npyv_storen_till_f64(dst, sdst, len, out);
        }
    }
    npyv_cleanup();
}
#endif

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_log2)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if SIMD_ARM
    npy_intp len = dimensions[0];
    /* Vectorize only contiguous non-overlapping runs; strided/overlap
     * falls back to scalar libm (policy note in loops_explog.h). */
    if (steps[0] == (npy_intp)sizeof(npy_float) &&
        steps[1] == (npy_intp)sizeof(npy_float) &&
        !is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        simd_log2_hwy_FLOAT((npy_float *)args[0], 1, (npy_float *)args[1], 1, len);
    } else {
        UNARY_LOOP {
            const npy_float in1 = *(npy_float *)ip1;
            *(npy_float *)op1 = npy_log2f(in1);
        }
    }
    return;
#elif NPY__SVML_IS_ENABLED
    const npy_float *src = (npy_float*)args[0];
          npy_float *dst = (npy_float*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        npyv_loadable_stride_f32(steps[0]) &&
        npyv_storable_stride_f32(steps[1])) {
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);
        simd_log2_svml_f32(src, ssrc, dst, sdst, len);
        return;
    }
#endif
    #if defined(SIMD_AVX2_FMA3) || defined(SIMD_AVX512F)
    if (IS_OUTPUT_BLOCKABLE_UNARY(sizeof(npy_float), sizeof(npy_float), 64)) {
        simd_log2_FLOAT((npy_float*)args[1], (npy_float*)args[0], dimensions[0], steps[0]);
    }
    else {
        UNARY_LOOP {
            simd_log2_FLOAT((npy_float *)op1, (npy_float *)ip1, 1, steps[0]);
        }
    }
    #else
    UNARY_LOOP {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = npy_log2f(in1);
    }
    #endif
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_log2)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if SIMD_ARM
    npy_intp len = dimensions[0];
    /* Vectorize only contiguous non-overlapping runs; strided/overlap
     * falls back to scalar libm (policy note in loops_explog.h). */
    if (steps[0] == (npy_intp)sizeof(npy_double) &&
        steps[1] == (npy_intp)sizeof(npy_double) &&
        !is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        simd_log2_hwy_DOUBLE((npy_double *)args[0], 1, (npy_double *)args[1], 1, len);
    } else {
        UNARY_LOOP {
            const npy_double in1 = *(npy_double *)ip1;
            *(npy_double *)op1 = npy_log2(in1);
        }
    }
    return;
#elif NPY__SVML_IS_ENABLED
    const npy_double *src = (npy_double*)args[0];
          npy_double *dst = (npy_double*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        npyv_loadable_stride_f64(steps[0]) &&
        npyv_storable_stride_f64(steps[1])) {
        const npy_intp ssrc = steps[0] / sizeof(npy_double);
        const npy_intp sdst = steps[1] / sizeof(npy_double);
        simd_log2_svml_f64(src, ssrc, dst, sdst, len);
        return;
    }
#endif
    UNARY_LOOP {
        const npy_double in1 = *(npy_double *)ip1;
        *(npy_double *)op1 = npy_log2(in1);
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_log2)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
#if SIMD_ARM && NPY_HWY_F16
    npy_intp len = dimensions[0];
    /* Vectorize only contiguous non-overlapping runs; strided/overlap
     * falls back to scalar libm (policy note in loops_explog.h). */
    if (steps[0] == (npy_intp)sizeof(npy_half) &&
        steps[1] == (npy_intp)sizeof(npy_half) &&
        !is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        simd_log2_hwy_HALF((npy_half *)args[0], 1, (npy_half *)args[1], 1, len);
    } else {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_log2f(in1));
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
        __svml_log2s32(src, dst, len);
    #else
        avx512_log2_f16(src, dst, len);
    #endif
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_log2f(in1));
    }
}
