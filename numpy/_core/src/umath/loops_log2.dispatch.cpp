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

static inline float64x2_t
neon_log2_inline_core(uint64x2_t u, uint64x2_t u_off)
{
  const float64x2_t c0 = vdupq_n_f64(-0x1.71547652b8300p-1);
  const float64x2_t c1 = vdupq_n_f64(0x1.ec709dc340953p-2);
  const float64x2_t c2 = vdupq_n_f64(-0x1.71547651c8f35p-2);
  const float64x2_t c3 = vdupq_n_f64(0x1.2777ebe12dda5p-2);
  const float64x2_t c4 = vdupq_n_f64(-0x1.ec738d616fe26p-3);
  const float64x2_t invln2 = vdupq_n_f64(0x1.71547652b82fep0);
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

  float64x2_t log2c = vmulq_f64(logc, invln2);
  float64x2_t r = vfmaq_f64(vdupq_n_f64(-1.0), z, invc);
  float64x2_t kd = vcvtq_f64_s64(k);

  float64x2_t hi = vfmaq_f64(vaddq_f64(log2c, kd), r, invln2);
  float64x2_t r2 = vmulq_f64(r, r);
  float64x2_t y = vfmaq_f64(c2, r, c3);
  float64x2_t p = vfmaq_f64(c0, r, c1);
  y = vfmaq_f64(y, r2, c4);
  y = vfmaq_f64(p, r2, y);

  return vfmaq_f64(hi, y, r2);
}

static void
simd_log2_neon_DOUBLE(const npy_double *src, npy_intp ssrc,
                       npy_double *dst, npy_intp sdst, npy_intp len)
{
  const float64x2_t c0 = vdupq_n_f64(-0x1.71547652b8300p-1);
  const float64x2_t c1 = vdupq_n_f64(0x1.ec709dc340953p-2);
  const float64x2_t c2 = vdupq_n_f64(-0x1.71547651c8f35p-2);
  const float64x2_t c3 = vdupq_n_f64(0x1.2777ebe12dda5p-2);
  const float64x2_t c4 = vdupq_n_f64(-0x1.ec738d616fe26p-3);
  const float64x2_t invln2 = vdupq_n_f64(0x1.71547652b82fep0);
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
    float64x2_t x[UNROLL], safe[UNROLL], result[UNROLL];
    uint64x2_t u[UNROLL], is_nan[UNROLL], is_pos_inf[UNROLL], is_neg_inf[UNROLL];
    uint64x2_t is_zero[UNROLL], is_neg[UNROLL], full_mask[UNROLL];

    if (ssrc == 1) {
      __builtin_prefetch(infp + 128, 0, 3);
      for (int i = 0; i < UNROLL; i++)
        x[i] = vld1q_f64(infp + i * VEC_SIZE);
    } else {
      for (int i = 0; i < UNROLL; i++) {
        double vals[2] = {infp[2 * i * ssrc], infp[(2 * i + 1) * ssrc]};
        x[i] = vld1q_f64(vals);
      }
    }

    for (int i = 0; i < UNROLL; i++) u[i] = vreinterpretq_u64_f64(x[i]);

    for (int i = 0; i < UNROLL; i++) {
      is_nan[i] = vceqq_f64(x[i], x[i]);
      is_nan[i] = veorq_u64(is_nan[i], all_ones);
      is_pos_inf[i] = vceqq_f64(x[i], pos_inf);
      is_neg_inf[i] = vceqq_f64(x[i], neg_inf_val);
      is_zero[i] = vceqq_f64(x[i], zeros_f);
      is_neg[i] = vcltq_f64(x[i], zeros_f);
      is_neg[i] = vbicq_u64(is_neg[i], vorrq_u64(is_neg_inf[i], is_zero[i]));
      uint64x2_t special_mask = vorrq_u64(vorrq_u64(is_nan[i], is_pos_inf[i]),
                                           vorrq_u64(is_neg_inf[i], is_zero[i]));
      full_mask[i] = vorrq_u64(special_mask, is_neg[i]);
      safe[i] = vbslq_f64(full_mask[i], ones_f, x[i]);
      u[i] = vreinterpretq_u64_f64(safe[i]);
    }

    uint64x2_t u_off[UNROLL];
    int64x2_t k[UNROLL];
    float64x2_t z[UNROLL], r[UNROLL], r2[UNROLL], invc[UNROLL], log2c[UNROLL];
    float64x2_t kd[UNROLL], hi[UNROLL], y[UNROLL], p[UNROLL];

    for (int i = 0; i < UNROLL; i++) {
      u_off[i] = vsubq_u64(u[i], off);
      k[i] = vshrq_n_s64(vreinterpretq_s64_u64(u_off[i]), 52);
      uint64x2_t iz = vsubq_u64(u[i], vandq_u64(u_off[i], sign_exp_mask));
      z[i] = vreinterpretq_f64_u64(iz);

      uint64_t idx0 = (vgetq_lane_u64(u_off[i], 0) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;
      uint64_t idx1 = (vgetq_lane_u64(u_off[i], 1) >> (52 - V_LOG_TABLE_BITS)) & V_LOG_INDEX_MASK;
      invc[i] = vcombine_f64(vdup_n_f64(neon_log_table[idx0].invc), vdup_n_f64(neon_log_table[idx1].invc));
      float64x2_t logc = vcombine_f64(vdup_n_f64(neon_log_table[idx0].logc), vdup_n_f64(neon_log_table[idx1].logc));
      log2c[i] = vmulq_f64(logc, invln2);

      r[i] = vfmaq_f64(neg_one, z[i], invc[i]);
      kd[i] = vcvtq_f64_s64(k[i]);
      hi[i] = vfmaq_f64(vaddq_f64(log2c[i], kd[i]), r[i], invln2);
      r2[i] = vmulq_f64(r[i], r[i]);
      y[i] = vfmaq_f64(c2, r[i], c3);
      p[i] = vfmaq_f64(c0, r[i], c1);
      y[i] = vfmaq_f64(y[i], r2[i], c4);
      y[i] = vfmaq_f64(p[i], r2[i], y[i]);
      result[i] = vfmaq_f64(hi[i], y[i], r2[i]);
    }

    for (int i = 0; i < UNROLL; i++) {
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
      for (int i = 0; i < VEC_SIZE; i++) {
        for (int j = 0; j < UNROLL; j++)
          outp[(i + j * VEC_SIZE) * sdst] = vgetq_lane_f64(result[j], i);
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
      float64x2_t result = neon_log2_inline_core(u, u_off);

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
      uint64_t arr_nan[2], arr_pinf[2], arr_ninf[2], arr_neg[2], arr_zero[2];
      vst1q_u64(arr_nan, is_nan);
      vst1q_u64(arr_pinf, is_pos_inf);
      vst1q_u64(arr_ninf, is_neg_inf);
      vst1q_u64(arr_neg, is_neg);
      vst1q_u64(arr_zero, is_zero);
      const npy_intp ds = (sdst == 1) ? 1 : sdst;
      for (int i = 0; i < current_len; i++) {
        double val = (ssrc == 1) ? infp[i] : infp[i * ssrc];
        double r;
        if      (arr_nan[i])  r = val;
        else if (arr_pinf[i]) r = NPY_INFINITY;
        else if (arr_ninf[i]) r = NPY_NAN;
        else if (arr_zero[i]) r = -NPY_INFINITY;
        else if (arr_neg[i])  r = -NPY_NAN;
        else                  r = npy_log2(val);
        outp[i * ds] = r;
      }
    }

    infp += ssrc * current_len;
    outp += sdst * current_len;
    remaining -= current_len;
  }
  feclearexcept(FE_ALL_EXCEPT);
}

static inline float32x4_t
neon_log2f_inline_core_preloaded(uint32x4_t u_off, float32x4_t n,
                                  float32x4_t c0, float32x4_t c2, float32x4_t c4,
                                  float32x4_t c6, float32x4_t c8, float32x4_t c1357)
{
  const uint32x4_t off = vdupq_n_u32(0x3f2aaaab);
  const uint32x4_t mantissa_mask = vdupq_n_u32(0x007fffff);

  uint32x4_t u = vaddq_u32(vandq_u32(u_off, mantissa_mask), off);
  float32x4_t r = vsubq_f32(vreinterpretq_f32_u32(u), vdupq_n_f32(1.0f));

  float32x4_t r2 = vmulq_f32(r, r);

  float32x4_t c01 = vfmaq_laneq_f32(c0, r, c1357, 0);
  float32x4_t c23 = vfmaq_laneq_f32(c2, r, c1357, 1);
  float32x4_t c45 = vfmaq_laneq_f32(c4, r, c1357, 2);
  float32x4_t c67 = vfmaq_laneq_f32(c6, r, c1357, 3);
  float32x4_t p68 = vfmaq_f32(c67, r2, c8);
  float32x4_t p48 = vfmaq_f32(c45, r2, p68);
  float32x4_t p28 = vfmaq_f32(c23, r2, p48);
  float32x4_t p = vfmaq_f32(c01, r2, p28);

  return vfmaq_f32(n, p, r);
}


static void __attribute__((optimize("O3")))
simd_log2_neon_FLOAT(const npy_float *src, npy_intp ssrc,
                      npy_float *dst, npy_intp sdst, npy_intp len)
{
  const float32x4_t c0 = vdupq_n_f32(0x1.715476p0f);
  const float32x4_t c2 = vdupq_n_f32(0x1.ec701cp-2f);
  const float32x4_t c4 = vdupq_n_f32(0x1.27a0b8p-2f);
  const float32x4_t c6 = vdupq_n_f32(0x1.9d8ecap-3f);
  const float32x4_t c8 = vdupq_n_f32(0x1.9e495p-3f);
  static const float c1357_arr[4] __attribute__((aligned(16))) = {
    -0x1.715458p-1f, -0x1.7171a4p-2f, -0x1.e5143ep-3f, -0x1.c675bp-3f
  };
  const uint32x4_t off = vdupq_n_u32(0x3f2aaaab);
  const float32x4_t zeros_f = vdupq_n_f32(0.0f);
  const float32x4_t ones_f = vdupq_n_f32(1.0f);
  const float32x4_t pos_inf = vdupq_n_f32(NPY_INFINITYF);
  const float32x4_t neg_inf_val = vdupq_n_f32(-NPY_INFINITYF);
  const float32x4_t nan_val = vdupq_n_f32(NPY_NANF);
  const float32x4_t neg_nan = vdupq_n_f32(-NPY_NANF);
  const float32x4_t c1357 = vld1q_f32(c1357_arr);

  const npy_float *infp = src;
  npy_float *outp = dst;
  npy_intp remaining = len;

  feclearexcept(FE_ALL_EXCEPT);

  const npy_intp UNROLL = 4;
  const npy_intp VEC_SIZE = 4;

  while (remaining >= UNROLL * VEC_SIZE) {
    float32x4_t x[UNROLL], result[UNROLL];
    uint32x4_t u[UNROLL], is_nan[UNROLL], is_pos_inf[UNROLL], is_neg_inf[UNROLL];
    uint32x4_t is_zero[UNROLL], is_neg[UNROLL], full_mask[UNROLL];

    if (ssrc == 1) {
      __builtin_prefetch(infp + 128, 0, 3);
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

    for (int i = 0; i < UNROLL; i++) u[i] = vreinterpretq_u32_f32(x[i]);

    for (int i = 0; i < UNROLL; i++) {
      is_nan[i] = vmvnq_u32(vceqq_f32(x[i], x[i]));
      is_pos_inf[i] = vceqq_f32(x[i], pos_inf);
      is_neg_inf[i] = vceqq_f32(x[i], neg_inf_val);
      is_zero[i] = vceqq_f32(x[i], zeros_f);
      is_neg[i] = vcltq_f32(x[i], zeros_f);
      is_neg[i] = vbicq_u32(is_neg[i], vorrq_u32(is_neg_inf[i], is_zero[i]));
      uint32x4_t special_mask = vorrq_u32(vorrq_u32(is_nan[i], is_pos_inf[i]),
                                           vorrq_u32(is_neg_inf[i], is_zero[i]));
      full_mask[i] = vorrq_u32(special_mask, is_neg[i]);
      float32x4_t safe = vbslq_f32(full_mask[i], ones_f, x[i]);
      u[i] = vreinterpretq_u32_f32(safe);
    }

    for (int i = 0; i < UNROLL; i++) {
      uint32x4_t u_off = vsubq_u32(u[i], off);
      float32x4_t n = vcvtq_f32_s32(vshrq_n_s32(vreinterpretq_s32_u32(u_off), 23));
      result[i] = neon_log2f_inline_core_preloaded(u_off, n, c0, c2, c4, c6, c8, c1357);
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
      for (int i = 0; i < VEC_SIZE; i++) {
        for (int j = 0; j < UNROLL; j++)
          outp[(i + j * VEC_SIZE) * sdst] = vgetq_lane_f32(result[j], i);
      }
    }

    infp += ssrc * UNROLL * VEC_SIZE;
    outp += sdst * UNROLL * VEC_SIZE;
    remaining -= UNROLL * VEC_SIZE;
  }

  while (remaining > 0) {
    int current_len = (remaining < VEC_SIZE) ? remaining : VEC_SIZE;
    float32x4_t x;
    if (ssrc == 1 && current_len == VEC_SIZE) {
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
      float32x4_t result = neon_log2f_inline_core_preloaded(u_off, n, c0, c2, c4, c6, c8, c1357);

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
      uint32_t arr_nan[4], arr_pinf[4], arr_ninf[4], arr_neg[4], arr_zero[4];
      vst1q_u32(arr_nan, is_nan);
      vst1q_u32(arr_pinf, is_pos_inf);
      vst1q_u32(arr_ninf, is_neg_inf);
      vst1q_u32(arr_neg, is_neg);
      vst1q_u32(arr_zero, is_zero);
      const npy_intp ds = (sdst == 1) ? 1 : sdst;
      for (int k = 0; k < current_len; k++) {
        float val = infp[k * ssrc];
        float r;
        if      (arr_nan[k])  r = val;
        else if (arr_pinf[k]) r = NPY_INFINITYF;
        else if (arr_ninf[k]) r = NPY_NANF;
        else if (arr_zero[k]) r = -NPY_INFINITYF;
        else if (arr_neg[k])  r = -NPY_NANF;
        else                  r = npy_log2f(val);
        outp[k * ds] = r;
      }
    }

    infp += ssrc * current_len;
    outp += sdst * current_len;
    remaining -= current_len;
  }
  feclearexcept(FE_ALL_EXCEPT);
}

static inline void
simd_log2_neon_HALF_stride1(const npy_half *src, npy_half *dst, npy_intp len)
{
  const uint32x4_t off_v = vdupq_n_u32(0x3f2aaaab);
  const float32x4_t zeros_v = vdupq_n_f32(0.0f);
  const float32x4_t ones_v = vdupq_n_f32(1.0f);
  const float32x4_t pos_inf_v = vdupq_n_f32(NPY_INFINITYF);
  const float32x4_t neg_inf_v = vdupq_n_f32(-NPY_INFINITYF);
  const float32x4_t nan_v = vdupq_n_f32(NPY_NANF);
  const float32x4_t neg_nan_v = vdupq_n_f32(-NPY_NANF);

  const float32x4_t c0 = vdupq_n_f32(0x1.715476p0f);
  const float32x4_t c2 = vdupq_n_f32(0x1.ec701cp-2f);
  const float32x4_t c4 = vdupq_n_f32(0x1.27a0b8p-2f);
  const float32x4_t c6 = vdupq_n_f32(0x1.9d8ecap-3f);
  const float32x4_t c8 = vdupq_n_f32(0x1.9e495p-3f);
  static const float c1357_arr[4] __attribute__((aligned(16))) = {
    -0x1.715458p-1f, -0x1.7171a4p-2f, -0x1.e5143ep-3f, -0x1.c675bp-3f
  };
  const float32x4_t c1357 = vld1q_f32(c1357_arr);

  const npy_half *infp = src;
  npy_half *outp = dst;
  npy_intp remaining = len;

  feclearexcept(FE_ALL_EXCEPT);

  while (remaining >= 16) {
    __builtin_prefetch(infp + 64, 0, 3);
    __builtin_prefetch(outp + 64, 1, 3);

    float16x8_t h0 = vld1q_f16((const float16_t *)infp);
    float16x8_t h1 = vld1q_f16((const float16_t *)(infp + 8));

    float32x4_t x[4], sf[4], res[4];
    uint32x4_t isn[4], isp[4], isni[4], isz[4], isneg[4], full[4], u_off[4];
    float32x4_t n_val[4];

    x[0] = vcvt_f32_f16(vget_low_f16(h0));
    x[1] = vcvt_f32_f16(vget_high_f16(h0));
    x[2] = vcvt_f32_f16(vget_low_f16(h1));
    x[3] = vcvt_f32_f16(vget_high_f16(h1));

    for (int k = 0; k < 4; k++) {
      isn[k] = vmvnq_u32(vceqq_f32(x[k], x[k]));
      isp[k] = vceqq_f32(x[k], pos_inf_v);
      isni[k] = vceqq_f32(x[k], neg_inf_v);
      isz[k] = vceqq_f32(x[k], zeros_v);
      isneg[k] = vbicq_u32(vcltq_f32(x[k], zeros_v), vorrq_u32(isni[k], isz[k]));
      full[k] = vorrq_u32(vorrq_u32(vorrq_u32(isn[k], isp[k]), vorrq_u32(isni[k], isz[k])), isneg[k]);
      sf[k] = vbslq_f32(full[k], ones_v, x[k]);
      u_off[k] = vsubq_u32(vreinterpretq_u32_f32(sf[k]), off_v);
      n_val[k] = vcvtq_f32_s32(vshrq_n_s32(vreinterpretq_s32_u32(u_off[k]), 23));
      res[k] = neon_log2f_inline_core_preloaded(u_off[k], n_val[k], c0, c2, c4, c6, c8, c1357);
      res[k] = vbslq_f32(isn[k], x[k], res[k]);
      res[k] = vbslq_f32(isp[k], pos_inf_v, res[k]);
      res[k] = vbslq_f32(isni[k], nan_v, res[k]);
      res[k] = vbslq_f32(isz[k], neg_inf_v, res[k]);
      res[k] = vbslq_f32(isneg[k], neg_nan_v, res[k]);
    }

    float16x4_t rh[4];
    for (int k = 0; k < 4; k++)
      rh[k] = vcvt_f16_f32(res[k]);
    for (int k = 0; k < 2; k++)
      vst1q_f16((float16_t *)(outp + k * 8), vcombine_f16(rh[k*2], rh[k*2+1]));

    infp += 16;
    outp += 16;
    remaining -= 16;
  }

  while (remaining >= 8) {
    float16x8_t h0 = vld1q_f16((const float16_t *)infp);
    float32x4_t x[2], sf[2], res[2];
    uint32x4_t isn[2], isp[2], isni[2], isz[2], isneg[2], full[2], u_off[2];
    float32x4_t n_val[2];

    x[0] = vcvt_f32_f16(vget_low_f16(h0));
    x[1] = vcvt_f32_f16(vget_high_f16(h0));

    for (int k = 0; k < 2; k++) {
      isn[k] = vmvnq_u32(vceqq_f32(x[k], x[k]));
      isp[k] = vceqq_f32(x[k], pos_inf_v);
      isni[k] = vceqq_f32(x[k], neg_inf_v);
      isz[k] = vceqq_f32(x[k], zeros_v);
      isneg[k] = vbicq_u32(vcltq_f32(x[k], zeros_v), vorrq_u32(isni[k], isz[k]));
      full[k] = vorrq_u32(vorrq_u32(vorrq_u32(isn[k], isp[k]), vorrq_u32(isni[k], isz[k])), isneg[k]);
      sf[k] = vbslq_f32(full[k], ones_v, x[k]);
      u_off[k] = vsubq_u32(vreinterpretq_u32_f32(sf[k]), off_v);
      n_val[k] = vcvtq_f32_s32(vshrq_n_s32(vreinterpretq_s32_u32(u_off[k]), 23));
      res[k] = neon_log2f_inline_core_preloaded(u_off[k], n_val[k], c0, c2, c4, c6, c8, c1357);
      res[k] = vbslq_f32(isn[k], x[k], res[k]);
      res[k] = vbslq_f32(isp[k], pos_inf_v, res[k]);
      res[k] = vbslq_f32(isni[k], nan_v, res[k]);
      res[k] = vbslq_f32(isz[k], neg_inf_v, res[k]);
      res[k] = vbslq_f32(isneg[k], neg_nan_v, res[k]);
    }

    float16x4_t rh[2];
    for (int k = 0; k < 2; k++)
      rh[k] = vcvt_f16_f32(res[k]);
    vst1q_f16((float16_t *)outp, vcombine_f16(rh[0], rh[1]));

    infp += 8;
    outp += 8;
    remaining -= 8;
  }

  for (npy_intp i = 0; i < remaining; i++) {
    outp[i] = npy_float_to_half(npy_log2f(npy_half_to_float(infp[i])));
  }

  feclearexcept(FE_ALL_EXCEPT);
}

static void
simd_log2_neon_HALF(const npy_half *src, npy_intp ssrc,
                    npy_half *dst, npy_intp sdst, npy_intp len)
{
  if (ssrc == 1 && sdst == 1) {
    simd_log2_neon_HALF_stride1(src, dst, len);
    return;
  }
  const npy_intp BLOCK_SIZE = 8192;
  const npy_intp VEC_SIZE = 8;
  const npy_intp UNROLL = 4;
  
  npy_float *tmp_src = (npy_float *)malloc(BLOCK_SIZE * sizeof(npy_float));
  npy_float *tmp_dst = (npy_float *)malloc(BLOCK_SIZE * sizeof(npy_float));
  
    if (!tmp_src || !tmp_dst) {
      if (tmp_src) free(tmp_src);
      if (tmp_dst) free(tmp_dst);
      feclearexcept(FE_ALL_EXCEPT);
      for (npy_intp i = 0; i < len; i++) {
        npy_float in1 = npy_half_to_float(src[i * ssrc]);
        dst[i * sdst] = npy_float_to_half(npy_log2f(in1));
      }
    return;
  }

  for (npy_intp offset = 0; offset < len; offset += BLOCK_SIZE) {
    npy_intp block_len = (len - offset > BLOCK_SIZE) ? BLOCK_SIZE : (len - offset);
    const npy_half *block_src = src + offset * ssrc;
    npy_half *block_dst = dst + offset * sdst;
    
    npy_intp i = 0;
    if (ssrc == 1) {
      for (; i + UNROLL * VEC_SIZE <= block_len; i += UNROLL * VEC_SIZE) {
        __builtin_prefetch(block_src + i + 256, 0, 3);
        __builtin_prefetch(block_src + i + 512, 0, 3);
        float16x8_t h[4];
        for (int u = 0; u < 4; u++)
          h[u] = vld1q_f16((const float16_t *)(block_src + i + u * 8));
        for (int u = 0; u < 4; u++) {
          vst1q_f32(tmp_src + i + u * 8, vcvt_f32_f16(vget_low_f16(h[u])));
          vst1q_f32(tmp_src + i + u * 8 + 4, vcvt_f32_f16(vget_high_f16(h[u])));
        }
      }
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
        __builtin_prefetch(block_src + i + 64, 0, 3);
        float16x8_t h0 = vld1q_f16((const float16_t *)(block_src + i));
        vst1q_f32(tmp_src + i, vcvt_f32_f16(vget_low_f16(h0)));
        vst1q_f32(tmp_src + i + 4, vcvt_f32_f16(vget_high_f16(h0)));
      }
    } else {
      for (; i + UNROLL * VEC_SIZE <= block_len; i += UNROLL * VEC_SIZE) {
        for (int j = 0; j < UNROLL * VEC_SIZE; j++) {
          tmp_src[i + j] = npy_half_to_float(block_src[(i + j) * ssrc]);
        }
      }
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
        for (int j = 0; j < VEC_SIZE; j++) {
          tmp_src[i + j] = npy_half_to_float(block_src[(i + j) * ssrc]);
        }
      }
    }
    for (; i < block_len; i++) {
      tmp_src[i] = npy_half_to_float(block_src[i * ssrc]);
    }
    
    simd_log2_neon_FLOAT(tmp_src, 1, tmp_dst, 1, block_len);
    
    i = 0;
    if (sdst == 1) {
      for (; i + UNROLL * VEC_SIZE <= block_len; i += UNROLL * VEC_SIZE) {
        __builtin_prefetch(block_dst + i + 256, 1, 3);
        __builtin_prefetch(block_dst + i + 512, 1, 3);
        float32x4_t f[8];
        for (int u = 0; u < 8; u++)
          f[u] = vld1q_f32(tmp_dst + i + u * 4);
        float16x4_t h[8];
        for (int u = 0; u < 8; u++)
          h[u] = vcvt_f16_f32(f[u]);
        for (int u = 0; u < 4; u++)
          vst1q_f16((float16_t *)(block_dst + i + u * 8), vcombine_f16(h[u*2], h[u*2+1]));
      }
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
        __builtin_prefetch(block_dst + i + 64, 1, 3);
        float32x4_t f0 = vld1q_f32(tmp_dst + i);
        float32x4_t f1 = vld1q_f32(tmp_dst + i + 4);
        vst1q_f16((float16_t *)(block_dst + i), vcombine_f16(vcvt_f16_f32(f0), vcvt_f16_f32(f1)));
      }
    } else {
      for (; i + UNROLL * VEC_SIZE <= block_len; i += UNROLL * VEC_SIZE) {
        for (int j = 0; j < UNROLL * VEC_SIZE; j++) {
          block_dst[(i + j) * sdst] = npy_float_to_half(tmp_dst[i + j]);
        }
      }
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
        for (int j = 0; j < VEC_SIZE; j++) {
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
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_float in1 = *(npy_float *)ip1;
            *(npy_float *)op1 = npy_log2f(in1);
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_float *src = (npy_float*)args[0];
        npy_float *dst = (npy_float*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);
        simd_log2_neon_FLOAT(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            simd_log2_neon_FLOAT((npy_float *)ip1, 1, (npy_float *)op1, 1, 1);
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
#else
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
#endif
}
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_log2)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_double in1 = *(npy_double *)ip1;
            *(npy_double *)op1 = npy_log2(in1);
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_double *src = (npy_double*)args[0];
        npy_double *dst = (npy_double*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_double);
        const npy_intp sdst = steps[1] / sizeof(npy_double);
        simd_log2_neon_DOUBLE(src, ssrc, dst, sdst, len);
    }
    else {
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
    UNARY_LOOP {
        const npy_double in1 = *(npy_double *)ip1;
        *(npy_double *)op1 = npy_log2(in1);
    }
#else
    UNARY_LOOP {
        const npy_double in1 = *(npy_double *)ip1;
        *(npy_double *)op1 = npy_log2(in1);
    }
#endif
}
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_log2)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
#if SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_log2f(in1));
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_half *src = (npy_half*)args[0];
        npy_half *dst = (npy_half*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_half);
        const npy_intp sdst = steps[1] / sizeof(npy_half);
        simd_log2_neon_HALF(src, ssrc, dst, sdst, len);
    }
    else {
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
