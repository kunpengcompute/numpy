#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "loops_explog.h"
#include "loops.h"

#if SIMD_ARM

static inline float32x4_t __attribute__((always_inline))
neon_exp2f_inline_core(float32x4_t x,
                        float32x4_t shift, uint32x4_t exponent_bias,
                        float32x4_t c0, float32x4_t c1, float32x4_t c2,
                        float32x4_t c3, float32x4_t c4)
{
  float32x4_t z = vaddq_f32(x, shift);
  float32x4_t n = vsubq_f32(z, shift);
  float32x4_t r = vsubq_f32(x, n);
  int32x4_t ni = vcvtq_s32_f32(n);
  uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_s32(ni), 23);
  float32x4_t scale = vreinterpretq_f32_u32(vaddq_u32(e, exponent_bias));
  float32x4_t r2 = vmulq_f32(r, r);
  float32x4_t p = vfmaq_f32(c1, r, c0);
  float32x4_t q = vfmaq_f32(c3, r, c2);
  q = vfmaq_f32(q, p, r2);
  float32x4_t poly = vfmaq_f32(vmulq_f32(c4, r), q, r2);
  return vfmaq_f32(scale, poly, scale);
}

static void
simd_exp2_neon_FLOAT(const npy_float *src, npy_intp ssrc,
                     npy_float *dst, npy_intp sdst, npy_intp len)
{
  const float32x4_t pos_inf = vdupq_n_f32(NPY_INFINITYF);
  const float32x4_t neg_inf = vdupq_n_f32(-NPY_INFINITYF);
  const float32x4_t zero_val = vdupq_n_f32(0.0f);
  const float32x4_t shift = vdupq_n_f32(0x1.8p+23f);
  const uint32x4_t exponent_bias = vdupq_n_u32(0x3f800000);
  const float32x4_t exp_max = vdupq_n_f32(128.0f);
  const float32x4_t exp_min = vdupq_n_f32(-126.0f);
  const float32x4_t c0 = vdupq_n_f32(0x1.5f74aep-10f);
  const float32x4_t c1 = vdupq_n_f32(0x1.3b2c9cp-7f);
  const float32x4_t c2 = vdupq_n_f32(0x1.c6b08ep-5f);
  const float32x4_t c3 = vdupq_n_f32(0x1.ebfbecep-3f);
  const float32x4_t c4 = vdupq_n_f32(0x1.62e42fep-1f);

  const npy_float *infp = src;
  npy_float *outp = dst;
  npy_intp remaining = len;

  feclearexcept(FE_ALL_EXCEPT);

  const npy_intp UNROLL = 8;
  const npy_intp VEC_SIZE = 4;

  while (remaining >= UNROLL * VEC_SIZE) {
    float32x4_t x[UNROLL], result[UNROLL];
    uint32x4_t is_nan[UNROLL], is_overflow[UNROLL], is_underflow[UNROLL];

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

    const float32x4_t safe_val = vdupq_n_f32(126.0f);
    for (int i = 0; i < UNROLL; i++) {
      is_nan[i] = vmvnq_u32(vceqq_f32(x[i], x[i]));
      is_overflow[i] = vcgtq_f32(x[i], exp_max);
      is_underflow[i] = vcltq_f32(x[i], exp_min);
      uint32x4_t need_special = vorrq_u32(vorrq_u32(vorrq_u32(is_overflow[i], is_underflow[i]),
                              vceqq_f32(x[i], pos_inf)), vceqq_f32(x[i], neg_inf));
      need_special = vorrq_u32(need_special, is_nan[i]);
      float32x4_t safe_x = vbslq_f32(need_special, safe_val, x[i]);
      result[i] = neon_exp2f_inline_core(safe_x, shift, exponent_bias, c0, c1, c2, c3, c4);
      result[i] = vbslq_f32(is_nan[i], x[i], result[i]);
      result[i] = vbslq_f32(vceqq_f32(x[i], pos_inf), pos_inf, result[i]);
      result[i] = vbslq_f32(vceqq_f32(x[i], neg_inf), zero_val, result[i]);
      result[i] = vbslq_f32(is_overflow[i], pos_inf, result[i]);
      result[i] = vbslq_f32(is_underflow[i], zero_val, result[i]);
    }

    if (sdst == 1) {
      __builtin_prefetch(outp + 128, 1, 3);
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
      float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int i = 0; i < current_len; i++) vals[i] = infp[i * ssrc];
      x = vld1q_f32(vals);
    }

    uint32x4_t is_nan = vceqq_f32(x, x);
    is_nan = vmvnq_u32(is_nan);
    uint32x4_t is_overflow = vcgtq_f32(x, exp_max);
    uint32x4_t is_underflow = vcltq_f32(x, exp_min);
    uint32x4_t need_special = vorrq_u32(vorrq_u32(vorrq_u32(is_overflow, is_underflow), vceqq_f32(x, pos_inf)), vceqq_f32(x, neg_inf));
    need_special = vorrq_u32(need_special, is_nan);

    const float32x4_t safe_val = vdupq_n_f32(126.0f);
    float32x4_t safe = vbslq_f32(need_special, safe_val, x);
    float32x4_t result = neon_exp2f_inline_core(safe, shift, exponent_bias, c0, c1, c2, c3, c4);

    result = vbslq_f32(is_nan, x, result);
    result = vbslq_f32(vceqq_f32(x, pos_inf), pos_inf, result);
    result = vbslq_f32(vceqq_f32(x, neg_inf), zero_val, result);
    result = vbslq_f32(is_overflow, pos_inf, result);
    result = vbslq_f32(is_underflow, zero_val, result);

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

static inline float64x2_t __attribute__((always_inline))
neon_exp2_inline_core_f64(float64x2_t x,
                           float64x2_t shift,
                           float64x2_t poly0, float64x2_t poly1,
                           float64x2_t poly2, float64x2_t poly3)
{
  float64x2_t z = vaddq_f64(shift, x);
  uint64x2_t u = vreinterpretq_u64_f64(z);
  float64x2_t n = vsubq_f64(z, shift);
  float64x2_t r = vsubq_f64(x, n);
  uint64x2_t e = vshlq_n_u64(u, 52 - V_EXP_TABLE_BITS);
  uint64x2_t tab = neon_exp_lookup_sbits(u);
  float64x2_t scale = vreinterpretq_f64_u64(vaddq_u64(tab, e));
  float64x2_t r2 = vmulq_f64(r, r);
  float64x2_t p = vfmaq_f64(poly0, r, poly1);
  float64x2_t q = vfmaq_f64(poly2, r, poly3);
  float64x2_t pv = vfmaq_f64(vmulq_f64(r, p), r2, q);
  return vfmaq_f64(scale, scale, pv);
}

static void
simd_exp2_neon_DOUBLE(const npy_double *src, npy_intp ssrc,
                      npy_double *dst, npy_intp sdst, npy_intp len)
{
  const float64x2_t pos_inf = vdupq_n_f64(NPY_INFINITY);
  const float64x2_t neg_inf = vdupq_n_f64(-NPY_INFINITY);
  const float64x2_t zero_val = vdupq_n_f64(0.0);
  const float64x2_t special_bound = vdupq_n_f64(1022.0);
  const float64x2_t shift = vdupq_n_f64(0x1.8p+52 / V_EXP_N);
  const float64x2_t poly0 = vdupq_n_f64(0x1.62e42fefa3686p-1);
  const float64x2_t poly1 = vdupq_n_f64(0x1.ebfbdff82c241p-3);
  const float64x2_t poly2 = vdupq_n_f64(0x1.c6b09b16de99ap-5);
  const float64x2_t poly3 = vdupq_n_f64(0x1.3b2abf5571ad8p-7);
  const uint64x2_t all_bits = vdupq_n_u64(0xFFFFFFFFFFFFFFFFULL);

  const npy_double *infp = src;
  npy_double *outp = dst;
  npy_intp remaining = len;

  feclearexcept(FE_ALL_EXCEPT);

  const npy_intp UNROLL = 8;
  const npy_intp VEC_SIZE = 2;

  while (remaining >= UNROLL * VEC_SIZE) {
    float64x2_t x[UNROLL], result[UNROLL];
    uint64x2_t need_special[UNROLL];

    if (ssrc == 1) {
      __builtin_prefetch(infp + 64, 0, 3);
      for (int i = 0; i < UNROLL; i++)
        x[i] = vld1q_f64(infp + i * VEC_SIZE);
    } else {
      for (int i = 0; i < UNROLL; i++) {
        double vals[VEC_SIZE];
        for (int j = 0; j < VEC_SIZE; j++)
          vals[j] = infp[(i * VEC_SIZE + j) * ssrc];
        x[i] = vld1q_f64(vals);
      }
    }

    for (int i = 0; i < UNROLL; i++) {
      need_special[i] = vcagtq_f64(x[i], special_bound);
      result[i] = neon_exp2_inline_core_f64(x[i], shift, poly0, poly1, poly2, poly3);
      if (neon_has_any_lane_u64(need_special[i])) {
        uint64x2_t is_nan = vceqq_f64(x[i], x[i]);
        is_nan = veorq_u64(is_nan, all_bits);
        result[i] = vbslq_f64(is_nan, x[i], result[i]);
        result[i] = vbslq_f64(vceqq_f64(x[i], pos_inf), pos_inf, result[i]);
        result[i] = vbslq_f64(vceqq_f64(x[i], neg_inf), zero_val, result[i]);
      }
    }

    if (sdst == 1) {
      __builtin_prefetch(outp + 64, 1, 3);
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
      x = vld1q_f64(infp);
    } else {
      double vals[2] = {0.0, 0.0};
      for (int i = 0; i < current_len; i++) vals[i] = infp[i * ssrc];
      x = vld1q_f64(vals);
    }

    uint64x2_t need_special = vcagtq_f64(x, special_bound);
    float64x2_t result = neon_exp2_inline_core_f64(x, shift, poly0, poly1, poly2, poly3);

    if (neon_has_any_lane_u64(need_special)) {
      uint64x2_t is_nan = vceqq_f64(x, x);
      is_nan = veorq_u64(is_nan, all_bits);
      result = vbslq_f64(is_nan, x, result);
      result = vbslq_f64(vceqq_f64(x, pos_inf), pos_inf, result);
      result = vbslq_f64(vceqq_f64(x, neg_inf), zero_val, result);
    }

    if (sdst == 1) {
      if (current_len == 1) vst1q_lane_f64(outp, result, 0);
      else vst1q_f64(outp, result);
    } else {
      for (int i = 0; i < current_len; i++) outp[i * sdst] = vgetq_lane_f64(result, i);
    }

    infp += ssrc * current_len;
    outp += sdst * current_len;
    remaining -= current_len;
  }
  feclearexcept(FE_ALL_EXCEPT);
}

static inline void
simd_exp2_neon_HALF_stride2(const npy_half *src, npy_intp ssrc,
                             npy_half *dst, npy_intp sdst, npy_intp len)
{
  const float32x4_t pos_inf_v = vdupq_n_f32(NPY_INFINITYF);
  const float32x4_t neg_inf_v = vdupq_n_f32(-NPY_INFINITYF);
  const float32x4_t zero_v = vdupq_n_f32(0.0f);
  const float32x4_t shift_v = vdupq_n_f32(0x1.8p+23f);
  const uint32x4_t bias_v = vdupq_n_u32(0x3f800000);
  const float32x4_t exp_max_v = vdupq_n_f32(15.0f);
  const float32x4_t exp_min_v = vdupq_n_f32(-14.0f);
  const float32x4_t c0_v = vdupq_n_f32(0x1.5f74aep-10f);
  const float32x4_t c1_v = vdupq_n_f32(0x1.3b2c9cp-7f);
  const float32x4_t c2_v = vdupq_n_f32(0x1.c6b08ep-5f);
  const float32x4_t c3_v = vdupq_n_f32(0x1.ebfbecep-3f);
  const float32x4_t c4_v = vdupq_n_f32(0x1.62e42fep-1f);
  const float32x4_t safe_v = vdupq_n_f32(14.0f);

  const npy_half *infp = src;
  npy_half *outp = dst;
  npy_intp remaining = len;

  feclearexcept(FE_ALL_EXCEPT);

  while (remaining >= 16) {
    __builtin_prefetch(infp + ssrc * 64, 0, 3);
    __builtin_prefetch(outp + sdst * 64, 1, 3);

    float32x4_t x[4], result[4];
    for (int k = 0; k < 4; k++) {
      float16_t h_arr[4];
      for (int j = 0; j < 4; j++)
        h_arr[j] = (float16_t)infp[(k * 4 + j) * ssrc];
      x[k] = vcvt_f32_f16(vld1_f16(h_arr));
    }

    for (int k = 0; k < 4; k++) {
      uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x[k], x[k]));
      uint32x4_t is_overflow = vcgtq_f32(x[k], exp_max_v);
      uint32x4_t is_underflow = vcltq_f32(x[k], exp_min_v);
      uint32x4_t need_special = vorrq_u32(
          vorrq_u32(vorrq_u32(is_overflow, is_underflow),
                    vceqq_f32(x[k], pos_inf_v)),
          vceqq_f32(x[k], neg_inf_v));
      need_special = vorrq_u32(need_special, is_nan);

      float32x4_t safe = vbslq_f32(need_special, safe_v, x[k]);
      result[k] = neon_exp2f_inline_core(safe, shift_v, bias_v,
                                            c0_v, c1_v, c2_v, c3_v, c4_v);

      result[k] = vbslq_f32(is_nan, x[k], result[k]);
      result[k] = vbslq_f32(vceqq_f32(x[k], pos_inf_v), pos_inf_v, result[k]);
      result[k] = vbslq_f32(vceqq_f32(x[k], neg_inf_v), zero_v, result[k]);
      result[k] = vbslq_f32(is_overflow, pos_inf_v, result[k]);
      result[k] = vbslq_f32(is_underflow, zero_v, result[k]);
    }

    if (sdst == 2) {
      for (int k = 0; k < 4; k++) {
        float16x4_t rh = vcvt_f16_f32(result[k]);
        float16x4_t z = vdup_n_f16(0);
        float16x4x2_t zipped = vzip_f16(rh, z);
        vst1q_f16((float16_t *)(outp + k * 8), vcombine_f16(zipped.val[0], zipped.val[1]));
      }
    } else {
      for (int k = 0; k < 4; k++) {
        float16x4_t rh = vcvt_f16_f32(result[k]);
        for (int j = 0; j < 4; j++)
          outp[(k * 4 + j) * sdst] = vget_lane_f16(rh, j);
      }
    }

    infp += ssrc * 16;
    outp += sdst * 16;
    remaining -= 16;
  }

  while (remaining >= 4) {
    float16_t h_arr[4];
    for (int j = 0; j < 4; j++)
      h_arr[j] = (float16_t)infp[j * ssrc];
    float32x4_t x = vcvt_f32_f16(vld1_f16(h_arr));

    uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x, x));
    uint32x4_t is_overflow = vcgtq_f32(x, exp_max_v);
    uint32x4_t is_underflow = vcltq_f32(x, exp_min_v);
    uint32x4_t need_special = vorrq_u32(
        vorrq_u32(vorrq_u32(is_overflow, is_underflow),
                  vceqq_f32(x, pos_inf_v)),
        vceqq_f32(x, neg_inf_v));
    need_special = vorrq_u32(need_special, is_nan);

    float32x4_t safe = vbslq_f32(need_special, safe_v, x);
    float32x4_t result = neon_exp2f_inline_core(safe, shift_v, bias_v,
                                                  c0_v, c1_v, c2_v, c3_v, c4_v);

    result = vbslq_f32(is_nan, x, result);
    result = vbslq_f32(vceqq_f32(x, pos_inf_v), pos_inf_v, result);
    result = vbslq_f32(vceqq_f32(x, neg_inf_v), zero_v, result);
    result = vbslq_f32(is_overflow, pos_inf_v, result);
    result = vbslq_f32(is_underflow, zero_v, result);

    float16x4_t rh = vcvt_f16_f32(result);
    if (sdst == 2) {
      float16x4_t z = vdup_n_f16(0);
      float16x4x2_t zipped = vzip_f16(rh, z);
      vst1q_f16((float16_t *)outp, vcombine_f16(zipped.val[0], zipped.val[1]));
    } else {
      for (int j = 0; j < 4; j++)
        outp[j * sdst] = vget_lane_f16(rh, j);
    }

    infp += ssrc * 4;
    outp += sdst * 4;
    remaining -= 4;
  }

  for (npy_intp i = 0; i < remaining; i++) {
    float16_t h_val = (float16_t)(*infp);
    float32x4_t x = vcvt_f32_f16(vdup_n_f16(h_val));

    uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x, x));
    uint32x4_t is_overflow = vcgtq_f32(x, exp_max_v);
    uint32x4_t is_underflow = vcltq_f32(x, exp_min_v);
    uint32x4_t need_special = vorrq_u32(
        vorrq_u32(vorrq_u32(is_overflow, is_underflow),
                  vceqq_f32(x, pos_inf_v)),
        vceqq_f32(x, neg_inf_v));
    need_special = vorrq_u32(need_special, is_nan);

    float32x4_t safe = vbslq_f32(need_special, safe_v, x);
    float32x4_t result = neon_exp2f_inline_core(safe, shift_v, bias_v,
                                                  c0_v, c1_v, c2_v, c3_v, c4_v);
    result = vbslq_f32(is_nan, x, result);
    result = vbslq_f32(vceqq_f32(x, pos_inf_v), pos_inf_v, result);
    result = vbslq_f32(vceqq_f32(x, neg_inf_v), zero_v, result);
    result = vbslq_f32(is_overflow, pos_inf_v, result);
    result = vbslq_f32(is_underflow, zero_v, result);

    float16x4_t rh = vcvt_f16_f32(result);
    *outp = (npy_half)vget_lane_f16(rh, 0);
    infp += ssrc;
    outp += sdst;
  }

  feclearexcept(FE_ALL_EXCEPT);
}

static inline void
simd_exp2_neon_HALF_stride1(const npy_half *src, npy_half *dst, npy_intp len)
{
  const float32x4_t pos_inf_v = vdupq_n_f32(NPY_INFINITYF);
  const float32x4_t neg_inf_v = vdupq_n_f32(-NPY_INFINITYF);
  const float32x4_t zero_v = vdupq_n_f32(0.0f);
  const float32x4_t shift_v = vdupq_n_f32(0x1.8p+23f);
  const uint32x4_t bias_v = vdupq_n_u32(0x3f800000);
  const float32x4_t exp_max_v = vdupq_n_f32(128.0f);
  const float32x4_t exp_min_v = vdupq_n_f32(-126.0f);
  const float32x4_t c0_v = vdupq_n_f32(0x1.5f74aep-10f);
  const float32x4_t c1_v = vdupq_n_f32(0x1.3b2c9cp-7f);
  const float32x4_t c2_v = vdupq_n_f32(0x1.c6b08ep-5f);
  const float32x4_t c3_v = vdupq_n_f32(0x1.ebfbecep-3f);
  const float32x4_t c4_v = vdupq_n_f32(0x1.62e42fep-1f);
  const float32x4_t safe_v = vdupq_n_f32(126.0f);

  const npy_half *infp = src;
  npy_half *outp = dst;
  npy_intp remaining = len;

  feclearexcept(FE_ALL_EXCEPT);

  while (remaining >= 16) {
    __builtin_prefetch(infp + 64, 0, 3);
    __builtin_prefetch(outp + 64, 1, 3);

    float16x8_t h0 = vld1q_f16((const float16_t *)infp);
    float16x8_t h1 = vld1q_f16((const float16_t *)(infp + 8));

    float32x4_t x[4], res[4];

    x[0] = vcvt_f32_f16(vget_low_f16(h0));
    x[1] = vcvt_f32_f16(vget_high_f16(h0));
    x[2] = vcvt_f32_f16(vget_low_f16(h1));
    x[3] = vcvt_f32_f16(vget_high_f16(h1));

    for (int k = 0; k < 4; k++) {
      uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x[k], x[k]));
      uint32x4_t is_overflow = vcgtq_f32(x[k], exp_max_v);
      uint32x4_t is_underflow = vcltq_f32(x[k], exp_min_v);
      uint32x4_t need_s = vorrq_u32(vorrq_u32(vorrq_u32(is_overflow, is_underflow),
                          vceqq_f32(x[k], pos_inf_v)), vceqq_f32(x[k], neg_inf_v));
      need_s = vorrq_u32(need_s, is_nan);
      float32x4_t sf = vbslq_f32(need_s, safe_v, x[k]);
      res[k] = neon_exp2f_inline_core(sf, shift_v, bias_v, c0_v, c1_v, c2_v, c3_v, c4_v);
      res[k] = vbslq_f32(is_nan, x[k], res[k]);
      res[k] = vbslq_f32(vceqq_f32(x[k], pos_inf_v), pos_inf_v, res[k]);
      res[k] = vbslq_f32(vceqq_f32(x[k], neg_inf_v), zero_v, res[k]);
      res[k] = vbslq_f32(is_overflow, pos_inf_v, res[k]);
      res[k] = vbslq_f32(is_underflow, zero_v, res[k]);
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
    float32x4_t x[2], res[2];

    x[0] = vcvt_f32_f16(vget_low_f16(h0));
    x[1] = vcvt_f32_f16(vget_high_f16(h0));

    for (int k = 0; k < 2; k++) {
      uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x[k], x[k]));
      uint32x4_t is_overflow = vcgtq_f32(x[k], exp_max_v);
      uint32x4_t is_underflow = vcltq_f32(x[k], exp_min_v);
      uint32x4_t need_s = vorrq_u32(vorrq_u32(vorrq_u32(is_overflow, is_underflow),
                          vceqq_f32(x[k], pos_inf_v)), vceqq_f32(x[k], neg_inf_v));
      need_s = vorrq_u32(need_s, is_nan);
      float32x4_t sf = vbslq_f32(need_s, safe_v, x[k]);
      res[k] = neon_exp2f_inline_core(sf, shift_v, bias_v, c0_v, c1_v, c2_v, c3_v, c4_v);
      res[k] = vbslq_f32(is_nan, x[k], res[k]);
      res[k] = vbslq_f32(vceqq_f32(x[k], pos_inf_v), pos_inf_v, res[k]);
      res[k] = vbslq_f32(vceqq_f32(x[k], neg_inf_v), zero_v, res[k]);
      res[k] = vbslq_f32(is_overflow, pos_inf_v, res[k]);
      res[k] = vbslq_f32(is_underflow, zero_v, res[k]);
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
    outp[i] = npy_float_to_half(npy_exp2f(npy_half_to_float(infp[i])));
  }

  feclearexcept(FE_ALL_EXCEPT);
}

static void __attribute__((optimize("O3")))
simd_exp2_neon_HALF(const npy_half *src, npy_intp ssrc,
                     npy_half *dst, npy_intp sdst, npy_intp len)
{
  if (ssrc == 1 && sdst == 1) {
    simd_exp2_neon_HALF_stride1(src, dst, len);
    return;
  }
  if ((ssrc == 2 || ssrc == 4) && (sdst == 2 || sdst == 4)) {
    simd_exp2_neon_HALF_stride2(src, ssrc, dst, sdst, len);
    return;
  }
  const npy_intp LARGE_BLOCK = 8192;
  const npy_intp SMALL_BLOCK = 256;
  const npy_intp VEC_SIZE = 8;
  const npy_intp UNROLL = 4;
  
  npy_float tmp_src_small[SMALL_BLOCK];
  npy_float tmp_dst_small[SMALL_BLOCK];
  
  npy_float *tmp_src = NULL;
  npy_float *tmp_dst = NULL;
  
  if (len > SMALL_BLOCK) {
    tmp_src = (npy_float *)malloc(LARGE_BLOCK * sizeof(npy_float));
    tmp_dst = (npy_float *)malloc(LARGE_BLOCK * sizeof(npy_float));
    if (!tmp_src || !tmp_dst) {
      if (tmp_src) free(tmp_src);
      if (tmp_dst) free(tmp_dst);
      feclearexcept(FE_ALL_EXCEPT);
      for (npy_intp i = 0; i < len; i++) {
        npy_float in1 = npy_half_to_float(src[i * ssrc]);
        dst[i * sdst] = npy_float_to_half(npy_exp2f(in1));
      }
      return;
    }
  }
  
  const npy_intp BLOCK_SIZE = (len > SMALL_BLOCK) ? LARGE_BLOCK : SMALL_BLOCK;
  npy_float *src_buf = (len > SMALL_BLOCK) ? tmp_src : tmp_src_small;
  npy_float *dst_buf = (len > SMALL_BLOCK) ? tmp_dst : tmp_dst_small;

  for (npy_intp offset = 0; offset < len; offset += BLOCK_SIZE) {
    npy_intp block_len = (len - offset > BLOCK_SIZE) ? BLOCK_SIZE : (len - offset);
    const npy_half *block_src = src + offset * ssrc;
    npy_half *block_dst = dst + offset * sdst;
    
    npy_intp i = 0;
    if (ssrc == 1) {
      for (; i + UNROLL * VEC_SIZE <= block_len; i += UNROLL * VEC_SIZE) {
        __builtin_prefetch(block_src + i + 256, 0, 3);
        float16x8_t h[4];
        for (int u = 0; u < 4; u++)
          h[u] = vld1q_f16((const float16_t *)(block_src + i + u * 8));
        for (int u = 0; u < 4; u++) {
          vst1q_f32(src_buf + i + u * 8, vcvt_f32_f16(vget_low_f16(h[u])));
          vst1q_f32(src_buf + i + u * 8 + 4, vcvt_f32_f16(vget_high_f16(h[u])));
        }
      }
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
        __builtin_prefetch(block_src + i + 128, 0, 3);
        float16x8_t h0 = vld1q_f16((const float16_t *)(block_src + i));
        vst1q_f32(src_buf + i, vcvt_f32_f16(vget_low_f16(h0)));
        vst1q_f32(src_buf + i + 4, vcvt_f32_f16(vget_high_f16(h0)));
      }
    } else {
      for (; i + 8 <= block_len; i += 8) {
        for (int j = 0; j < 8; j++) {
          src_buf[i + j] = npy_half_to_float(block_src[(i + j) * ssrc]);
        }
      }
    }
    for (; i < block_len; i++) {
      src_buf[i] = npy_half_to_float(block_src[i * ssrc]);
    }
    
    simd_exp2_neon_FLOAT(src_buf, 1, dst_buf, 1, block_len);
    
    i = 0;
    if (sdst == 1) {
      for (; i + UNROLL * VEC_SIZE <= block_len; i += UNROLL * VEC_SIZE) {
        __builtin_prefetch(block_dst + i + 256, 1, 3);
        for (int u = 0; u < 4; u++) {
          float16x8_t hv = vcombine_f16(vcvt_f16_f32(vld1q_f32(dst_buf + i + u * 8)),
                                         vcvt_f16_f32(vld1q_f32(dst_buf + i + u * 8 + 4)));
          vst1q_f16((float16_t *)(block_dst + i + u * 8), hv);
        }
      }
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
        __builtin_prefetch(block_dst + i + 128, 1, 3);
        float16x8_t h_combined = vcombine_f16(vcvt_f16_f32(vld1q_f32(dst_buf + i)), vcvt_f16_f32(vld1q_f32(dst_buf + i + 4)));
        vst1q_f16((float16_t *)(block_dst + i), h_combined);
      }
    } else {
      for (; i + 8 <= block_len; i += 8) {
        for (int j = 0; j < 8; j++) {
          block_dst[(i + j) * sdst] = npy_float_to_half(dst_buf[i + j]);
        }
      }
    }
    for (; i < block_len; i++) {
      block_dst[i * sdst] = npy_float_to_half(dst_buf[i]);
    }
  }
  
  if (len > SMALL_BLOCK) {
    free(tmp_src);
    free(tmp_dst);
  }
  feclearexcept(FE_ALL_EXCEPT);
}


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
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_float in1 = *(npy_float *)ip1;
            *(npy_float *)op1 = npy_exp2f(in1);
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_float *src = (npy_float*)args[0];
        npy_float *dst = (npy_float*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);
        simd_exp2_neon_FLOAT(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            simd_exp2_neon_FLOAT((npy_float *)ip1, 1, (npy_float *)op1, 1, 1);
        }
    }
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
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_double in1 = *(npy_double *)ip1;
            *(npy_double *)op1 = npy_exp2(in1);
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_double *src = (npy_double*)args[0];
        npy_double *dst = (npy_double*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_double);
        const npy_intp sdst = steps[1] / sizeof(npy_double);
        simd_exp2_neon_DOUBLE(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            simd_exp2_neon_DOUBLE((npy_double *)ip1, 1, (npy_double *)op1, 1, 1);
        }
    }
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
#if SIMD_ARM
    const npy_intp len = dimensions[0];
    if (len == 1) {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_exp2f(in1));
        }
    }
    else if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        const npy_half *src = (npy_half*)args[0];
        npy_half *dst = (npy_half*)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_half);
        const npy_intp sdst = steps[1] / sizeof(npy_half);
        simd_exp2_neon_HALF(src, ssrc, dst, sdst, len);
    }
    else {
        UNARY_LOOP {
            const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
            *((npy_half *)op1) = npy_float_to_half(npy_exp2f(in1));
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
