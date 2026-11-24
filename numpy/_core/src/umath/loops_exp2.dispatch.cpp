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
                        float32x4_t c3, float32x4_t c4,
                        float32x4_t d0, float32x4_t d1)
{
  float32x4_t z = vaddq_f32(x, shift);
  float32x4_t n = vsubq_f32(z, shift);
  float32x4_t r = vsubq_f32(x, n);
  int32x4_t ni = vcvtq_s32_f32(n);
  uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_s32(ni), 23);
  float32x4_t scale = vreinterpretq_f32_u32(vaddq_u32(e, exponent_bias));
  /* Floor the products at |r| >= 2^-21 (even powers) / 2^-124 (the
   * inner*r term): a tiny in-band r -- a tiny x, or an x near a
   * multiple of ln2/32 after range reduction -- would otherwise
   * multiply into the subnormal range and raise a spurious underflow
   * flag, while exp2(x) = fl(1 + x*ln2) still rounds to the same value
   * there (1.0 for the floored zone).  The inner*r term keeps r's sign
   * through the floored magnitude; +-0 stays exact (exp2(+-0) = 1). */
  float32x4_t r_abs = vabsq_f32(r);
  float32x4_t r_even = vmaxq_f32(r_abs, vdupq_n_f32(0x1p-21f));
  float32x4_t r_lin = vbslq_f32(vdupq_n_u32(0x80000000u), r,
                                vmaxq_f32(r_abs, vdupq_n_f32(0x1p-124f)));
  float32x4_t r2 = vmulq_f32(r_even, r_even);
  float32x4_t r4 = vmulq_f32(r2, r2);
  /*
   * Degree-7 minimax approximation of 2^r - 1 for r in [-0.5, 0.5]:
   *   r * (c4 + c3*r + (c2 + c1*r)*r^2 + (c0 + d1*r)*r^4 + d0*r^6)
   * Measured over the whole [-126, 127.5] window: max 1 ulp, ~88%
   * correctly rounded. (The previous degree-5 Taylor coefficients were
   * up to ~41 ulp off across this range.)
   */
  float32x4_t A = vfmaq_f32(c4, r, c3);
  float32x4_t B = vfmaq_f32(c2, r, c1);
  float32x4_t C = vfmaq_f32(c0, r, d1);
  float32x4_t inner = vfmaq_f32(A, B, r2);
  inner = vfmaq_f32(inner, C, r4);
  inner = vfmaq_f32(inner, d0, vmulq_f32(r4, r2));
  float32x4_t poly = vmulq_f32(inner, r_lin);
  return vfmaq_f32(scale, poly, scale);
}

static void
simd_exp2_neon_FLOAT(const npy_float *src, npy_intp ssrc,
                     npy_float *dst, npy_intp sdst, npy_intp len)
{
  const float32x4_t pos_inf = vdupq_n_f32(NPY_INFINITYF);
  const float32x4_t neg_inf = vdupq_n_f32(-NPY_INFINITYF);
  const float32x4_t shift = vdupq_n_f32(0x1.8p+23f);
  const uint32x4_t exponent_bias = vdupq_n_u32(0x3f800000);
  /* the scale bit-trick inside the polynomial only produces a valid float
   * while the rounded exponent stays within [-126, 127]; lanes outside
   * (incl. +-inf) fall back to libm, which also raises the overflow/
   * underflow FP flags like the scalar path does */
  const float32x4_t exp_max = vdupq_n_f32(127.5f);
  const float32x4_t exp_min = vdupq_n_f32(-126.0f);
  const float32x4_t c0 = vdupq_n_f32(0x1.5d95e6p-10f);
  const float32x4_t c1 = vdupq_n_f32(0x1.3b29f4p-7f);
  const float32x4_t c2 = vdupq_n_f32(0x1.c6b08p-5f);
  const float32x4_t c3 = vdupq_n_f32(0x1.ebfbe0p-3f);
  const float32x4_t c4 = vdupq_n_f32(0x1.62e430p-1f);
  const float32x4_t d0 = vdupq_n_f32(0x1.f1a68ap-17f);
  const float32x4_t d1 = vdupq_n_f32(0x1.44627p-13f);

  const npy_float *infp = src;
  npy_float *outp = dst;
  npy_intp remaining = len;

  const npy_intp UNROLL = 8;
  const npy_intp VEC_SIZE = 4;

  while (remaining >= UNROLL * VEC_SIZE) {
    float32x4_t x[UNROLL], result[UNROLL];

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

    const float32x4_t safe_val = vdupq_n_f32(0.0f);
    uint32x4_t any_libm = vdupq_n_u32(0);
    uint32x4_t need_libm[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x[i], x[i]));
      /* inequality compares (FCMGE/FCMGT) raise IOC on QNaN lanes, unlike
       * vceqq; mask NaN lanes out before comparing (they compare false
       * anyway) */
      float32x4_t x_nn = vbslq_f32(is_nan, vdupq_n_f32(0.0f), x[i]);
      uint32x4_t is_overflow = vcgeq_f32(x_nn, exp_max);
      uint32x4_t is_underflow = vcltq_f32(x_nn, exp_min);
      need_libm[i] =
          vorrq_u32(vorrq_u32(is_overflow, is_underflow),
                    vorrq_u32(is_nan,
                              vorrq_u32(vceqq_f32(x[i], pos_inf),
                                        vceqq_f32(x[i], neg_inf))));
      any_libm = vorrq_u32(any_libm, need_libm[i]);
      /* Evaluate the polynomial on sanitized inputs unconditionally: with
       * the default -fno-trapping-math the compiler may speculate the
       * vector path, so it must never see the raw out-of-range/NaN bit
       * patterns, which would raise spurious FP flags. */
      float32x4_t xs = vbslq_f32(need_libm[i], safe_val, x[i]);
      result[i] = neon_exp2f_inline_core(xs, shift, exponent_bias, c0, c1, c2, c3, c4, d0, d1);
    }
    /* Lanes outside the polynomial's window (incl. NaN/inf) come from
     * libm, which yields the exact scalar-path value and raises the
     * overflow/underflow FP flags like the scalar path does. Deciding
     * once per block keeps the common (all-in-window) case branch-free. */
    if (__builtin_expect(neon_has_any_lane_u32(any_libm), 0)) {
      for (int i = 0; i < UNROLL; i++) {
        if (__builtin_expect(neon_has_any_lane_u32(need_libm[i]), 0)) {
          float vals[4], outs[4];
          uint32_t m[4];
          vst1q_f32(vals, x[i]);
          vst1q_f32(outs, result[i]);
          vst1q_u32(m, need_libm[i]);
          for (int j = 0; j < 4; j++)
            if (m[j]) outs[j] = npy_exp2f(vals[j]);
          result[i] = vld1q_f32(outs);
        }
      }
    }

    if (sdst == 1) {
      for (int i = 0; i < UNROLL; i++)
        vst1q_f32(outp + i * VEC_SIZE, result[i]);
    } else {
      for (int j = 0; j < UNROLL; j++) {
        float tmp[VEC_SIZE];
        vst1q_f32(tmp, result[j]);
        for (int i = 0; i < VEC_SIZE; i++)
          outp[(i + j * VEC_SIZE) * sdst] = tmp[i];
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

    uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x, x));
    /* mask NaN lanes out of the inequality compares (see the main loop) */
    float32x4_t x_nn = vbslq_f32(is_nan, vdupq_n_f32(0.0f), x);
    uint32x4_t is_overflow = vcgeq_f32(x_nn, exp_max);
    uint32x4_t is_underflow = vcltq_f32(x_nn, exp_min);
    uint32x4_t need_libm = vorrq_u32(vorrq_u32(is_overflow, is_underflow),
                                     vorrq_u32(is_nan,
                                               vorrq_u32(vceqq_f32(x, pos_inf),
                                                         vceqq_f32(x, neg_inf))));

    /* sanitized polynomial evaluation (see the main loop) */
    const float32x4_t safe_val = vdupq_n_f32(0.0f);
    float32x4_t xs = vbslq_f32(need_libm, safe_val, x);
    float32x4_t result = neon_exp2f_inline_core(xs, shift, exponent_bias, c0, c1, c2, c3, c4, d0, d1);
    if (__builtin_expect(neon_has_any_lane_u32(need_libm), 0)) {
      float vals[4], outs[4];
      uint32_t m[4];
      vst1q_f32(vals, x);
      vst1q_f32(outs, result);
      vst1q_u32(m, need_libm);
      for (int j = 0; j < current_len; j++)
        if (m[j]) outs[j] = npy_exp2f(vals[j]);
      result = vld1q_f32(outs);
    }

    if (sdst == 1) {
      if (current_len == 1) vst1q_lane_f32(outp, result, 0);
      else if (current_len == 2) vst1_f32(outp, vget_low_f32(result));
      else if (current_len == 3) {
        vst1q_lane_f32(outp, result, 0);
        vst1q_lane_f32(outp + 1, result, 1);
        vst1q_lane_f32(outp + 2, result, 2);
      } else vst1q_f32(outp, result);
    } else {
      float tmp[4];
      vst1q_f32(tmp, result);
      for (int i = 0; i < current_len; i++) outp[i * sdst] = tmp[i];
    }

    infp += ssrc * current_len;
    outp += sdst * current_len;
    remaining -= current_len;
  }
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
  /*
   * Estrin evaluation of the degree-4 Taylor series of 2^r - 1 with
   * {poly0..poly3} = {ln2, ln2^2/2, ln2^3/6, ln2^4/24}:
   *   poly0*r + poly1*r^2 + poly2*r^3 + poly3*r^4
   *     = r*(poly0 + r*poly1) + r^3*(poly2 + r*poly3)
   * The previous code multiplied q by r^2, which double-counted the r^2
   * term and dropped the r^4 one (~5.4e-7 relative error for any x that
   * is not a multiple of 1/128).
   */
  float64x2_t pv = vfmaq_f64(vmulq_f64(r, p), vmulq_f64(r2, r), q);
  return vfmaq_f64(scale, scale, pv);
}

static void
simd_exp2_neon_DOUBLE(const npy_double *src, npy_intp ssrc,
                      npy_double *dst, npy_intp sdst, npy_intp len)
{
  const float64x2_t special_bound = vdupq_n_f64(1022.0);
  /* Nonzero |x| < 2^-30: r == x there and the polynomial's r*r, r*p
   * and r^3 products (and the pv sum) would go subnormal, raising a
   * spurious underflow flag although exp2(x) = fl(1 + x*ln2) ~ 1.0.
   * libm returns that value with the scalar path's (no) flags; +-0
   * stays in-band (exp2(+-0) is exact).  Lanes with n != 0 keep
   * |r| >~ 2^-57, far above every product's subnormal boundary. */
  const float64x2_t tiny_bound = vdupq_n_f64(0x1p-30);
  const float64x2_t zero_val = vdupq_n_f64(0.0);
  const float64x2_t shift = vdupq_n_f64(0x1.8p+52 / V_EXP_N);
  const float64x2_t poly0 = vdupq_n_f64(0x1.62e42fefa3686p-1);
  const float64x2_t poly1 = vdupq_n_f64(0x1.ebfbdff82c241p-3);
  const float64x2_t poly2 = vdupq_n_f64(0x1.c6b09b16de99ap-5);
  const float64x2_t poly3 = vdupq_n_f64(0x1.3b2abf5571ad8p-7);
  const uint64x2_t all_bits = vdupq_n_u64(0xFFFFFFFFFFFFFFFFULL);

  const npy_double *infp = src;
  npy_double *outp = dst;
  npy_intp remaining = len;

  const npy_intp UNROLL = 8;
  const npy_intp VEC_SIZE = 2;

  while (remaining >= UNROLL * VEC_SIZE) {
    float64x2_t x[UNROLL], result[UNROLL];

    if (ssrc == 1) {
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

    const float64x2_t safe_val = vdupq_n_f64(0.0);
    uint64x2_t any_libm = vdupq_n_u64(0);
    uint64x2_t need_libm[UNROLL];
    for (int i = 0; i < UNROLL; i++) {
      /* |x| > 1022 (incl. +-inf) or NaN: the scale bit-trick behind the
       * polynomial breaks down across the overflow/subnormal boundaries;
       * evaluate those lanes via libm, which also raises the overflow/
       * underflow FP flags like the scalar path does. */
      uint64x2_t is_nan = veorq_u64(vceqq_f64(x[i], x[i]), all_bits);
      /* mask NaN lanes out of the inequality compare (FCAGT raises IOC on
       * QNaN lanes, unlike vceqq) */
      float64x2_t x_nn = vbslq_f64(is_nan, vdupq_n_f64(0.0), x[i]);
      need_libm[i] =
          vorrq_u64(vcagtq_f64(x_nn, special_bound), is_nan);
      need_libm[i] = vorrq_u64(
          need_libm[i],
          vbicq_u64(vcltq_f64(vabsq_f64(x_nn), tiny_bound),
                    vceqq_f64(x_nn, zero_val)));
      any_libm = vorrq_u64(any_libm, need_libm[i]);
      /* sanitized polynomial evaluation (speculation-safe, see the
       * FLOAT kernel) */
      float64x2_t xs = vbslq_f64(need_libm[i], safe_val, x[i]);
      result[i] = neon_exp2_inline_core_f64(xs, shift, poly0, poly1, poly2, poly3);
    }
    /* decide once per block: the common (all-in-window) case stays
     * branch-free, see the FLOAT kernel */
    if (__builtin_expect(neon_has_any_lane_u64(any_libm), 0)) {
      for (int i = 0; i < UNROLL; i++) {
        if (__builtin_expect(neon_has_any_lane_u64(need_libm[i]), 0)) {
          double vals[2], outs[2];
          uint64_t m[2];
          vst1q_f64(vals, x[i]);
          vst1q_f64(outs, result[i]);
          vst1q_u64(m, need_libm[i]);
          for (int j = 0; j < 2; j++)
            if (m[j]) outs[j] = npy_exp2(vals[j]);
          result[i] = vld1q_f64(outs);
        }
      }
    }

    if (sdst == 1) {
      for (int i = 0; i < UNROLL; i++)
        vst1q_f64(outp + i * VEC_SIZE, result[i]);
    } else {
      for (int j = 0; j < UNROLL; j++) {
        for (int i = 0; i < VEC_SIZE; i++)
          vst1q_lane_f64(outp + (i + j * VEC_SIZE) * sdst, result[j], i);
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
      /* broadcast the single valid lane: loading the full vector would
       * read one element past the buffer (and its garbage bits must
       * never reach the polynomial) */
      if (current_len == 1) x = vld1q_dup_f64(infp);
      else x = vld1q_f64(infp);
    } else {
      double vals[2] = {0.0, 0.0};
      for (int i = 0; i < current_len; i++) vals[i] = infp[i * ssrc];
      x = vld1q_f64(vals);
    }

    uint64x2_t is_nan = veorq_u64(vceqq_f64(x, x), all_bits);
    /* mask NaN lanes out of the inequality compare (FCAGT raises IOC on
     * QNaN lanes, unlike vceqq) */
    float64x2_t x_nn = vbslq_f64(is_nan, vdupq_n_f64(0.0), x);
    uint64x2_t need_libm =
        vorrq_u64(vcagtq_f64(x_nn, special_bound), is_nan);
    need_libm = vorrq_u64(
        need_libm,
        vbicq_u64(vcltq_f64(vabsq_f64(x_nn), tiny_bound),
                  vceqq_f64(x_nn, zero_val)));

    /* sanitized polynomial evaluation (see the main loop) */
    const float64x2_t safe_val = vdupq_n_f64(0.0);
    float64x2_t xs = vbslq_f64(need_libm, safe_val, x);
    float64x2_t result = neon_exp2_inline_core_f64(xs, shift, poly0, poly1, poly2, poly3);
    if (__builtin_expect(neon_has_any_lane_u64(need_libm), 0)) {
      double vals[2], outs[2];
      uint64_t m[2];
      vst1q_f64(vals, x);
      vst1q_f64(outs, result);
      vst1q_u64(m, need_libm);
      for (int j = 0; j < current_len; j++)
        if (m[j]) outs[j] = npy_exp2(vals[j]);
      result = vld1q_f64(outs);
    }

    if (sdst == 1) {
      if (current_len == 1) vst1q_lane_f64(outp, result, 0);
      else vst1q_f64(outp, result);
    } else {
      for (int i = 0; i < current_len; i++)
        vst1q_lane_f64(outp + i * sdst, result, i);
    }

    infp += ssrc * current_len;
    outp += sdst * current_len;
    remaining -= current_len;
  }
}

static inline void
simd_exp2_neon_HALF_stride2(const npy_half *src, npy_intp ssrc,
                             npy_half *dst, npy_intp sdst, npy_intp len)
{
  const float32x4_t pos_inf_v = vdupq_n_f32(NPY_INFINITYF);
  const float32x4_t neg_inf_v = vdupq_n_f32(-NPY_INFINITYF);
  const float32x4_t shift_v = vdupq_n_f32(0x1.8p+23f);
  const uint32x4_t bias_v = vdupq_n_u32(0x3f800000);
  /* half precision results overflow past 2^16 and go subnormal below
   * 2^-14; lanes outside that window (incl. NaN/inf) fall back to libm
   * plus software conversion, matching the scalar path for both value
   * and FP flags */
  const float32x4_t exp_max_v = vdupq_n_f32(16.0f);
  const float32x4_t exp_min_v = vdupq_n_f32(-14.0f);
  const float32x4_t c0_v = vdupq_n_f32(0x1.5d95e6p-10f);
  const float32x4_t c1_v = vdupq_n_f32(0x1.3b29f4p-7f);
  const float32x4_t c2_v = vdupq_n_f32(0x1.c6b08p-5f);
  const float32x4_t c3_v = vdupq_n_f32(0x1.ebfbe0p-3f);
  const float32x4_t c4_v = vdupq_n_f32(0x1.62e430p-1f);
  const float32x4_t d0_v = vdupq_n_f32(0x1.f1a68ap-17f);
  const float32x4_t d1_v = vdupq_n_f32(0x1.44627p-13f);

  const npy_half *infp = src;
  npy_half *outp = dst;
  npy_intp remaining = len;

  while (remaining >= 16) {

    float32x4_t x[4], result[4];
    for (int k = 0; k < 4; k++) {
      uint16_t h_raw[4];
      for (int j = 0; j < 4; j++)
        h_raw[j] = infp[(k * 4 + j) * ssrc];
      x[k] = vcvt_f32_f16(vld1_f16((const float16_t *)h_raw));
    }

    const float32x4_t safe_val = vdupq_n_f32(0.0f);
    uint32_t libm_mask[16];
    uint32x4_t any_libm = vdupq_n_u32(0);
    for (int k = 0; k < 4; k++) {
      uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x[k], x[k]));
      /* mask NaN lanes out of the inequality compares (see the FLOAT
       * kernel) */
      float32x4_t x_nn = vbslq_f32(is_nan, vdupq_n_f32(0.0f), x[k]);
      uint32x4_t is_overflow = vcgeq_f32(x_nn, exp_max_v);
      uint32x4_t is_underflow = vcltq_f32(x_nn, exp_min_v);
      uint32x4_t need_libm =
          vorrq_u32(vorrq_u32(is_overflow, is_underflow),
                    vorrq_u32(is_nan,
                              vorrq_u32(vceqq_f32(x[k], pos_inf_v),
                                        vceqq_f32(x[k], neg_inf_v))));
      any_libm = vorrq_u32(any_libm, need_libm);
      /* sanitized polynomial evaluation (speculation-safe, see the FLOAT
       * kernel); lanes routed to libm keep a placeholder (exp2(0) == 1.0)
       * that converts to half precision without raising FP flags */
      float32x4_t xs = vbslq_f32(need_libm, safe_val, x[k]);
      result[k] = neon_exp2f_inline_core(xs, shift_v, bias_v,
                                           c0_v, c1_v, c2_v, c3_v, c4_v, d0_v, d1_v);
      vst1q_u32(&libm_mask[k * 4], need_libm);
    }

    /* save the raw inputs before the store below: when the views overlap
     * (src == dst with matching strides) the vector store overwrites
     * them, and the libm fixup must not re-read the clobbered memory */
    float xin[16];
    for (int k = 0; k < 4; k++)
      vst1q_f32(&xin[k * 4], x[k]);

    for (int k = 0; k < 4; k++) {
      float16x4_t rh = vcvt_f16_f32(result[k]);
      for (int j = 0; j < 4; j++) {
        union { float16_t f; npy_half h; } u;
        u.f = vget_lane_f16(rh, j);
        outp[(k * 4 + j) * sdst] = u.h;
      }
    }
    /* libm-routed lanes: software evaluation + conversion, matching the
     * scalar path for value and FP flags (npy_float_to_half raises the
     * same legitimate OFC/UFC conversion flags as the vector vcvt);
     * the single any-lane check keeps clean blocks branch-free */
    if (__builtin_expect(neon_has_any_lane_u32(any_libm), 0)) {
      for (int j = 0; j < 16; j++)
        if (libm_mask[j])
          outp[j * sdst] = npy_float_to_half(npy_exp2f(xin[j]));
    }

    infp += ssrc * 16;
    outp += sdst * 16;
    remaining -= 16;
  }

  while (remaining >= 4) {
    uint16_t h_raw[4];
    for (int j = 0; j < 4; j++)
      h_raw[j] = infp[j * ssrc];
    float32x4_t x = vcvt_f32_f16(vld1_f16((const float16_t *)h_raw));

    uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x, x));
    /* mask NaN lanes out of the inequality compares (see the FLOAT
     * kernel) */
    float32x4_t x_nn = vbslq_f32(is_nan, vdupq_n_f32(0.0f), x);
    uint32x4_t is_overflow = vcgeq_f32(x_nn, exp_max_v);
    uint32x4_t is_underflow = vcltq_f32(x_nn, exp_min_v);
    uint32x4_t need_libm = vorrq_u32(vorrq_u32(is_overflow, is_underflow),
                                     vorrq_u32(is_nan,
                                               vorrq_u32(vceqq_f32(x, pos_inf_v),
                                                         vceqq_f32(x, neg_inf_v))));

    /* sanitized polynomial evaluation (see the 16-wide loop above) */
    const float32x4_t safe_val = vdupq_n_f32(0.0f);
    float32x4_t xs = vbslq_f32(need_libm, safe_val, x);
    float32x4_t result = neon_exp2f_inline_core(xs, shift_v, bias_v,
                                                  c0_v, c1_v, c2_v, c3_v, c4_v, d0_v, d1_v);

    /* save the raw inputs before the store below (see the 16-wide loop) */
    float xin[4];
    vst1q_f32(xin, x);

    float16x4_t rh = vcvt_f16_f32(result);
    for (int j = 0; j < 4; j++) {
      union { float16_t f; npy_half h; } u;
      u.f = vget_lane_f16(rh, j);
      outp[j * sdst] = u.h;
    }
    /* libm-routed lanes: software evaluation + conversion (see above);
     * the single any-lane check keeps clean blocks branch-free */
    if (__builtin_expect(neon_has_any_lane_u32(need_libm), 0)) {
      uint32_t libm_mask[4];
      vst1q_u32(libm_mask, need_libm);
      for (int j = 0; j < 4; j++)
        if (libm_mask[j])
          outp[j * sdst] = npy_float_to_half(npy_exp2f(xin[j]));
    }

    infp += ssrc * 4;
    outp += sdst * 4;
    remaining -= 4;
  }

  for (npy_intp i = 0; i < remaining; i++) {
    outp[i * sdst] = npy_float_to_half(npy_exp2f(npy_half_to_float(infp[i * ssrc])));
  }
}

static inline void
simd_exp2_neon_HALF_stride1(const npy_half *src, npy_half *dst, npy_intp len)
{
  const float32x4_t pos_inf_v = vdupq_n_f32(NPY_INFINITYF);
  const float32x4_t neg_inf_v = vdupq_n_f32(-NPY_INFINITYF);
  const float32x4_t shift_v = vdupq_n_f32(0x1.8p+23f);
  const uint32x4_t bias_v = vdupq_n_u32(0x3f800000);
  /* half precision results overflow past 2^16 and go subnormal below
   * 2^-14; lanes outside that window (incl. NaN/inf) fall back to libm
   * plus software conversion, matching the scalar path for both value
   * and FP flags */
  const float32x4_t exp_max_v = vdupq_n_f32(16.0f);
  const float32x4_t exp_min_v = vdupq_n_f32(-14.0f);
  const float32x4_t c0_v = vdupq_n_f32(0x1.5d95e6p-10f);
  const float32x4_t c1_v = vdupq_n_f32(0x1.3b29f4p-7f);
  const float32x4_t c2_v = vdupq_n_f32(0x1.c6b08p-5f);
  const float32x4_t c3_v = vdupq_n_f32(0x1.ebfbe0p-3f);
  const float32x4_t c4_v = vdupq_n_f32(0x1.62e430p-1f);
  const float32x4_t d0_v = vdupq_n_f32(0x1.f1a68ap-17f);
  const float32x4_t d1_v = vdupq_n_f32(0x1.44627p-13f);

  const npy_half *infp = src;
  npy_half *outp = dst;
  npy_intp remaining = len;

  while (remaining >= 16) {

    float16x8_t h0 = vld1q_f16((const float16_t *)infp);
    float16x8_t h1 = vld1q_f16((const float16_t *)(infp + 8));

    float32x4_t x[4], res[4];

    x[0] = vcvt_f32_f16(vget_low_f16(h0));
    x[1] = vcvt_f32_f16(vget_high_f16(h0));
    x[2] = vcvt_f32_f16(vget_low_f16(h1));
    x[3] = vcvt_f32_f16(vget_high_f16(h1));

    const float32x4_t safe_val = vdupq_n_f32(0.0f);
    uint32_t libm_mask[16];
    uint32x4_t any_libm = vdupq_n_u32(0);
    for (int k = 0; k < 4; k++) {
      uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x[k], x[k]));
      /* mask NaN lanes out of the inequality compares (see the FLOAT
       * kernel) */
      float32x4_t x_nn = vbslq_f32(is_nan, vdupq_n_f32(0.0f), x[k]);
      uint32x4_t is_overflow = vcgeq_f32(x_nn, exp_max_v);
      uint32x4_t is_underflow = vcltq_f32(x_nn, exp_min_v);
      uint32x4_t need_libm =
          vorrq_u32(vorrq_u32(is_overflow, is_underflow),
                    vorrq_u32(is_nan,
                              vorrq_u32(vceqq_f32(x[k], pos_inf_v),
                                        vceqq_f32(x[k], neg_inf_v))));
      any_libm = vorrq_u32(any_libm, need_libm);
      /* sanitized polynomial evaluation (speculation-safe, see the FLOAT
       * kernel); lanes routed to libm keep a placeholder (exp2(0) == 1.0)
       * that converts to half precision without raising FP flags */
      float32x4_t xs = vbslq_f32(need_libm, safe_val, x[k]);
      res[k] = neon_exp2f_inline_core(xs, shift_v, bias_v, c0_v, c1_v, c2_v, c3_v, c4_v, d0_v, d1_v);
      vst1q_u32(&libm_mask[k * 4], need_libm);
    }

    /* save the raw inputs before the store below: when src == dst the
     * vector store overwrites them, and the libm fixup must not re-read
     * the clobbered memory */
    float xin[16];
    for (int k = 0; k < 4; k++)
      vst1q_f32(&xin[k * 4], x[k]);

    float16x4_t rh[4];
    for (int k = 0; k < 4; k++)
      rh[k] = vcvt_f16_f32(res[k]);
    for (int k = 0; k < 2; k++)
      vst1q_f16((float16_t *)(outp + k * 8), vcombine_f16(rh[k*2], rh[k*2+1]));
    /* libm-routed lanes: evaluate and convert per lane. npy_float_to_half
     * raises the same legitimate OFC/UFC conversion flags as the vector
     * vcvt above (aarch64 uses the same hardware fcvt), keeping the
     * array path in flag parity with the scalar path for the
     * out-of-half-range results; the single any-lane check keeps clean
     * blocks branch-free */
    if (__builtin_expect(neon_has_any_lane_u32(any_libm), 0)) {
      for (int j = 0; j < 16; j++)
        if (libm_mask[j])
          outp[j] = npy_float_to_half(npy_exp2f(xin[j]));
    }

    infp += 16;
    outp += 16;
    remaining -= 16;
  }

  while (remaining >= 8) {
    float16x8_t h0 = vld1q_f16((const float16_t *)infp);
    float32x4_t x[2], res[2];

    x[0] = vcvt_f32_f16(vget_low_f16(h0));
    x[1] = vcvt_f32_f16(vget_high_f16(h0));

    const float32x4_t safe_val = vdupq_n_f32(0.0f);
    uint32_t libm_mask[8];
    uint32x4_t any_libm = vdupq_n_u32(0);
    for (int k = 0; k < 2; k++) {
      uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x[k], x[k]));
      /* mask NaN lanes out of the inequality compares (see the FLOAT
       * kernel) */
      float32x4_t x_nn = vbslq_f32(is_nan, vdupq_n_f32(0.0f), x[k]);
      uint32x4_t is_overflow = vcgeq_f32(x_nn, exp_max_v);
      uint32x4_t is_underflow = vcltq_f32(x_nn, exp_min_v);
      uint32x4_t need_libm =
          vorrq_u32(vorrq_u32(is_overflow, is_underflow),
                    vorrq_u32(is_nan,
                              vorrq_u32(vceqq_f32(x[k], pos_inf_v),
                                        vceqq_f32(x[k], neg_inf_v))));
      any_libm = vorrq_u32(any_libm, need_libm);
      /* sanitized polynomial evaluation (see the 16-wide loop above) */
      float32x4_t xs = vbslq_f32(need_libm, safe_val, x[k]);
      res[k] = neon_exp2f_inline_core(xs, shift_v, bias_v, c0_v, c1_v, c2_v, c3_v, c4_v, d0_v, d1_v);
      vst1q_u32(&libm_mask[k * 4], need_libm);
    }

    /* save the raw inputs before the store below (see the 16-wide loop) */
    float xin[8];
    for (int k = 0; k < 2; k++)
      vst1q_f32(&xin[k * 4], x[k]);

    float16x4_t rh[2];
    for (int k = 0; k < 2; k++)
      rh[k] = vcvt_f16_f32(res[k]);
    vst1q_f16((float16_t *)outp, vcombine_f16(rh[0], rh[1]));
    /* libm-routed lanes: software evaluation + conversion (see above) */
    if (__builtin_expect(neon_has_any_lane_u32(any_libm), 0)) {
      for (int j = 0; j < 8; j++)
        if (libm_mask[j])
          outp[j] = npy_float_to_half(npy_exp2f(xin[j]));
    }

    infp += 8;
    outp += 8;
    remaining -= 8;
  }

  for (npy_intp i = 0; i < remaining; i++) {
    outp[i] = npy_float_to_half(npy_exp2f(npy_half_to_float(infp[i])));
  }
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
        float16x8_t h[4];
        for (int u = 0; u < 4; u++)
          h[u] = vld1q_f16((const float16_t *)(block_src + i + u * 8));
        for (int u = 0; u < 4; u++) {
          vst1q_f32(src_buf + i + u * 8, vcvt_f32_f16(vget_low_f16(h[u])));
          vst1q_f32(src_buf + i + u * 8 + 4, vcvt_f32_f16(vget_high_f16(h[u])));
        }
      }
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
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
    /* The vectorized f32->f16 narrowing conversion raises OFC/UFC for
     * results that overflow or go subnormal in half precision, but these
     * are legitimate: the scalar path converts through npy_float_to_half,
     * which uses the same hardware fcvt on aarch64 (and a software
     * conversion that explicitly generates the same flags elsewhere), so
     * the flags must flow to the caller unmodified for parity. No
     * save/clear/restore is wanted here. */
    if (sdst == 1) {
      for (; i + UNROLL * VEC_SIZE <= block_len; i += UNROLL * VEC_SIZE) {
        for (int u = 0; u < 4; u++) {
          float16x8_t hv = vcombine_f16(vcvt_f16_f32(vld1q_f32(dst_buf + i + u * 8)),
                                         vcvt_f16_f32(vld1q_f32(dst_buf + i + u * 8 + 4)));
          vst1q_f16((float16_t *)(block_dst + i + u * 8), hv);
        }
      }
      for (; i + VEC_SIZE <= block_len; i += VEC_SIZE) {
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
        if (ssrc <= 2) {
            simd_exp2_neon_FLOAT(src, ssrc, dst, sdst, len);
        } else {
            const npy_float * __restrict ip = src;
            npy_float * __restrict op = dst;
            for (npy_intp i = 0; i < len; i++) {
                *op = npy_exp2f(*ip);
                ip += ssrc;
                op += sdst;
            }
        }
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
        if (ssrc <= 2) {
            simd_exp2_neon_DOUBLE(src, ssrc, dst, sdst, len);
        } else {
            const npy_double * __restrict ip = src;
            npy_double * __restrict op = dst;
            for (npy_intp i = 0; i < len; i++) {
                *op = npy_exp2(*ip);
                ip += ssrc;
                op += sdst;
            }
        }
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
