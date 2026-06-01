#include "fast_loop_macros.h"
#include "loops.h"
#include "loops_utils.h"
#include "numpy/npy_math.h"

#include "simd/simd.h"
#include "simd/simd.hpp"
#include <hwy/highway.h>

#include "npy_svml.h"

namespace hn = hwy::HWY_NAMESPACE;

#if defined(__aarch64__) && NPY_SIMD_FMA3
#define SIMD_ARM 1
#endif

#if SIMD_ARM
static inline hn::Vec<hn::ScalableTag<float>>
hwy_cbrt_f32_impl(hn::Vec<hn::ScalableTag<float>> x)
{
    const hn::ScalableTag<float> d;
    const hn::ScalableTag<uint32_t> du;

    const float poly_coefs[] = {0x1.2c74c2p-3f, -0x1.08e81ap-1f, 0x1.dd2d3p-1f, 0x1.c14e96p-2f};
    const float table[] = {0x1.428a3p-1f, 0x1.965feap-1f, 0x1p0f, 0x1.428a3p0f, 0x1.965feap0f};
    const float one_third = 0x1.555556p-2f;

    auto abs_x = hn::Abs(x);
    auto sign_mask = hn::CopySign(hn::Zero(d), x);

    auto iax = hn::BitCast(du, abs_x);

    auto tiny_bound = hn::Set(du, 0x00800000);
    auto thresh = hn::Set(du, 0x7f800000);
    auto special_mask = hn::Or(hn::Lt(iax, tiny_bound), hn::Ge(iax, thresh));

    auto m = hn::BitCast(d, hn::Or(hn::And(iax, hn::Set(du, 0x007fffff)), hn::Set(du, 0x3f000000)));

    auto e = hn::Sub(hn::ShiftRight<23>(iax), hn::Set(du, 126));
    auto e_f = hn::ConvertTo(d, hn::BitCast(hn::ScalableTag<int32_t>(), e));

    auto p = hn::MulAdd(hn::MulAdd(hn::MulAdd(hn::Set(d, poly_coefs[0]), m, hn::Set(d, poly_coefs[1])), m, hn::Set(d, poly_coefs[2])), m, hn::Set(d, poly_coefs[3]));

    auto one_third_v = hn::Set(d, one_third);
    auto two_thirds_v = hn::Add(one_third_v, one_third_v);
    auto m_by_3 = hn::Mul(m, one_third_v);
    auto p2 = hn::Mul(p, p);
    auto a = hn::MulAdd(p, two_thirds_v, hn::Div(m_by_3, p2));

    auto ef = hn::Mul(e_f, one_third_v);
    auto ey = hn::ConvertTo(hn::ScalableTag<int32_t>(), ef);
    auto em3 = hn::Sub(e, hn::Mul(hn::BitCast(du, ey), hn::Set(du, 3)));

    auto em3_idx = hn::Add(em3, hn::Set(du, 2));
    em3_idx = hn::Min(hn::Max(em3_idx, hn::Set(du, 0)), hn::Set(du, 4));

    auto lanes = hn::Lanes(d);
    alignas(64) float table_buf[64];
    for (size_t i = 0; i < lanes; ++i) {
        table_buf[i] = table[hn::ExtractLane(hn::BitCast(hn::ScalableTag<int32_t>(), em3_idx), i)];
    }
    auto my = hn::LoadN(d, table_buf, lanes);
    my = hn::Mul(my, a);

    auto exp_part = hn::BitCast(d, hn::ShiftLeft<23>(hn::Add(ey, hn::Set(hn::ScalableTag<int32_t>(), 127))));
    auto y = hn::Mul(exp_part, my);

    auto result = hn::Or(y, sign_mask);
    auto result_bits = hn::BitCast(du, result);
    auto x_bits = hn::BitCast(du, x);

    auto final_bits = hn::IfThenElse(special_mask, x_bits, result_bits);

    return hn::BitCast(d, final_bits);
}

static inline hn::Vec<hn::ScalableTag<double>>
hwy_cbrt_f64_impl(hn::Vec<hn::ScalableTag<double>> x)
{
    const hn::ScalableTag<double> d;
    const hn::ScalableTag<uint64_t> du;

    const double poly_coefs[] = {0x1.2c74eaa3ba428p-3, -0x1.08e83026b7e74p-1, 0x1.dd2d3f99e4c0ep-1, 0x1.c14e8ee44767p-2};
    const double table[] = {0x1.428a2f98d728bp-1, 0x1.965fea53d6e3dp-1, 0x1p0,
                            0x1.428a2f98d728bp0, 0x1.965fea53d6e3dp0};
    const double one_third = 0x1.5555555555555p-2;

    auto abs_x = hn::Abs(x);
    auto sign_mask = hn::CopySign(hn::Zero(d), x);

    auto iax = hn::BitCast(du, abs_x);

    auto tiny_bound = hn::Set(du, 0x0010000000000000ULL);
    auto thresh = hn::Set(du, 0x7ff0000000000000ULL);
    auto special_mask = hn::Or(hn::Lt(iax, tiny_bound), hn::Ge(iax, thresh));

    auto m = hn::BitCast(d, hn::Or(hn::And(iax, hn::Set(du, 0x000fffffffffffffULL)), hn::Set(du, 0x3fe0000000000000ULL)));

    auto e = hn::Sub(hn::ShiftRight<52>(iax), hn::Set(du, 1022));
    auto e_f = hn::ConvertTo(d, hn::BitCast(hn::ScalableTag<int64_t>(), e));

    auto p = hn::MulAdd(hn::MulAdd(hn::MulAdd(hn::Set(d, poly_coefs[0]), m, hn::Set(d, poly_coefs[1])), m, hn::Set(d, poly_coefs[2])), m, hn::Set(d, poly_coefs[3]));

    auto one_third_v = hn::Set(d, one_third);
    auto two_thirds_v = hn::Add(one_third_v, one_third_v);
    auto m_by_3 = hn::Mul(m, one_third_v);

    auto p2 = hn::Mul(p, p);
    auto a = hn::MulAdd(p, two_thirds_v, hn::Div(m_by_3, p2));

    auto a2 = hn::Mul(a, a);
    a = hn::MulAdd(a, two_thirds_v, hn::Div(m_by_3, a2));

    auto ef = hn::Mul(e_f, one_third_v);
    auto eb3f = hn::Round(ef);
    auto ey = hn::ConvertTo(hn::ScalableTag<int64_t>(), eb3f);
    auto em3 = hn::Sub(hn::BitCast(hn::ScalableTag<int64_t>(), e), hn::Mul(ey, hn::Set(hn::ScalableTag<int64_t>(), 3)));

    auto em3_idx = hn::Add(em3, hn::Set(hn::ScalableTag<int64_t>(), 2));
    em3_idx = hn::Min(hn::Max(em3_idx, hn::Set(hn::ScalableTag<int64_t>(), 0)), hn::Set(hn::ScalableTag<int64_t>(), 4));

    auto lanes = hn::Lanes(d);
    alignas(64) double table_buf[64];
    for (size_t i = 0; i < lanes; ++i) {
        table_buf[i] = table[hn::ExtractLane(em3_idx, i)];
    }
    auto my = hn::LoadN(d, table_buf, lanes);
    my = hn::Mul(my, a);

    auto exp_part = hn::BitCast(d, hn::ShiftLeft<52>(hn::Add(ey, hn::Set(hn::ScalableTag<int64_t>(), 1023))));
    auto y = hn::Mul(exp_part, my);

    auto result = hn::Or(y, sign_mask);
    auto result_bits = hn::BitCast(du, result);
    auto x_bits = hn::BitCast(du, x);

    auto final_bits = hn::IfThenElse(special_mask, x_bits, result_bits);

    return hn::BitCast(d, final_bits);
}

template <typename T, int BatchSize, typename VecOp, typename ScalarOp>
static void HWY_ATTR
simd_unary_fp_impl(const T *src, npy_intp ssrc, T *dst, npy_intp sdst, npy_intp len, VecOp vec_op, ScalarOp scalar_op)
{
    if (len == 1) {
        dst[0] = scalar_op(src[0]);
        return;
    }

    using Tag = hn::ScalableTag<T>;
    const Tag d;
    const int lanes = hn::Lanes(d);

    if (ssrc == 1 && sdst == 1) {
        if (BatchSize > 1) {
            for (; len >= BatchSize * lanes; len -= BatchSize * lanes, src += BatchSize * lanes, dst += BatchSize * lanes) {
                hn::Vec<Tag> xv[BatchSize], yv[BatchSize];
                for (int b = 0; b < BatchSize; ++b) {
                    xv[b] = hn::Load(d, src + b * lanes);
                }
                for (int b = 0; b < BatchSize; ++b) {
                    yv[b] = vec_op(xv[b]);
                }
                for (int b = 0; b < BatchSize; ++b) {
                    hn::Store(yv[b], d, dst + b * lanes);
                }
            }
        }
        for (; len > 0; len -= lanes, src += lanes, dst += lanes) {
            auto x = hn::LoadN(d, src, len);
            auto y = vec_op(x);
            hn::StoreN(y, d, dst, len);
        }
    } else {
        alignas(64) T tmp_in[64];
        alignas(64) T tmp_out[64];

        if (BatchSize > 1) {
            for (; len >= BatchSize * lanes; len -= BatchSize * lanes, src += ssrc * BatchSize * lanes, dst += sdst * BatchSize * lanes) {
                for (int b = 0; b < BatchSize; ++b) {
                    for (int j = 0; j < lanes; ++j) {
                        tmp_in[b * lanes + j] = src[(b * lanes + j) * ssrc];
                    }
                }
                for (int b = 0; b < BatchSize; ++b) {
                    auto x = hn::Load(d, tmp_in + b * lanes);
                    auto y = vec_op(x);
                    hn::Store(y, d, tmp_out + b * lanes);
                }
                for (int b = 0; b < BatchSize; ++b) {
                    for (int j = 0; j < lanes; ++j) {
                        dst[(b * lanes + j) * sdst] = tmp_out[b * lanes + j];
                    }
                }
            }
        }
        for (; len > 0; len -= lanes, src += ssrc * lanes, dst += sdst * lanes) {
            npy_intp current_len = (len < lanes) ? len : lanes;
            for (int j = 0; j < current_len; ++j) {
                tmp_in[j] = src[j * ssrc];
            }
            auto x = hn::LoadN(d, tmp_in, current_len);
            auto y = vec_op(x);
            hn::StoreN(y, d, tmp_out, current_len);
            for (int j = 0; j < current_len; ++j) {
                dst[j * sdst] = tmp_out[j];
            }
        }
    }
    npyv_cleanup();
}

template <typename F32Op>
static void HWY_ATTR
simd_unary_f16_impl(const npy_half *src, npy_intp ssrc, npy_half *dst, npy_intp sdst, npy_intp len, F32Op f32_op)
{
    const ::hwy::float16_t* src16 = (const ::hwy::float16_t*)src;
    ::hwy::float16_t* dst16 = (::hwy::float16_t*)dst;

    const hn::ScalableTag<npy_float> f32_tag_t;
    const hn::ScalableTag<::hwy::float16_t> f16_tag_t;
    const hn::Half<decltype(f16_tag_t)> half_f16_tag_t;
    const int lanes = hn::Lanes(f16_tag_t);

    npy_half NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) fallback_buf[hn::MaxLanes(f16_tag_t) * 2];

    for (; len > 0; len -= lanes, src16 += ssrc * lanes, dst16 += sdst * lanes) {
        const npy_intp current_len = (len < lanes) ? len : lanes;
        hn::Vec<decltype(f16_tag_t)> x16_in;
        if (ssrc == 1) {
            x16_in = hn::LoadN(f16_tag_t, src16, current_len);
        } else {
            ::hwy::float16_t NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) tmp[hn::MaxLanes(f16_tag_t)] = {};
            for (int j = 0; j < current_len; ++j) {
                tmp[j] = src16[j * ssrc];
            }
            x16_in = hn::LoadN(f16_tag_t, tmp, current_len);
        }

        hn::Store(x16_in, f16_tag_t, (::hwy::float16_t*)fallback_buf);

        auto x32_in0 = hn::PromoteLowerTo(f32_tag_t, x16_in);
        auto x32_in1 = hn::PromoteUpperTo(f32_tag_t, x16_in);

        auto y32_out0 = f32_op(x32_in0);
        auto y32_out1 = f32_op(x32_in1);

        auto y16_out0 = hn::DemoteTo(half_f16_tag_t, y32_out0);
        auto y16_out1 = hn::DemoteTo(half_f16_tag_t, y32_out1);
        auto y16_out = hn::Combine(f16_tag_t, y16_out1, y16_out0);

        hn::Store(y16_out, f16_tag_t, (::hwy::float16_t*)fallback_buf + lanes);

        for (int j = 0; j < lanes / 2 && j < len; ++j) {
            npy_half out = *((npy_half*)(fallback_buf + lanes) + j);
            if (sdst == 1) {
                dst16[j] = ::hwy::float16_t::FromBits(out);
            } else {
                dst16[j * sdst] = ::hwy::float16_t::FromBits(out);
            }
        }
        for (int j = 0; j < lanes / 2 && (lanes / 2 + j) < len; ++j) {
            npy_half out = *((npy_half*)(fallback_buf + lanes) + lanes / 2 + j);
            if (sdst == 1) {
                dst16[lanes / 2 + j] = ::hwy::float16_t::FromBits(out);
            } else {
                dst16[(lanes / 2 + j) * sdst] = ::hwy::float16_t::FromBits(out);
            }
        }
    }
    npyv_cleanup();
}
#endif

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

static void
avx512_cbrt_f16(const npy_half *src, npy_half *dst, npy_intp len)
{
    const int num_lanes = npyv_nlanes_f32;
    npyvh_f16 x, out;
    npyv_f32 x_ps, out_ps;
    for (; len > 0; len -= num_lanes, src += num_lanes, dst += num_lanes) {
        if (len >= num_lanes) {
            x       = npyvh_load_f16(src);
            x_ps    = npyv_cvt_f16_f32(x);
            out_ps  = __svml_cbrtf16(x_ps);
            out     = npyv_cvt_f32_f16(out_ps, 0);
            npyvh_store_f16(dst, out);
        }
        else {
            x       = npyvh_load_till_f16(src, len, 0);
            x_ps    = npyv_cvt_f16_f32(x);
            out_ps  = __svml_cbrtf16(x_ps);
            out     = npyv_cvt_f32_f16(out_ps, 0);
            npyvh_store_till_f16(dst, len, out);
        }
    }
    npyv_cleanup();
}
#endif

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(HALF_sqrt)(char **args, npy_intp const *dimensions,
                                   npy_intp const *steps,
                                   void *NPY_UNUSED(data))
{
#if NPY_SIMD_FMA3 && SIMD_ARM
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        steps[0] % sizeof(npy_half) == 0 &&
        steps[1] % sizeof(npy_half) == 0)
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_half);
        const npy_intp sdst = steps[1] / sizeof(npy_half);
        simd_unary_f16_impl(src, ssrc, dst, sdst, len, [](auto x) { return hn::Sqrt(x); });
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_sqrtf(in1));
    }
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(HALF_cbrt)(char **args, npy_intp const *dimensions,
                                   npy_intp const *steps,
                                   void *NPY_UNUSED(data))
{
#if NPY_SIMD_FMA3 && SIMD_ARM
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        steps[0] % sizeof(npy_half) == 0 &&
        steps[1] % sizeof(npy_half) == 0)
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_half);
        const npy_intp sdst = steps[1] / sizeof(npy_half);
        simd_unary_f16_impl(src, ssrc, dst, sdst, len, [](auto x) { return hwy_cbrt_f32_impl(x); });
        return;
    }
#elif NPY__SVML_IS_ENABLED
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];

    const npy_intp len = dimensions[0];

    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        (steps[0] == sizeof(npy_half)) &&
        (steps[1] == sizeof(npy_half))) {
    #ifdef NPY_HAVE_AVX512_SPR
        __svml_cbrts32(src, dst, len);
    #else
        avx512_cbrt_f16(src, dst, len);
    #endif
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_cbrtf(in1));
    }
}

#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
static void
simd_cbrt_f32(const npyv_lanetype_f32 *src, npy_intp ssrc,
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
        npyv_f32 out = __svml_cbrtf16(x);
        if (sdst == 1) {
            npyv_store_till_f32(dst, len, out);
        } else {
            npyv_storen_till_f32(dst, sdst, len, out);
        }
    }
    npyv_cleanup();
}

static void
simd_cbrt_f64(const npyv_lanetype_f64 *src, npy_intp ssrc,
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
        npyv_f64 out = __svml_cbrt8_ha(x);
        if (sdst == 1) {
            npyv_store_till_f64(dst, len, out);
        } else {
            npyv_storen_till_f64(dst, sdst, len, out);
        }
    }
    npyv_cleanup();
}
#endif

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(FLOAT_cbrt)(char **args, npy_intp const *dimensions,
                                   npy_intp const *steps,
                                   void *NPY_UNUSED(data))
{
#if SIMD_ARM
    const npy_float *src = (npy_float*)args[0];
          npy_float *dst = (npy_float*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        steps[0] % sizeof(npy_float) == 0 &&
        steps[1] % sizeof(npy_float) == 0)
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);
        simd_unary_fp_impl<npy_float, 1>(src, ssrc, dst, sdst, len, [](auto x) { return hwy_cbrt_f32_impl(x); }, [](npy_float v) { return npy_cbrtf(v); });
        return;
    }
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
        simd_cbrt_f32(src, ssrc, dst, sdst, len);
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = *(npy_float *)ip1;
        *((npy_float *)op1) = npy_cbrtf(in1);
    }
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(DOUBLE_cbrt)(char **args, npy_intp const *dimensions,
                                     npy_intp const *steps,
                                     void *NPY_UNUSED(data))
{
#if SIMD_ARM
    const npy_double *src = (npy_double*)args[0];
          npy_double *dst = (npy_double*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        steps[0] % sizeof(npy_double) == 0 &&
        steps[1] % sizeof(npy_double) == 0)
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_double);
        const npy_intp sdst = steps[1] / sizeof(npy_double);
        simd_unary_fp_impl<npy_double, 4>(src, ssrc, dst, sdst, len, [](auto x) { return hwy_cbrt_f64_impl(x); }, [](npy_double v) { return npy_cbrt(v); });
        return;
    }
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
        simd_cbrt_f64(src, ssrc, dst, sdst, len);
        return;
    }
#endif
    UNARY_LOOP {
        const npy_double in1 = *(npy_double *)ip1;
        *((npy_double *)op1) = npy_cbrt(in1);
    }
}
