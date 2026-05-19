#include "fast_loop_macros.h"
#include "loops.h"
#include "loops_utils.h"
#include "numpy/npy_math.h"

#include "simd/simd.h"
#include "simd/simd.hpp"
#include <hwy/highway.h>
#include <cstring>

namespace hn = hwy::HWY_NAMESPACE;

#if defined(__aarch64__) && NPY_SIMD_FMA3
#define SIMD_ARM 1
#endif

#if SIMD_ARM
static void HWY_ATTR
simd_sqrt_f16_impl(const npy_half *src, npy_intp ssrc, npy_half *dst, npy_intp sdst, npy_intp len)
{
    const ::hwy::float16_t* src16 = (const ::hwy::float16_t*)src;
    ::hwy::float16_t* dst16 = (::hwy::float16_t*)dst;

    const hn::ScalableTag<npy_float> f32_tag_t;
    const hn::ScalableTag<::hwy::float16_t> f16_tag_t;
    const hn::Half<decltype(f16_tag_t)> half_f16_tag_t;
    const int lanes = hn::Lanes(f16_tag_t);

    npy_half NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) fallback_buf[hn::MaxLanes(f16_tag_t) * 2];

    for (; len > 0; len -= lanes, src16 += ssrc * lanes, dst16 += sdst * lanes) {
        hn::Vec<decltype(f16_tag_t)> x16_in;
        if (ssrc == 1) {
            x16_in = hn::LoadN(f16_tag_t, src16, len);
        } else {
            ::hwy::float16_t tmp[hn::MaxLanes(f16_tag_t)];
            for (int j = 0; j < lanes && j < len; ++j) {
                tmp[j] = src16[j * ssrc];
            }
            x16_in = hn::LoadN(f16_tag_t, tmp, lanes);
        }

        hn::Store(x16_in, f16_tag_t, (::hwy::float16_t*)fallback_buf);

        auto x32_in0 = hn::PromoteLowerTo(f32_tag_t, x16_in);
        auto x32_in1 = hn::PromoteUpperTo(f32_tag_t, x16_in);

        auto y32_out0 = hn::Sqrt(x32_in0);
        auto y32_out1 = hn::Sqrt(x32_in1);

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
        simd_sqrt_f16_impl(src, ssrc, dst, sdst, len);
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_sqrtf(in1));
    }
}