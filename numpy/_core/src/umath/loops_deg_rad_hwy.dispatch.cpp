#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_math.h"
#include "numpy/halffloat.h"
#include "fast_loop_macros.h"

#include "npy_cpu_dispatch.h"
#include "loops_deg_rad_hwy.dispatch.h"

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_ATTR static inline void UnaryMulContigImpl(const T* src, T* dst, size_t len, T c) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto vc = hn::Set(d, c);

    size_t i = 0;

    for (; i + 4 * N <= len; i += 4 * N) {
        const auto v0 = hn::LoadU(d, src + i + 0 * N);
        const auto v1 = hn::LoadU(d, src + i + 1 * N);
        const auto v2 = hn::LoadU(d, src + i + 2 * N);
        const auto v3 = hn::LoadU(d, src + i + 3 * N);

        hn::StoreU(hn::Mul(v0, vc), d, dst + i + 0 * N);
        hn::StoreU(hn::Mul(v1, vc), d, dst + i + 1 * N);
        hn::StoreU(hn::Mul(v2, vc), d, dst + i + 2 * N);
        hn::StoreU(hn::Mul(v3, vc), d, dst + i + 3 * N);
    }

    for (; i + N <= len; i += N) {
        const auto vx = hn::LoadU(d, src + i);
        hn::StoreU(hn::Mul(vx, vc), d, dst + i);
    }

    for (; i < len; ++i) {
        dst[i] = src[i] * c;
    }
}

HWY_ATTR void NpyDeg2RadFloat(const float* src, float* dst, size_t len) {
    UnaryMulContigImpl<float>(src, dst, len, 0.017453292519943295769f);
}

HWY_ATTR void NpyRad2DegFloat(const float* src, float* dst, size_t len) {
    UnaryMulContigImpl<float>(src, dst, len, 57.295779513082320877f);
}

HWY_ATTR void NpyDeg2RadDouble(const double* src, double* dst, size_t len) {
    UnaryMulContigImpl<double>(src, dst, len, 0.017453292519943295769);
}

HWY_ATTR void NpyRad2DegDouble(const double* src, double* dst, size_t len) {
    UnaryMulContigImpl<double>(src, dst, len, 57.295779513082320877);
}

}  // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();

extern "C" {

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(HALF_rad2deg)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *(npy_half *)op1 = npy_float_to_half(npy_rad2degf(in1));
    }
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(HALF_deg2rad)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *(npy_half *)op1 = npy_float_to_half(npy_deg2radf(in1));
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_rad2deg)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    {
        const npy_intp n = dimensions[0];
        const npy_intp is1 = steps[0];
        const npy_intp os1 = steps[1];
        if (is1 == (npy_intp)sizeof(npy_float) &&
            os1 == (npy_intp)sizeof(npy_float)) {
            const npy_float *ip = (const npy_float *)args[0];
            npy_float *op = (npy_float *)args[1];
            HWY_STATIC_DISPATCH(NpyRad2DegFloat)(ip, op, (size_t)n);
            return;
        }
    }
    UNARY_LOOP {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = npy_rad2degf(in1);
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_deg2rad)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    {
        const npy_intp n = dimensions[0];
        const npy_intp is1 = steps[0];
        const npy_intp os1 = steps[1];
        if (is1 == (npy_intp)sizeof(npy_float) &&
            os1 == (npy_intp)sizeof(npy_float)) {
            const npy_float *ip = (const npy_float *)args[0];
            npy_float *op = (npy_float *)args[1];
            HWY_STATIC_DISPATCH(NpyDeg2RadFloat)(ip, op, (size_t)n);
            return;
        }
    }
    UNARY_LOOP {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = npy_deg2radf(in1);
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_rad2deg)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    {
        const npy_intp n = dimensions[0];
        const npy_intp is1 = steps[0];
        const npy_intp os1 = steps[1];
        if (is1 == (npy_intp)sizeof(npy_double) &&
            os1 == (npy_intp)sizeof(npy_double)) {
            const npy_double *ip = (const npy_double *)args[0];
            npy_double *op = (npy_double *)args[1];
            HWY_STATIC_DISPATCH(NpyRad2DegDouble)(ip, op, (size_t)n);
            return;
        }
    }
    UNARY_LOOP {
        const npy_double in1 = *(npy_double *)ip1;
        *(npy_double *)op1 = npy_rad2deg(in1);
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_deg2rad)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    {
        const npy_intp n = dimensions[0];
        const npy_intp is1 = steps[0];
        const npy_intp os1 = steps[1];
        if (is1 == (npy_intp)sizeof(npy_double) &&
            os1 == (npy_intp)sizeof(npy_double)) {
            const npy_double *ip = (const npy_double *)args[0];
            npy_double *op = (npy_double *)args[1];
            HWY_STATIC_DISPATCH(NpyDeg2RadDouble)(ip, op, (size_t)n);
            return;
        }
    }
    UNARY_LOOP {
        const npy_double in1 = *(npy_double *)ip1;
        *(npy_double *)op1 = npy_deg2rad(in1);
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(LONGDOUBLE_rad2deg)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    UNARY_LOOP {
        const npy_longdouble in1 = *(npy_longdouble *)ip1;
        *(npy_longdouble *)op1 = npy_rad2degl(in1);
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(LONGDOUBLE_deg2rad)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    UNARY_LOOP {
        const npy_longdouble in1 = *(npy_longdouble *)ip1;
        *(npy_longdouble *)op1 = npy_deg2radl(in1);
    }
}

}  // extern "C"