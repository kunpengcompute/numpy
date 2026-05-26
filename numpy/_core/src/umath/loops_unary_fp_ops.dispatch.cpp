#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
#include "lowlevel_strided_loops.h"
#include "fast_loop_macros.h"
#include "simd/simd.hpp"
#include <hwy/highway.h>
#include <type_traits>
#include <cmath>

namespace {
using namespace np::simd;

/*******************************************************************************
 ** Unary operation traits -- define per-op behaviour here.
 ** To add a new unary op, specialize UnaryOpTraits<tag_t> with:
 **   - op(Vec<T>)    : the SIMD operation
 **   - tail_fill<T>() : Vec<T> fill value for tail lanes (default: Zero)
 **   - scalar_op(T)  : scalar fallback
 ******************************************************************************/
template<typename Op>
struct UnaryOpTraits;

struct reciprocal_t {};

template<>
struct UnaryOpTraits<reciprocal_t> {
#if NPY_HWY
    template<typename T>
    static HWY_ATTR HWY_INLINE Vec<T> simd_op(Vec<T> a)
    {
        if constexpr (std::is_same_v<T, hwy::float16_t>) {
            return Div(Set(hwy::F16FromF32(1.0f)), a);
        } else {
            return Div(Set(T(1.0)), a);
        }
    }
#endif

    template<typename T>
    static inline auto scalar_op(T a)
    {
#if NPY_HWY_F16
        if constexpr (std::is_same_v<T, hwy::float16_t>) {
            return hwy::F16FromF32(1.0f / hwy::F32FromF16(a));
        }
#endif
        return T(1.0) / a;
    }
};

struct ceil_t {};
struct floor_t {};
struct trunc_t {};
struct rint_t {};

template<>
struct UnaryOpTraits<ceil_t> {
#if NPY_HWY
    template<typename T>
    static HWY_ATTR HWY_INLINE Vec<T> simd_op(Vec<T> a)
    {
        return Ceil(a);
    }
#endif

    template<typename T>
    static inline T scalar_op(T a)
    {
#if NPY_HWY_F16
        if constexpr (std::is_same_v<T, hwy::float16_t>) {
            return hwy::F16FromF32(std::ceil(hwy::F32FromF16(a)));
        }
#endif
        return static_cast<T>(std::ceil(a));
    }
};

template<>
struct UnaryOpTraits<floor_t> {
#if NPY_HWY
    template<typename T>
    static HWY_ATTR HWY_INLINE Vec<T> simd_op(Vec<T> a)
    {
        return Floor(a);
    }
#endif

    template<typename T>
    static inline T scalar_op(T a)
    {
#if NPY_HWY_F16
        if constexpr (std::is_same_v<T, hwy::float16_t>) {
            return hwy::F16FromF32(std::floor(hwy::F32FromF16(a)));
        }
#endif
        return static_cast<T>(std::floor(a));
    }
};

template<>
struct UnaryOpTraits<trunc_t> {
#if NPY_HWY
    template<typename T>
    static HWY_ATTR HWY_INLINE Vec<T> simd_op(Vec<T> a)
    {
        return Trunc(a);
    }
#endif

    template<typename T>
    static inline T scalar_op(T a)
    {
#if NPY_HWY_F16
        if constexpr (std::is_same_v<T, hwy::float16_t>) {
            return hwy::F16FromF32(std::trunc(hwy::F32FromF16(a)));
        }
#endif
        return static_cast<T>(std::trunc(a));
    }
};

template<>
struct UnaryOpTraits<rint_t> {
#if NPY_HWY
    template<typename T>
    static HWY_ATTR HWY_INLINE Vec<T> simd_op(Vec<T> a)
    {
        return Round(a);
    }
#endif

    template<typename T>
    static inline T scalar_op(T a)
    {
#if NPY_HWY_F16
        if constexpr (std::is_same_v<T, hwy::float16_t>) {
            return hwy::F16FromF32(std::rint(hwy::F32FromF16(a)));
        }
#endif
        return static_cast<T>(std::rint(a));
    }
};

#if NPY_HWY

template<typename Op, typename T>
HWY_ATTR HWY_INLINE Vec<T> unary_tail_fill()
{
    return Zero<T>();
}

template<>
HWY_ATTR HWY_INLINE Vec<float> unary_tail_fill<reciprocal_t, float>()
{
    return Set(1.0f);
}

template<>
HWY_ATTR HWY_INLINE Vec<double> unary_tail_fill<reciprocal_t, double>()
{
    return Set(1.0);
}

#if NPY_HWY_F16
template<>
HWY_ATTR HWY_INLINE Vec<hwy::float16_t> unary_tail_fill<reciprocal_t, hwy::float16_t>()
{
    return Set(hwy::F16FromF32(1.0f));
}
#endif

template<typename T>
HWY_ATTR HWY_INLINE auto make_index(npy_intp stride_el)
{
    auto tag = _SignedTag<T>{};
    using S = hn::TFromD<decltype(tag)>;
    return hn::Mul(hn::Iota(tag, S{0}), hn::Set(tag, static_cast<S>(stride_el)));
}

// ---- Generic unary kernel templates (parameterized on Op + T) ----

// CONTIG_CONTIG
template<typename Op, typename T>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_unary_cc(T* op, const T* ip, npy_intp len)
{
    using Traits = UnaryOpTraits<Op>;
    constexpr int UNROLL = 8;
    HWY_LANES_CONSTEXPR int vstep = Lanes(T{});
    const int wstep = vstep * UNROLL;
    auto fill = unary_tail_fill<Op, T>();
    for (; len >= wstep; len -= wstep, ip += wstep, op += wstep) {
        for (int i = 0; i < UNROLL; i++) {
            StoreU(Traits::template simd_op<T>(LoadU(ip + vstep * i)), op + vstep * i);
        }
    }
    for (; len >= vstep; len -= vstep, ip += vstep, op += vstep) {
        StoreU(Traits::template simd_op<T>(LoadU(ip)), op);
    }
    if (len > 0) {
        StoreN(Traits::template simd_op<T>(LoadNOr(fill, ip, len)), op, len);
    }
}

// NCONTIG_CONTIG
template<typename Op, typename T>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_unary_nc(T* op, const T* ip, npy_intp is, npy_intp len)
{
    using Traits = UnaryOpTraits<Op>;
    constexpr int UNROLL = 4;
    HWY_LANES_CONSTEXPR int vstep = Lanes(T{});
    const int wstep = vstep * UNROLL;
    auto fill = unary_tail_fill<Op, T>();
    auto idx = make_index<T>(is);
    for (; len >= wstep; len -= wstep, ip += is * wstep, op += wstep) {
        for (int i = 0; i < UNROLL; i++) {
            StoreU(Traits::template simd_op<T>(GatherIndex(ip + is * vstep * i, idx)), op + vstep * i);
        }
    }
    for (; len >= vstep; len -= vstep, ip += is * vstep, op += vstep) {
        StoreU(Traits::template simd_op<T>(GatherIndex(ip, idx)), op);
    }
    if (len > 0) {
        StoreN(Traits::template simd_op<T>(GatherIndexNOr(fill, ip, idx, len)), op, len);
    }
}

// CONTIG_NCONTIG
template<typename Op, typename T>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_unary_cn(T* op, npy_intp os, const T* ip, npy_intp len)
{
    using Traits = UnaryOpTraits<Op>;
    constexpr int UNROLL = 4;
    HWY_LANES_CONSTEXPR int vstep = Lanes(T{});
    const int wstep = vstep * UNROLL;
    auto fill = unary_tail_fill<Op, T>();
    auto idx = make_index<T>(os);
    for (; len >= wstep; len -= wstep, ip += wstep, op += os * wstep) {
        for (int i = 0; i < UNROLL; i++) {
            ScatterIndex(Traits::template simd_op<T>(LoadU(ip + vstep * i)), op + os * vstep * i, idx);
        }
    }
    for (; len >= vstep; len -= vstep, ip += vstep, op += os * vstep) {
        ScatterIndex(Traits::template simd_op<T>(LoadU(ip)), op, idx);
    }
    if (len > 0) {
        ScatterIndexN(Traits::template simd_op<T>(LoadNOr(fill, ip, len)), op, idx, len);
    }
}

// NCONTIG_NCONTIG
template<typename Op, typename T>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_unary_nn(T* op, npy_intp os, const T* ip, npy_intp is, npy_intp len)
{
    using Traits = UnaryOpTraits<Op>;
    constexpr int UNROLL = 4;
    HWY_LANES_CONSTEXPR int vstep = Lanes(T{});
    const int wstep = vstep * UNROLL;
    auto fill = unary_tail_fill<Op, T>();
    auto src_idx = make_index<T>(is);
    auto dst_idx = make_index<T>(os);
    for (; len >= wstep; len -= wstep, ip += is * wstep, op += os * wstep) {
        for (int i = 0; i < UNROLL; i++) {
            ScatterIndex(Traits::template simd_op<T>(GatherIndex(ip + is * vstep * i, src_idx)), op + os * vstep * i, dst_idx);
        }
    }
    for (; len >= vstep; len -= vstep, ip += is * vstep, op += os * vstep) {
        ScatterIndex(Traits::template simd_op<T>(GatherIndex(ip, src_idx)), op, dst_idx);
    }
    if (len > 0) {
        ScatterIndexN(Traits::template simd_op<T>(GatherIndexNOr(fill, ip, src_idx, len)), op, dst_idx, len);
    }
}

// ---- Generic unary dispatch / wrapper (parameterized on Op + T) ----

template<typename Op, typename T>
static NPY_INLINE int run_unary_simd(
    char** args, npy_intp const* dimensions, npy_intp const* steps)
{
    npy_intp len = dimensions[0];
    if (is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        return 0;
    }

    npy_intp ssrc = steps[0];
    npy_intp sdst = steps[1];
    const npy_intp esize = sizeof(T);

    if (!((ssrc & (esize - 1)) == 0 && llabs(ssrc) < MAX_STEP_SIZE) ||
        !((sdst & (esize - 1)) == 0 && llabs(sdst) < MAX_STEP_SIZE)) {
        return 0;
    }
    // Highway gather/scatter uses signed index vectors with the same
    // bit width as the lane type. For 16-bit lanes (e.g. float16_t),
    // the index type is int16_t; ensure element strides don't overflow.
    if constexpr (sizeof(T) == 2) {
        HWY_LANES_CONSTEXPR int vstep = Lanes(T{});
        if (llabs(ssrc / esize) * (vstep - 1) > INT16_MAX ||
            llabs(sdst / esize) * (vstep - 1) > INT16_MAX) {
            return 0;
        }
    }

    if (sdst == esize && ssrc == esize) {
        simd_unary_cc<Op, T>((T*)args[1], (const T*)args[0], len);
    } else if (sdst == esize) {
        simd_unary_nc<Op, T>((T*)args[1], (const T*)args[0], ssrc / esize, len);
    } else if (ssrc == esize) {
        simd_unary_cn<Op, T>((T*)args[1], sdst / esize, (const T*)args[0], len);
    } else {
        simd_unary_nn<Op, T>((T*)args[1], sdst / esize, (const T*)args[0], ssrc / esize, len);
    }
    return 1;
}
#endif  // NPY_HWY

template<typename Op, typename T>
void unary_wrapper(char** args, npy_intp const* dimensions, npy_intp const* steps)
{
#if NPY_HWY
    if (run_unary_simd<Op, T>(args, dimensions, steps)) {
        return;
    }
#endif

    using Traits = UnaryOpTraits<Op>;
    npy_intp len = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp ssrc = steps[0], sdst = steps[1];

    const T *ip = (const T*)src;
    T *op = (T*)dst;
    npy_intp is = ssrc / sizeof(T), os = sdst / sizeof(T);
    for (; len > 0; --len, ip += is, op += os) {
        *op = Traits::template scalar_op<T>(*ip);
    }
}

} // namespace anonymous

/*******************************************************************************
 ** Exported ufunc inner functions
 ******************************************************************************/

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_reciprocal)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_wrapper<reciprocal_t, npy_float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_reciprocal)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_wrapper<reciprocal_t, npy_double>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_reciprocal)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
#if NPY_HWY_F16
    unary_wrapper<reciprocal_t, hwy::float16_t>(args, dimensions, steps);
#else
    UNARY_LOOP {
        const float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(1.0f / in1);
    }
#endif
}

/*******************************************************************************
 ** Complex reciprocal -- Aho-Weinberger algorithm with Highway SIMD
 **
 ** 1/(a+bi) uses pivot on max(|re|, |im|) to avoid overflow:
 **   If |im| <= |re|:  r = im/re, d = re + im*r → [1/d, -r/d]
 **   Else:             r = re/im, d = re*r + im → [r/d, -1/d]
 **
 ** SIMD: compute both branches, blend via IfThenElse mask.
 ******************************************************************************/

#if NPY_HWY

template<typename T>
HWY_ATTR static void simd_creciprocal_cc(T* op, const T* ip, npy_intp len)
{
    using D = _Tag<T>;
    const D tag;
    HWY_LANES_CONSTEXPR int vstep = Lanes(T{});
    for (; len >= vstep; len -= vstep, ip += 2 * vstep, op += 2 * vstep) {
        Vec<T> re, im;
        hn::LoadInterleaved2(tag, ip, re, im);
        auto use_b2 = Gt(Abs(im), Abs(re));
        auto one = Set(T{1});
        auto left = IfThenElse(use_b2, re, im);
        auto right = IfThenElse(use_b2, im, re);
        auto r = Div(left, right);
        auto d = MulAdd(left, r, right);
        auto div1 = IfThenElse(use_b2, r, one);
        auto res1 = Div(div1, d);
        auto div2 = IfThenElse(use_b2, one, r);
        auto res2 = Neg(Div(div2, d));

        hn::StoreInterleaved2(res1, res2, tag, op);
    }

    for (; len > 0; len--, ip += 2, op += 2) {
        const T in1r = ip[0];
        const T in1i = ip[1];
        if (std::abs(in1i) <= std::abs(in1r)) {
            const T r = in1i / in1r;
            const T d = in1r + in1i * r;
            op[0] = 1 / d;
            op[1] = -r / d;
        } else {
            const T r = in1r / in1i;
            const T d = in1r * r + in1i;
            op[0] = r / d;
            op[1] = -1 / d;
        }
    }
}
#endif  // NPY_HWY

template<typename FTYPE, typename CTYPE>
void creciprocal_wrapper(char** args, npy_intp const* dimensions, npy_intp const* steps)
{
#if NPY_HWY
    npy_intp len = dimensions[0];
    if (!is_mem_overlap(args[1], steps[1], args[0], steps[0], len)) {
        npy_intp ssrc = steps[0];
        npy_intp sdst = steps[1];
        const npy_intp esize = sizeof(CTYPE);
        if (ssrc == esize && sdst == esize) {
            simd_creciprocal_cc<FTYPE>((FTYPE*)(args[1]), (const FTYPE*)(args[0]), len);
            return;
        }
    }
#endif

    UNARY_LOOP {
        const FTYPE in1r = ((FTYPE *)ip1)[0];
        const FTYPE in1i = ((FTYPE *)ip1)[1];
        if (std::abs(in1i) <= std::abs(in1r)) {
            const FTYPE r = in1i / in1r;
            const FTYPE d = in1r + in1i * r;
            ((FTYPE *)op1)[0] = 1 / d;
            ((FTYPE *)op1)[1] = -r / d;
        } else {
            const FTYPE r = in1r / in1i;
            const FTYPE d = in1r * r + in1i;
            ((FTYPE *)op1)[0] = r / d;
            ((FTYPE *)op1)[1] = -1 / d;
        }
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(CFLOAT_reciprocal)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    creciprocal_wrapper<npy_float, npy_cfloat>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(CDOUBLE_reciprocal)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    creciprocal_wrapper<npy_double, npy_cdouble>(args, dimensions, steps);
}

/*******************************************************************************
 ** Exported ufunc inner functions for ceil, floor, trunc
 ******************************************************************************/

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_ceil)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_wrapper<ceil_t, npy_float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_ceil)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_wrapper<ceil_t, npy_double>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_ceil)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
#if NPY_HWY_F16
    unary_wrapper<ceil_t, hwy::float16_t>(args, dimensions, steps);
#else
    UNARY_LOOP {
        const float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_ceilf(in1));
    }
#endif
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_floor)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_wrapper<floor_t, npy_float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_floor)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_wrapper<floor_t, npy_double>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_floor)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
#if NPY_HWY_F16
    unary_wrapper<floor_t, hwy::float16_t>(args, dimensions, steps);
#else
    UNARY_LOOP {
        const float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_floorf(in1));
    }
#endif
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_trunc)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_wrapper<trunc_t, npy_float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_trunc)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_wrapper<trunc_t, npy_double>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_trunc)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
#if NPY_HWY_F16
    unary_wrapper<trunc_t, hwy::float16_t>(args, dimensions, steps);
#else
    UNARY_LOOP {
        const float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_truncf(in1));
    }
#endif
}

/*******************************************************************************
 ** Exported ufunc inner functions for rint
 ******************************************************************************/

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_rint)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_wrapper<rint_t, npy_float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_rint)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_wrapper<rint_t, npy_double>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_rint)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
#if NPY_HWY_F16
    unary_wrapper<rint_t, hwy::float16_t>(args, dimensions, steps);
#else
    UNARY_LOOP {
        const float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_rintf(in1));
    }
#endif
}
