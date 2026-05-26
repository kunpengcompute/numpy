/**
 * Highway SIMD-optimized absolute for half-float type.
 *
 * Uses NumPy's CPU dispatch system (NPY_CPU_DISPATCH_CURFX) to compile
 * multiple target variants (SVE, ASIMD, NEON for ARM; X86_V4/V3/V2 for x86;
 * VSX2 for PowerPC; VX for Z Architecture; LSX for LoongArch; RVV for RISC-V).
 * Within each variant, Highway's HWY_STATIC_DISPATCH handles sub-target
 * selection at runtime (e.g. SVE vs SVE2, AVX2 vs AVX-512).
 *
 * Baseline (plain name, e.g. npy_highway_HALF_absolute_contig) acts as a
 * trampoline: it uses NPY_CPU_DISPATCH_CALL_XB to dispatch to the best
 * available target-specific variant at runtime.
 *
 * Contiguous: Highway SIMD absolute via bitwise AND with 0x7FFF mask.
 * Strided: SVE-specific gather/scatter for strided access on SVE targets only;
 *           scalar fallback for other targets (InsertLane/ExtractLane SIMD is
 *           slower than scalar for this simple bitwise AND operation).
 */

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdint.h>
#include <cstdint>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"

#include "npy_cpu_dispatch.h"
#include "loops_autovec_abs_hwy.dispatch.h"

#include <hwy/highway.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#include "loops_autovec_abs_hwy.h"

HWY_BEFORE_NAMESPACE();

namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <class T>
HWY_ATTR static void
HalfAbsoluteContig_SIMD(const T *HWY_RESTRICT in, T *HWY_RESTRICT out,
                        size_t count)
{
    const hn::ScalableTag<T> d;
    const size_t lanes = hn::Lanes(d);
    const auto v_mask = hn::Set(d, (T)0x7fff);
    size_t i = 0;

    for (; i + lanes <= count; i += lanes) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::And(v, v_mask), d, out + i);
    }

    for (; i < count; ++i) {
        out[i] = in[i] & (T)0x7fff;
    }
}

HWY_ATTR static void
HalfAbsolute_u16(const uint16_t *HWY_RESTRICT in, uint16_t *HWY_RESTRICT out,
                 size_t count)
{
    HalfAbsoluteContig_SIMD<uint16_t>(in, out, count);
}

#if defined(__ARM_FEATURE_SVE)
HWY_ATTR static void
HalfAbsoluteStrided_u16(const uint16_t *in, uint16_t *out,
                         npy_intp in_stride_elm, npy_intp out_stride_elm,
                         size_t count)
{
    uint64_t vl = svcntw();
    int64_t i = 0;

    int32_t ssrc_elm = (int32_t)in_stride_elm;
    int32_t sdst_elm = (int32_t)out_stride_elm;

    svint32_t v_idx_src = svindex_s32(0, ssrc_elm);
    svint32_t v_idx_dst = svindex_s32(0, sdst_elm);
    svuint32_t v_mask = svdup_u32(0x7fff);

    npy_intp jump_src = (npy_intp)vl * in_stride_elm * sizeof(uint16_t);
    npy_intp jump_dst = (npy_intp)vl * out_stride_elm * sizeof(uint16_t);

    svbool_t pg = svwhilelt_b32(i, (int64_t)count);

    while (svptest_any(svptrue_b32(), pg)) {
        svuint32_t v_data = svld1uh_gather_s32index_u32(pg, in, v_idx_src);
        v_data = svand_u32_x(pg, v_data, v_mask);
        svst1h_scatter_s32index_u32(pg, out, v_idx_dst, v_data);

        in = (const uint16_t *)((const char *)in + jump_src);
        out = (uint16_t *)((char *)out + jump_dst);
        i += (int64_t)vl;
        pg = svwhilelt_b32(i, (int64_t)count);
    }
}
#else
static void
HalfAbsoluteStrided_u16(const uint16_t *in, uint16_t *out,
                         npy_intp in_stride_elm, npy_intp out_stride_elm,
                         size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        out[i * out_stride_elm] = in[i * in_stride_elm] & 0x7fff;
    }
}
#endif

} // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();

extern "C" {

#ifdef NPY_MTARGETS_CURRENT
/*
 * Target-specific compilation (e.g. SVE).
 * NPY_CPU_DISPATCH_CURFX(func_name) produces e.g. func_name_SVE.
 * Uses HWY_STATIC_DISPATCH for Highway's own sub-target selection
 * (e.g. SVE vs SVE2).
 */
NPY_VISIBILITY_HIDDEN void
NPY_CPU_DISPATCH_CURFX(npy_highway_HALF_absolute_contig)(char **args,
                                                         npy_intp count)
{
    const uint16_t *in = reinterpret_cast<const uint16_t *>(args[0]);
    uint16_t *out = reinterpret_cast<uint16_t *>(args[1]);
    HWY_STATIC_DISPATCH(HalfAbsolute_u16)(in, out, (size_t)count);
}

NPY_VISIBILITY_HIDDEN void
NPY_CPU_DISPATCH_CURFX(npy_highway_HALF_absolute_strided)(char **args,
                                                          npy_intp src_step,
                                                          npy_intp dst_step,
                                                          npy_intp count)
{
    const uint16_t *in = reinterpret_cast<const uint16_t *>(args[0]);
    uint16_t *out = reinterpret_cast<uint16_t *>(args[1]);
    npy_intp in_stride_elm = src_step / (npy_intp)sizeof(uint16_t);
    npy_intp out_stride_elm = dst_step / (npy_intp)sizeof(uint16_t);
    HWY_STATIC_DISPATCH(HalfAbsoluteStrided_u16)(in, out, in_stride_elm,
                                                 out_stride_elm,
                                                 (size_t)count);
}
#else
/*
 * Baseline compilation: plain func_name is the caller-facing entry point.
 * Dispatches at runtime to the best available target (e.g. func_name_SVE)
 * via NPY_CPU_DISPATCH_CALL_XB. If no target is available, falls through
 * to HWY_STATIC_DISPATCH (typically NEON on ARM64).
 */
typedef void (*absolute_contig_func)(char **args, npy_intp count);
typedef void (*absolute_strided_func)(char **args, npy_intp src_step,
                                      npy_intp dst_step, npy_intp count);

NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_HALF_absolute_contig,
                            (char **args, npy_intp count));
NPY_CPU_DISPATCH_DECLARE_XB(void npy_highway_HALF_absolute_strided,
                            (char **args, npy_intp src_step,
                             npy_intp dst_step, npy_intp count));

NPY_VISIBILITY_HIDDEN void
NPY_CPU_DISPATCH_CURFX(npy_highway_HALF_absolute_contig)(char **args,
                                                         npy_intp count)
{
    absolute_contig_func _f = NULL;
    NPY_CPU_DISPATCH_CALL_XB(_f = npy_highway_HALF_absolute_contig);
    if (_f != NULL) {
        _f(args, count);
    } else {
        const uint16_t *in = reinterpret_cast<const uint16_t *>(args[0]);
        uint16_t *out = reinterpret_cast<uint16_t *>(args[1]);
        HWY_STATIC_DISPATCH(HalfAbsolute_u16)(in, out, (size_t)count);
    }
}

NPY_VISIBILITY_HIDDEN void
NPY_CPU_DISPATCH_CURFX(npy_highway_HALF_absolute_strided)(char **args,
                                                          npy_intp src_step,
                                                          npy_intp dst_step,
                                                          npy_intp count)
{
    absolute_strided_func _f = NULL;
    NPY_CPU_DISPATCH_CALL_XB(_f = npy_highway_HALF_absolute_strided);
    if (_f != NULL) {
        _f(args, src_step, dst_step, count);
    } else {
        const uint16_t *in = reinterpret_cast<const uint16_t *>(args[0]);
        uint16_t *out = reinterpret_cast<uint16_t *>(args[1]);
        npy_intp in_stride_elm = src_step / (npy_intp)sizeof(uint16_t);
        npy_intp out_stride_elm = dst_step / (npy_intp)sizeof(uint16_t);
        HWY_STATIC_DISPATCH(HalfAbsoluteStrided_u16)(in, out, in_stride_elm,
                                                     out_stride_elm,
                                                     (size_t)count);
    }
}
#endif

}