/**
 * Header for Highway SIMD absolute wrapper functions (half-float).
 *
 * Declares extern "C" functions callable from C code. The implementations
 * (in loops_autovec_abs_hwy.dispatch.cpp) use NumPy's CPU dispatch
 * to compile per-target variants (SVE, ASIMD, NEON for ARM;
 * X86_V4/V3/V2 for x86; VSX2 for PowerPC; VX for Z Architecture;
 * LSX for LoongArch; RVV for RISC-V).
 * Each variant internally uses Highway's HWY_STATIC_DISPATCH for
 * sub-target selection.
 *
 * Callers use the plain function names (e.g. npy_highway_HALF_absolute_contig).
 * The baseline function checks for available targets at runtime via
 * NPY_CPU_DISPATCH_CALL_XB; if a target-specific version is available
 * (e.g. func_name_SVE) it dispatches to it, otherwise falls through to
 * the baseline HWY_STATIC_DISPATCH (typically NEON on ARM64).
 *
 * Supports half-float (npy_half / uint16_t) for absolute value.
 */

#ifndef _NPY_UMATH_LOOPS_AUTOVEC_ABS_HWY_H_
#define _NPY_UMATH_LOOPS_AUTOVEC_ABS_HWY_H_

#include "numpy/ndarraytypes.h"

#ifdef __cplusplus
extern "C" {
#endif

NPY_VISIBILITY_HIDDEN int
npy_highway_absolute_half_available(void);

NPY_VISIBILITY_HIDDEN void
npy_highway_HALF_absolute_contig(char **args, npy_intp count);

NPY_VISIBILITY_HIDDEN int
npy_highway_absolute_half_strided_available(void);

NPY_VISIBILITY_HIDDEN void
npy_highway_HALF_absolute_strided(char **args, npy_intp src_step,
                                  npy_intp dst_step, npy_intp count);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_UMATH_LOOPS_AUTOVEC_ABS_HWY_H_ */