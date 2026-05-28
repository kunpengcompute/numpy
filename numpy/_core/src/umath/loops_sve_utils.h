#ifndef _NPY_UMATH_LOOPS_SVE_UTILS_H_
#define _NPY_UMATH_LOOPS_SVE_UTILS_H_

#include "npy_cpu_features.h"

#ifndef __has_include
    #define __has_include(header) 0
#endif

#if defined(__aarch64__) && __has_include(<arm_sve.h>)
    #include <arm_sve.h>
    #define NPY_HAVE_ARM_SVE_INTRINSICS 1
#else
    #define NPY_HAVE_ARM_SVE_INTRINSICS 0
#endif

#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    #define NPY_SVE_TARGET __attribute__((target("arch=armv8.2-a+sve")))
#else
    #define NPY_SVE_TARGET
#endif

static inline int
npy_sve_intrinsics_available(void)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    return npy_cpu_have(NPY_CPU_FEATURE_SVE);
#else
    return 0;
#endif
}

#endif /* _NPY_UMATH_LOOPS_SVE_UTILS_H_ */
