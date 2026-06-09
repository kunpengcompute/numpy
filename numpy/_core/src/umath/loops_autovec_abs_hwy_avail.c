/**
 * Availability check function for Highway SIMD absolute (half-float).
 *
 * This is compiled once (non-dispatched) as part of the _multiarray_umath
 * module and returns whether Highway SIMD is available for half-float
 * absolute. When NPY_HAVE_HIGHWAY is not defined (Highway disabled or
 * unavailable), always returns 0 so the .c.src template falls through
 * to scalar paths.
 */

#include "loops_autovec_abs_hwy.h"
#include "npy_cpu_features.h"

int
npy_highway_absolute_half_available(void)
{
#if defined(NPY_HAVE_HIGHWAY) && defined(__aarch64__)
    return 1;
#else
    return 0;
#endif
}

int
npy_highway_absolute_half_strided_available(void)
{
#ifdef NPY_HAVE_HIGHWAY
    return npy_cpu_have(NPY_CPU_FEATURE_SVE);
#else
    return 0;
#endif
}
