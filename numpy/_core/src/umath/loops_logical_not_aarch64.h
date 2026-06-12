#ifndef _NPY_UMATH_LOOPS_LOGICAL_NOT_AARCH64_H_
#define _NPY_UMATH_LOOPS_LOGICAL_NOT_AARCH64_H_

#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

NPY_VISIBILITY_HIDDEN int
npy_half_logical_not_aarch64(
        char **args, npy_intp const *dimensions, npy_intp const *steps);

NPY_VISIBILITY_HIDDEN int
npy_float_logical_not_aarch64(
        char **args, npy_intp const *dimensions, npy_intp const *steps,
        int itemsize);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_UMATH_LOOPS_LOGICAL_NOT_AARCH64_H_ */
