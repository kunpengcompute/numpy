#ifndef _NPY_UMATH_LOOPS_LOGICAL_SVE_H_
#define _NPY_UMATH_LOOPS_LOGICAL_SVE_H_

#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

NPY_VISIBILITY_HIDDEN int
npy_logical_or_sve_scalar_in0_available(void);

NPY_VISIBILITY_HIDDEN void
npy_logical_or_sve_scalar_in0(
        char **args, npy_intp const *dimensions, npy_intp const *steps,
        int itemsize);

NPY_VISIBILITY_HIDDEN void
npy_logical_or_sve_scalar_in1(
        char **args, npy_intp const *dimensions, npy_intp const *steps,
        int itemsize);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_UMATH_LOOPS_LOGICAL_SVE_H_ */
