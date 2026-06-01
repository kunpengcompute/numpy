#ifndef _NPY_UMATH_LOOPS_BITWISE_SVE_H_
#define _NPY_UMATH_LOOPS_BITWISE_SVE_H_

#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

NPY_VISIBILITY_HIDDEN int
npy_bitwise_sve_available(void);

NPY_VISIBILITY_HIDDEN void
npy_bitwise_and_sve_i64(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_bitwise_sve_contig(char **args, npy_intp len, int itemsize, int op);

NPY_VISIBILITY_HIDDEN void
npy_bitwise_sve_scalar_in0(char **args, npy_intp len, int itemsize, int op);

NPY_VISIBILITY_HIDDEN void
npy_bitwise_sve_scalar_in1(char **args, npy_intp len, int itemsize, int op);

NPY_VISIBILITY_HIDDEN void
npy_logical_xor_sve_u8(char **args, npy_intp len);

NPY_VISIBILITY_HIDDEN void
npy_logical_xor_sve_contig(char **args, npy_intp len, int itemsize);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_UMATH_LOOPS_BITWISE_SVE_H_ */
