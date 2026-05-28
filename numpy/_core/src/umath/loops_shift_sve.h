#ifndef _NPY_UMATH_LOOPS_SHIFT_SVE_H_
#define _NPY_UMATH_LOOPS_SHIFT_SVE_H_

#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

NPY_VISIBILITY_HIDDEN int
npy_shift_sve_scalar_available(void);

NPY_VISIBILITY_HIDDEN void
npy_left_shift_sve_contig(char **args, npy_intp len, int itemsize, int is_signed);

NPY_VISIBILITY_HIDDEN void
npy_right_shift_sve_contig(char **args, npy_intp len, int itemsize, int is_signed);

#define NPY_DECL_SHIFT_SVE(TYPE) \
    NPY_VISIBILITY_HIDDEN void \
    TYPE##_left_shift_sve_scalar_in0(char **args, npy_intp len); \
    NPY_VISIBILITY_HIDDEN void \
    TYPE##_right_shift_sve_scalar_in0(char **args, npy_intp len)

NPY_DECL_SHIFT_SVE(BYTE);
NPY_DECL_SHIFT_SVE(UBYTE);
NPY_DECL_SHIFT_SVE(SHORT);
NPY_DECL_SHIFT_SVE(USHORT);
NPY_DECL_SHIFT_SVE(LONG);
NPY_DECL_SHIFT_SVE(ULONG);
NPY_DECL_SHIFT_SVE(LONGLONG);
NPY_DECL_SHIFT_SVE(ULONGLONG);

#undef NPY_DECL_SHIFT_SVE

#ifdef __cplusplus
}
#endif

#endif /* _NPY_UMATH_LOOPS_SHIFT_SVE_H_ */
