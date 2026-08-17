#ifndef NUMPY_CORE_SRC_MULTIARRAY_CAST_HWY_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CAST_HWY_H_

#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>


#ifdef __cplusplus
extern "C" {
#endif

NPY_VISIBILITY_HIDDEN int
npy_cast_hwy_supports(int src_type, int dst_type);

NPY_VISIBILITY_HIDDEN int
npy_cast_hwy_2d(int src_type, int dst_type,
        const char *src, npy_intp src_outer_stride,
        char *dst, npy_intp dst_outer_stride,
        npy_intp inner_count, npy_intp outer_count);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_CAST_HWY_H_ */
