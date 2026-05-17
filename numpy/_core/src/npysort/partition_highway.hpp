#ifndef NUMPY_CORE_SRC_NPYSORT_PARTITION_HIGHWAY_HPP
#define NUMPY_CORE_SRC_NPYSORT_PARTITION_HIGHWAY_HPP

#include "numpy/ndarraytypes.h"
#include "npy_cpu_dispatch.h"

#include "partition_highway.dispatch.h"

namespace np::highway::partition_simd {

NPY_CPU_DISPATCH_DECLARE(
    int PartitionInt64,
    (npy_int64 *v, npy_intp ll, npy_intp hh, npy_int64 pivot,
     npy_intp *out_ll, npy_intp *out_hh)
)

NPY_CPU_DISPATCH_DECLARE(
    int PartitionDouble,
    (npy_double *v, npy_intp ll, npy_intp hh, npy_double pivot,
     npy_intp *out_ll, npy_intp *out_hh)
)

} // namespace np::highway::partition_simd

#endif // NUMPY_CORE_SRC_NPYSORT_PARTITION_HIGHWAY_HPP
