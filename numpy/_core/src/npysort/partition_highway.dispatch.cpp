#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"

#include "npy_cpu_dispatch.h"
#include "npy_cpu_features.h"

#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "hwy/highway.h"

#include "partition_highway.hpp"

namespace np::highway::partition_simd {
namespace hn = hwy::HWY_NAMESPACE;

static constexpr npy_intp kMinPartitionItems = 1024;
static constexpr npy_intp kMaxStackBytes = 4096;

template <typename T, typename D, typename V>
HWY_ATTR HWY_INLINE auto
LessThanPivotMask(D d, V values, T pivot)
{
    return hn::Lt(values, hn::Set(d, pivot));
}

template <typename D, typename V>
HWY_ATTR HWY_INLINE auto
LessThanPivotMask(D d, V values, npy_double pivot)
{
    if (npy_isnan(pivot)) {
        return hn::Not(hn::IsNaN(values));
    }
    return hn::Lt(values, hn::Set(d, pivot));
}

template <typename T>
HWY_ATTR bool
DoPartition(T *HWY_RESTRICT v, npy_intp ll, npy_intp hh, T pivot,
            T *HWY_RESTRICT tmp, npy_intp *out_ll, npy_intp *out_hh)
{
    const hn::ScalableTag<T> d;
    const npy_intp lanes = static_cast<npy_intp>(hn::Lanes(d));
    const npy_intp n = hh - ll + 1;

    if (n < kMinPartitionItems || n < 2 * lanes) {
        return false;
    }

    T *base = v + ll;
    npy_intp lt = 0;
    npy_intp ge = n - 1;
    HWY_ALIGN T ge_block[HWY_MAX_LANES_D(hn::ScalableTag<T>)];

    npy_intp i = 0;
    for (; i + lanes <= n; i += lanes) {
        const auto values = hn::LoadU(d, base + i);
        const auto mask_lt = LessThanPivotMask(d, values, pivot);
        const auto mask_ge = hn::Not(mask_lt);

        const npy_intp n_lt =
                static_cast<npy_intp>(hn::CompressStore(values, mask_lt, d, tmp + lt));
        lt += n_lt;

        const npy_intp n_ge = lanes - n_lt;
        hn::CompressStore(values, mask_ge, d, ge_block);
        for (npy_intp k = n_ge - 1; k >= 0; --k) {
            tmp[ge--] = ge_block[k];
        }
    }

    for (; i < n; ++i) {
        const T x = base[i];
        bool is_lt;
        if constexpr (std::is_same_v<T, npy_double>) {
            is_lt = npy_isnan(pivot) ? !npy_isnan(x) : x < pivot;
        }
        else {
            is_lt = x < pivot;
        }
        if (is_lt) {
            tmp[lt++] = x;
        }
        else {
            tmp[ge--] = x;
        }
    }

    /*
     * Avoid pathological duplicate-heavy partitions.  In particular, a uniform
     * segment would otherwise make only one element of progress per iteration.
     */
    if (lt == 0 || lt == n) {
        return false;
    }

    std::memcpy(base, tmp, static_cast<size_t>(n) * sizeof(T));
    *out_ll = ll + lt;
    *out_hh = ll + lt - 1;
    return true;
}

template <typename T>
static NPY_INLINE int
RunPartition(T *v, npy_intp ll, npy_intp hh, T pivot, T *tmp,
             npy_intp *out_ll, npy_intp *out_hh)
{
    const npy_intp n = hh - ll + 1;
    const npy_intp nbytes = n * static_cast<npy_intp>(sizeof(T));

    if (tmp != nullptr) {
        return DoPartition(v, ll, hh, pivot, tmp, out_ll, out_hh) ? 1 : 0;
    }
    if (nbytes <= kMaxStackBytes) {
        alignas(64) T stack_buf[kMaxStackBytes / sizeof(T)];
        return DoPartition(v, ll, hh, pivot, stack_buf, out_ll, out_hh) ? 1 : 0;
    }

    T *heap_buf = static_cast<T *>(std::malloc(static_cast<size_t>(nbytes)));
    if (heap_buf == nullptr) {
        return 0;
    }
    const bool ok = DoPartition(v, ll, hh, pivot, heap_buf, out_ll, out_hh);
    std::free(heap_buf);
    return ok ? 1 : 0;
}

int NPY_CPU_DISPATCH_CURFX(PartitionInt64)(
        npy_int64 *v, npy_intp ll, npy_intp hh, npy_int64 pivot,
        npy_int64 *tmp,
        npy_intp *out_ll, npy_intp *out_hh)
{
    return RunPartition(v, ll, hh, pivot, tmp, out_ll, out_hh);
}

int NPY_CPU_DISPATCH_CURFX(PartitionDouble)(
        npy_double *v, npy_intp ll, npy_intp hh, npy_double pivot,
        npy_double *tmp,
        npy_intp *out_ll, npy_intp *out_hh)
{
    return RunPartition(v, ll, hh, pivot, tmp, out_ll, out_hh);
}

} // namespace np::highway::partition_simd
