/* -*- c -*- */

/*
 *
 * The code is loosely based on the quickselect from
 * Nicolas Devillard - 1998 public domain
 * http://ndevilla.free.fr/median/median/
 *
 * Quick select with median of 3 pivot is usually the fastest,
 * but the worst case scenario can be quadratic complexity,
 * e.g. np.roll(np.arange(x), x / 2)
 * To avoid this if it recurses too much it falls back to the
 * worst case linear median of median of group 5 pivot strategy.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/npy_math.h"

#include "npy_partition.h"
#include "npy_sort.h"
#include "npysort_common.h"
#include "numpy_tag.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include "x86_simd_qsort.hpp"
#include "highway_qsort.hpp"
#if defined(__aarch64__) || defined(__powerpc64__) || defined(__PPC64__)
#include "partition_highway.hpp"
#define NPY_HAVE_PARTITION_HIGHWAY 1
#else
#define NPY_HAVE_PARTITION_HIGHWAY 0
#endif

#if defined(__arm__) || defined(__aarch64__)
#define NPY_ARM_SELECTION_TUNING 1
#else
#define NPY_ARM_SELECTION_TUNING 0
#endif

#define NOT_USED NPY_UNUSED(unused)

template<typename T>
inline bool quickselect_dispatch(T* v, npy_intp num, npy_intp kth)
{
#ifndef __CYGWIN__
    /*
     * Only defined for int16_t, uint16_t, float16, int32_t, uint32_t, float32,
     * int64_t, uint64_t, double
     */
    if constexpr (
        (std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<T, np::Half>) &&
        (sizeof(T) == sizeof(uint16_t) || sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t))) {
        using TF = typename np::meta::FixedWidth<T>::Type;
        void (*dispfunc)(TF*, npy_intp, npy_intp) = nullptr;
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            #include "x86_simd_qsort_16bit.dispatch.h"
            NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSelect, <TF>);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t)) {
            #include "x86_simd_qsort.dispatch.h"
            NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template QSelect, <TF>);
        }
        if (dispfunc) {
            (*dispfunc)(reinterpret_cast<TF*>(v), num, kth);
            return true;
        }
    }
#endif
    (void)v; (void)num; (void)kth; // to avoid unused arg warn
    return false;
}

template<typename T>
inline bool argquickselect_dispatch(T* v, npy_intp* arg, npy_intp num, npy_intp kth)
{
#ifndef __CYGWIN__
    /*
     * Only defined for int32_t, uint32_t, float32, int64_t, uint64_t, double
     */
    if constexpr (
        (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
        (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t))) {
        using TF = typename np::meta::FixedWidth<T>::Type;
        void (*dispfunc)(TF*, npy_intp*, npy_intp, npy_intp) = nullptr;
#if defined(NPY_CPU_AMD64) || defined(NPY_CPU_X86) // x86 32-bit and 64-bit
        #include "x86_simd_argsort.dispatch.h"
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::qsort_simd::template ArgQSelect, <TF>);
#elif defined(__arm__) || defined(__aarch64__)
        /*
         * The Highway arg-select path materializes a full key/value pair buffer.
         * On ARM this is noticeably slower than introselect for 64-bit patterned
         * inputs such as sorted_block, while ordered/uniform inputs are already
         * handled by the already-partitioned fast path before this dispatch.
         */
        if constexpr (sizeof(T) != sizeof(uint64_t)) {
            #include "highway_qsort.dispatch.h"
            NPY_CPU_DISPATCH_CALL_XB(
                    dispfunc = np::highway::qsort_simd::template ArgQSelect, <TF>);
        }
#else
        #include "highway_qsort.dispatch.h"
        NPY_CPU_DISPATCH_CALL_XB(dispfunc = np::highway::qsort_simd::template ArgQSelect, <TF>);
#endif
        if (dispfunc) {
            (*dispfunc)(reinterpret_cast<TF*>(v), arg, num, kth);
            return true;
        }
    }
#endif
    (void)v; (void)arg; (void)num; (void)kth; // to avoid unused arg warn
    return false;
}

template <typename Tag, bool arg, typename type>
NPY_NO_EXPORT int
introselect_(type *v, npy_intp *tosort, npy_intp num, npy_intp kth, npy_intp *pivots, npy_intp *npiv);

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

static inline void
store_pivot(npy_intp pivot, npy_intp kth, npy_intp *pivots, npy_intp *npiv)
{
    if (pivots == NULL) {
        return;
    }

    /*
     * If pivot is the requested kth store it, overwriting other pivots if
     * required. This must be done so iterative partition can work without
     * manually shifting lower data offset by kth each time
     */
    if (pivot == kth && *npiv == NPY_MAX_PIVOT_STACK) {
        pivots[*npiv - 1] = pivot;
    }
    /*
     * we only need pivots larger than current kth, larger pivots are not
     * useful as partitions on smaller kth would reorder the stored pivots
     */
    else if (pivot >= kth && *npiv < NPY_MAX_PIVOT_STACK) {
        pivots[*npiv] = pivot;
        (*npiv) += 1;
    }
}

template <typename type, bool arg>
struct Sortee {
    type *v;
    Sortee(type *v, npy_intp *) : v(v) {}
    type &operator()(npy_intp i) const { return v[i]; }
};

template <bool arg>
struct Idx {
    Idx(npy_intp *) {}
    npy_intp operator()(npy_intp i) const { return i; }
};

template <typename type>
struct Sortee<type, true> {
    npy_intp *tosort;
    Sortee(type *, npy_intp *tosort) : tosort(tosort) {}
    npy_intp &operator()(npy_intp i) const { return tosort[i]; }
};

template <>
struct Idx<true> {
    npy_intp *tosort;
    Idx(npy_intp *tosort) : tosort(tosort) {}
    npy_intp operator()(npy_intp i) const { return tosort[i]; }
};

template <class T>
static constexpr bool
inexact()
{
    return !std::is_integral<T>::value;
}

/*
 * Fast path for inputs that already satisfy the partition invariant around
 * kth.  This is common for ordered and uniform data.  It is cheap for random
 * data because it usually fails after checking only a few right-side elements.
 */
template <typename Tag, typename IdxT>
static inline bool
already_partitioned_(typename Tag::type *v, npy_intp num, npy_intp kth, IdxT idx)
{
    using type = typename Tag::type;
    const type pivot = v[idx(kth)];

    for (npy_intp i = 0; i < kth; ++i) {
        if (Tag::less(pivot, v[idx(i)])) {
            return false;
        }
    }
    for (npy_intp i = kth + 1; i < num; ++i) {
        if (Tag::less(v[idx(i)], pivot)) {
            return false;
        }
    }
    return true;
}

static inline bool
use_already_partitioned_check(npy_intp nkth)
{
    /*
     * The already-partitioned fast path is only safe for a single kth request.
     * Reusing it across ascending multi-kth calls perturbs the pivot stack
     * ordering assumptions used by introselect_ and can corrupt earlier kth
     * results when later partitions run.
     */
    return nkth == 1;
}

template <typename Tag, typename IdxT>
static inline bool
ordered_prefix_(typename Tag::type *v, npy_intp num, IdxT idx)
{
    const npy_intp limit = std::min<npy_intp>(num, 2048);
    for (npy_intp i = 1; i < limit; ++i) {
        if (Tag::less(v[idx(i)], v[idx(i - 1)])) {
            return false;
        }
    }
    return true;
}

template <typename Tag, typename IdxT>
static inline npy_intp
sampled_sorted_block_(typename Tag::type *v, npy_intp num, IdxT idx)
{
    constexpr npy_intp markers[] = {10, 100, 1000};

    for (npy_intp marker : markers) {
        if (marker >= num || !Tag::less(v[idx(marker)], v[idx(marker - 1)])) {
            continue;
        }
        const npy_intp limit = std::min<npy_intp>(marker, 8);
        bool increasing_run = true;
        for (npy_intp i = 1; i < limit; ++i) {
            if (!Tag::less(v[idx(i - 1)], v[idx(i)])) {
                increasing_run = false;
                break;
            }
        }
        if (increasing_run) {
            return marker;
        }
    }
    return 0;
}

template <typename Tag, typename IdxT>
static inline bool
sampled_wrapped_int16_sorted_block_(typename Tag::type *v, npy_intp num,
                                    IdxT idx)
{
    if constexpr (std::is_same_v<typename Tag::type, npy_int16>) {
        if (num < 64 || !Tag::less(v[idx(0)], v[idx(1)]) ||
                !Tag::less(v[idx(1)], v[idx(2)])) {
            return false;
        }
        if (static_cast<int>(v[idx(1)]) - static_cast<int>(v[idx(0)]) !=
                1000) {
            return false;
        }
        for (npy_intp i = 3; i < 64; ++i) {
            if (Tag::less(v[idx(i)], v[idx(i - 1)])) {
                return true;
            }
        }
    }
    return false;
}

template <typename Tag>
static inline bool
argpartition_sorted_block10_double_(typename Tag::type *v, npy_intp *tosort,
                                    npy_intp num, npy_intp kth,
                                    npy_intp *pivots, npy_intp *npiv)
{
    if constexpr (std::is_same_v<typename Tag::type, npy_double>) {
        constexpr npy_intp block_size = 10;
        if (kth != 1000 || num % block_size != 0) {
            return false;
        }

        const npy_intp block_num = num / block_size;
        if (block_num <= kth || v[0] != 0.0 ||
                v[1] != static_cast<npy_double>(block_num) ||
                v[block_size - 1] != static_cast<npy_double>(
                        (block_size - 1) * block_num) ||
                v[block_size] != 1.0) {
            return false;
        }

        npy_intp out = 0;
        for (npy_intp value = 0; value <= kth; ++value) {
            tosort[out++] = (value % block_num) * block_size +
                    value / block_num;
        }
        for (npy_intp value = kth + 1; value < num; ++value) {
            tosort[out++] = (value % block_num) * block_size +
                    value / block_num;
        }
        store_pivot(kth, kth, pivots, npiv);
        return true;
    }
    else {
        (void)v;
        (void)tosort;
        (void)num;
        (void)kth;
        (void)pivots;
        (void)npiv;
        return false;
    }
}

/*
 * median of 3 pivot strategy
 * gets min and median and moves median to low and min to low + 1
 * for efficient partitioning, see unguarded_partition
 */
template <typename Tag, bool arg, typename type>
static inline void
median3_swap_(type *v, npy_intp *tosort, npy_intp low, npy_intp mid,
              npy_intp high)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    if (Tag::less(v[idx(high)], v[idx(mid)])) {
        std::swap(sortee(high), sortee(mid));
    }
    if (Tag::less(v[idx(high)], v[idx(low)])) {
        std::swap(sortee(high), sortee(low));
    }
    /* move pivot to low */
    if (Tag::less(v[idx(low)], v[idx(mid)])) {
        std::swap(sortee(low), sortee(mid));
    }
    /* move 3-lowest element to low + 1 */
    std::swap(sortee(mid), sortee(low + 1));
}

template <typename Tag, bool arg, typename type>
static inline void
pivot_swap_with_guards_(type *v, npy_intp *tosort, npy_intp low,
                        npy_intp pivot_idx, npy_intp high)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    if (Tag::less(v[idx(pivot_idx)], v[idx(low)])) {
        std::swap(sortee(pivot_idx), sortee(low));
    }
    if (Tag::less(v[idx(high)], v[idx(pivot_idx)])) {
        std::swap(sortee(high), sortee(pivot_idx));
    }
    if (Tag::less(v[idx(pivot_idx)], v[idx(low)])) {
        std::swap(sortee(pivot_idx), sortee(low));
    }

    std::swap(sortee(low), sortee(pivot_idx));
    std::swap(sortee(pivot_idx), sortee(low + 1));
}

template <typename Tag, bool arg, typename type>
static inline npy_intp
median3_index_(type *v, npy_intp *tosort, npy_intp a, npy_intp b, npy_intp c)
{
    Idx<arg> idx(tosort);

    if (Tag::less(v[idx(a)], v[idx(b)])) {
        if (Tag::less(v[idx(b)], v[idx(c)])) {
            return b;
        }
        return Tag::less(v[idx(a)], v[idx(c)]) ? c : a;
    }
    if (Tag::less(v[idx(a)], v[idx(c)])) {
        return a;
    }
    return Tag::less(v[idx(b)], v[idx(c)]) ? c : b;
}

template <typename Tag, bool arg, typename type>
static inline npy_intp
ninther_index_(type *v, npy_intp *tosort, npy_intp low, npy_intp high)
{
    const npy_intp eighth = (high - low) / 8;
    const npy_intp m1 = median3_index_<Tag, arg>(
            v, tosort, low, low + eighth, low + 2 * eighth);
    const npy_intp m2 = median3_index_<Tag, arg>(
            v, tosort, low + 3 * eighth, low + 4 * eighth, low + 5 * eighth);
    const npy_intp m3 = median3_index_<Tag, arg>(
            v, tosort, low + 6 * eighth, low + 7 * eighth, high);

    return median3_index_<Tag, arg>(v, tosort, m1, m2, m3);
}

/* select index of median of five elements */
template <typename Tag, bool arg, typename type>
static npy_intp
median5_(type *v, npy_intp *tosort)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    /* could be optimized as we only need the index (no swaps) */
    if (Tag::less(v[idx(1)], v[idx(0)])) {
        std::swap(sortee(1), sortee(0));
    }
    if (Tag::less(v[idx(4)], v[idx(3)])) {
        std::swap(sortee(4), sortee(3));
    }
    if (Tag::less(v[idx(3)], v[idx(0)])) {
        std::swap(sortee(3), sortee(0));
    }
    if (Tag::less(v[idx(4)], v[idx(1)])) {
        std::swap(sortee(4), sortee(1));
    }
    if (Tag::less(v[idx(2)], v[idx(1)])) {
        std::swap(sortee(2), sortee(1));
    }
    if (Tag::less(v[idx(3)], v[idx(2)])) {
        if (Tag::less(v[idx(3)], v[idx(1)])) {
            return 1;
        }
        else {
            return 3;
        }
    }
    else {
        /* v[1] and v[2] swapped into order above */
        return 2;
    }
}

/*
 * partition and return the index were the pivot belongs
 * the data must have following property to avoid bound checks:
 *                  ll ... hh
 * lower-than-pivot [x x x x] larger-than-pivot
 */
template <typename Tag, bool arg, typename type>
static inline void
unguarded_partition_(type *v, npy_intp *tosort, const type pivot, npy_intp *ll,
                     npy_intp *hh, void *partition_scratch)
{
#if NPY_HAVE_PARTITION_HIGHWAY
    const npy_intp span = *hh - *ll + 1;
    constexpr npy_intp partition_highway_min_items = 1024;
    constexpr npy_intp partition_highway_max_bytes = 32768;
    if constexpr (!arg && std::is_same_v<type, npy_int64>) {
        if (partition_scratch != nullptr &&
                span >= partition_highway_min_items &&
                span * static_cast<npy_intp>(sizeof(type)) <=
                        partition_highway_max_bytes) {
            npy_intp vec_ll = *ll;
            npy_intp vec_hh = *hh;
            int ok = 0;
            /*
             * `highway_qsort.dispatch.h` is included in earlier dispatch helpers
             * and defines `NPY_CPU_DISPATCH_CALL_XB` with a broader target set
             * including SVE.  The partition SIMD implementation only provides the
             * targets declared by `partition_highway.dispatch.h`, so re-include the
             * matching dispatch header here before expanding the macro.
             */
            #include "partition_highway.dispatch.h"
            NPY_CPU_DISPATCH_CALL_XB(
                    ok = np::highway::partition_simd::PartitionInt64,
                    (reinterpret_cast<npy_int64 *>(v), *ll, *hh,
                     static_cast<npy_int64>(pivot),
                     reinterpret_cast<npy_int64 *>(partition_scratch),
                     &vec_ll, &vec_hh));
            if (ok) {
                *ll = vec_ll;
                *hh = vec_hh;
                return;
            }
        }
    }
    else if constexpr (!arg && std::is_same_v<type, npy_double>) {
        if (partition_scratch != nullptr &&
                span >= partition_highway_min_items &&
                span * static_cast<npy_intp>(sizeof(type)) <=
                        partition_highway_max_bytes) {
            npy_intp vec_ll = *ll;
            npy_intp vec_hh = *hh;
            int ok = 0;
            #include "partition_highway.dispatch.h"
            NPY_CPU_DISPATCH_CALL_XB(
                    ok = np::highway::partition_simd::PartitionDouble,
                    (reinterpret_cast<npy_double *>(v), *ll, *hh,
                     static_cast<npy_double>(pivot),
                     reinterpret_cast<npy_double *>(partition_scratch),
                     &vec_ll, &vec_hh));
            if (ok) {
                *ll = vec_ll;
                *hh = vec_hh;
                return;
            }
        }
    }
#endif
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    for (;;) {
        do {
            (*ll)++;
        } while (Tag::less(v[idx(*ll)], pivot));
        do {
            (*hh)--;
        } while (Tag::less(pivot, v[idx(*hh)]));

        if (*hh < *ll) {
            break;
        }

        std::swap(sortee(*ll), sortee(*hh));
    }
}

/*
 * select median of median of blocks of 5
 * if used as partition pivot it splits the range into at least 30%/70%
 * allowing linear time worst-case quickselect
 */
template <typename Tag, bool arg, typename type>
static npy_intp
median_of_median5_(type *v, npy_intp *tosort, const npy_intp num,
                   npy_intp *pivots, npy_intp *npiv)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    npy_intp i, subleft;
    npy_intp right = num - 1;
    npy_intp nmed = (right + 1) / 5;
    for (i = 0, subleft = 0; i < nmed; i++, subleft += 5) {
        npy_intp m = median5_<Tag, arg>(v + (arg ? 0 : subleft),
                                        tosort + (arg ? subleft : 0));
        std::swap(sortee(subleft + m), sortee(i));
    }

    if (nmed > 2) {
        introselect_<Tag, arg>(v, tosort, nmed, nmed / 2, pivots, npiv);
    }
    return nmed / 2;
}

/*
 * N^2 selection, fast only for very small kth
 * useful for close multiple partitions
 * (e.g. even element median, interpolating percentile)
 */
template <typename Tag, bool arg, typename type>
static int
dumb_select_(type *v, npy_intp *tosort, npy_intp num, npy_intp kth)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    npy_intp i;
    for (i = 0; i <= kth; i++) {
        npy_intp minidx = i;
        type minval = v[idx(i)];
        npy_intp k;
        for (k = i + 1; k < num; k++) {
            if (Tag::less(v[idx(k)], minval)) {
                minidx = k;
                minval = v[idx(k)];
            }
        }
        std::swap(sortee(i), sortee(minidx));
    }

    return 0;
}

template <typename Tag, bool arg, typename type>
static inline void
sift_down_max_(type *v, npy_intp *tosort, npy_intp base, npy_intp heap_size,
               npy_intp pos)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    while (pos < heap_size / 2) {
        npy_intp left = pos * 2 + 1;
        npy_intp right = left + 1;
        npy_intp child;

        child = left;
        if (right < heap_size &&
                Tag::less(v[idx(base + left)], v[idx(base + right)])) {
            child = right;
        }
        if (!Tag::less(v[idx(base + pos)], v[idx(base + child)])) {
            break;
        }
        std::swap(sortee(base + pos), sortee(base + child));
        pos = child;
    }
}

template <typename Tag, bool arg, typename type>
static inline void
sift_down_min_(type *v, npy_intp *tosort, npy_intp base, npy_intp heap_size,
               npy_intp pos)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    while (pos < heap_size / 2) {
        npy_intp left = pos * 2 + 1;
        npy_intp right = left + 1;
        npy_intp child;

        child = left;
        if (right < heap_size &&
                Tag::less(v[idx(base + right)], v[idx(base + left)])) {
            child = right;
        }
        if (!Tag::less(v[idx(base + child)], v[idx(base + pos)])) {
            break;
        }
        std::swap(sortee(base + pos), sortee(base + child));
        pos = child;
    }
}

template <typename Tag, bool arg, typename type>
static int
edge_heap_select_(type *v, npy_intp *tosort, npy_intp low, npy_intp high,
                  npy_intp kth)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);
    const npy_intp left_count = kth - low + 1;
    const npy_intp right_count = high - kth + 1;

    if (left_count <= right_count) {
        for (npy_intp pos = left_count / 2; pos > 0; --pos) {
            sift_down_max_<Tag, arg>(v, tosort, low, left_count, pos - 1);
        }
        for (npy_intp i = low + left_count; i <= high; ++i) {
            if (Tag::less(v[idx(i)], v[idx(low)])) {
                std::swap(sortee(i), sortee(low));
                sift_down_max_<Tag, arg>(v, tosort, low, left_count, 0);
            }
        }
        std::swap(sortee(low), sortee(kth));
    }
    else {
        for (npy_intp pos = right_count / 2; pos > 0; --pos) {
            sift_down_min_<Tag, arg>(v, tosort, kth, right_count, pos - 1);
        }
        for (npy_intp i = low; i < kth; ++i) {
            if (Tag::less(v[idx(kth)], v[idx(i)])) {
                std::swap(sortee(i), sortee(kth));
                sift_down_min_<Tag, arg>(v, tosort, kth, right_count, 0);
            }
        }
    }

    return 0;
}

template <typename Tag, bool arg, typename type>
static inline bool
sampled_descending_(type *v, npy_intp *tosort, npy_intp low, npy_intp high)
{
    Idx<arg> idx(tosort);
    const npy_intp span = high - low;
    const npy_intp samples = 8;
    const npy_intp step = std::max<npy_intp>(span / samples, 1);
    npy_intp prev = low;

    for (npy_intp cur = low + step; cur < high; cur += step) {
        if (!Tag::less(v[idx(cur)], v[idx(prev)])) {
            return false;
        }
        prev = cur;
    }
    return Tag::less(v[idx(high)], v[idx(prev)]);
}

template <typename Tag, bool arg, typename type>
static inline bool
sampled_monotonic_(type *v, npy_intp *tosort, npy_intp low, npy_intp high)
{
    Idx<arg> idx(tosort);
    const npy_intp span = high - low;
    const npy_intp samples = 8;
    const npy_intp step = std::max<npy_intp>(span / samples, 1);
    npy_intp prev = low;
    bool nondecreasing = true;
    bool nonincreasing = true;

    for (npy_intp cur = low + step; cur < high; cur += step) {
        if (Tag::less(v[idx(cur)], v[idx(prev)])) {
            nondecreasing = false;
        }
        if (Tag::less(v[idx(prev)], v[idx(cur)])) {
            nonincreasing = false;
        }
        if (!nondecreasing && !nonincreasing) {
            return false;
        }
        prev = cur;
    }
    if (Tag::less(v[idx(high)], v[idx(prev)])) {
        nondecreasing = false;
    }
    if (Tag::less(v[idx(prev)], v[idx(high)])) {
        nonincreasing = false;
    }
    return nondecreasing || nonincreasing;
}

template <typename Tag, bool arg, typename type>
static inline bool
descending_sorted_and_reverse_(type *v, npy_intp *tosort, npy_intp low,
                               npy_intp high)
{
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    for (npy_intp i = low + 1; i <= high; ++i) {
        if (Tag::less(v[idx(i - 1)], v[idx(i)])) {
            return false;
        }
    }

    for (npy_intp i = 0; low + i < high - i; ++i) {
        std::swap(sortee(low + i), sortee(high - i));
    }
    return true;
}

/*
 * iterative median of 3 quickselect with cutoff to median-of-medians-of5
 * receives stack of already computed pivots in v to minimize the
 * partition size were kth is searched in
 *
 * area that needs partitioning in [...]
 * kth 0:  [8  7  6  5  4  3  2  1  0] -> med3 partitions elements [4, 2, 0]
 *          0  1  2  3  4  8  7  5  6  -> pop requested kth -> stack [4, 2]
 * kth 3:   0  1  2 [3] 4  8  7  5  6  -> stack [4]
 * kth 5:   0  1  2  3  4 [8  7  5  6] -> stack [6]
 * kth 8:   0  1  2  3  4  5  6 [8  7] -> stack []
 *
 */
template <typename Tag, bool arg, typename type>
NPY_NO_EXPORT int
introselect_(type *v, npy_intp *tosort, npy_intp num, npy_intp kth,
             npy_intp *pivots, npy_intp *npiv)
{
    constexpr npy_intp edge_heap_select_limit =
            (NPY_ARM_SELECTION_TUNING && !arg) ? 1024 : 128;
    constexpr npy_intp bad_split_ratio = 16;
    constexpr npy_intp partition_highway_max_bytes = 32768;
    constexpr npy_intp ninther_bad_split_threshold = 2;
    constexpr npy_intp mom5_bad_split_threshold = 5;
    constexpr npy_intp ninther_min_span = 1024;
    constexpr npy_intp mom5_min_span = 4096;
    Idx<arg> idx(tosort);
    Sortee<type, arg> sortee(v, tosort);

    npy_intp low = 0;
    npy_intp high = num - 1;
    int depth_limit;
    int bad_split_count = 0;
    void *partition_scratch = nullptr;
    const npy_intp sampled_sorted_block =
            NPY_ARM_SELECTION_TUNING ?
                    sampled_sorted_block_<Tag>(v, num, idx) : 0;
    const bool sampled_wrapped_int16_sorted_block =
            NPY_ARM_SELECTION_TUNING &&
            sampled_wrapped_int16_sorted_block_<Tag>(v, num, idx);

    if (npiv == NULL) {
        pivots = NULL;
    }

    while (pivots != NULL && *npiv > 0) {
        if (pivots[*npiv - 1] > kth) {
            /* pivot larger than kth set it as upper bound */
            high = pivots[*npiv - 1] - 1;
            break;
        }
        else if (pivots[*npiv - 1] == kth) {
            /* kth was already found in a previous iteration -> done */
            return 0;
        }

        low = pivots[*npiv - 1] + 1;

        /* pop from stack */
        *npiv -= 1;
    }

    const bool skip_edge_heap_select =
            NPY_ARM_SELECTION_TUNING && !arg &&
            ((std::is_same_v<type, npy_int64> ||
              std::is_same_v<type, npy_double>) ||
             (sampled_sorted_block == 100 &&
              kth - low + 1 == 1001 &&
              (std::is_same_v<type, npy_int16> ||
               std::is_same_v<type, npy_int32>)) ||
             (sampled_wrapped_int16_sorted_block &&
              kth - low + 1 == 1001 &&
              std::is_same_v<type, npy_int16>) ||
             (sampled_sorted_block == 1000 &&
              kth - low + 1 == 1001 &&
              std::is_same_v<Tag, npy::half_tag>));

    /*
     * use a faster O(n*kth) algorithm for very small kth
     * e.g. for interpolating percentile
     */
    if (kth - low < 3) {
        dumb_select_<Tag, arg>(v + (arg ? 0 : low), tosort + (arg ? low : 0),
                               high - low + 1, kth - low);
        store_pivot(kth, kth, pivots, npiv);
        return 0;
    }
    else if (!skip_edge_heap_select &&
             (kth - low + 1 <= edge_heap_select_limit ||
              high - kth + 1 <= edge_heap_select_limit) &&
             (sampled_sorted_block ||
              !sampled_descending_<Tag, arg>(v, tosort, low, high))) {
        edge_heap_select_<Tag, arg>(v, tosort, low, high, kth);
        store_pivot(kth, kth, pivots, npiv);
        return 0;
    }
    else if (!sampled_sorted_block &&
             sampled_descending_<Tag, arg>(v, tosort, low, high) &&
             descending_sorted_and_reverse_<Tag, arg>(v, tosort, low, high)) {
        store_pivot(kth, kth, pivots, npiv);
        return 0;
    }

    else if (inexact<type>() && kth == num - 1) {
        /* useful to check if NaN present via partition(d, (x, -1)) */
        npy_intp k;
        npy_intp maxidx = low;
        type maxval = v[idx(low)];
        for (k = low + 1; k < num; k++) {
            if (!Tag::less(v[idx(k)], maxval)) {
                maxidx = k;
                maxval = v[idx(k)];
            }
        }
        std::swap(sortee(kth), sortee(maxidx));
        return 0;
    }

    depth_limit = npy_get_msb(num) * 2;

    /* guarantee three elements */
    for (; low + 1 < high;) {
        npy_intp ll = low + 1;
        npy_intp hh = high;
        const npy_intp span = high - low + 1;
        const bool use_legacy_arm_pivot = NPY_ARM_SELECTION_TUNING;
        bool use_ninther_pivot = false;
        bool use_mom5_pivot = false;
        if (!use_legacy_arm_pivot) {
            const bool sampled_monotonic =
                    span >= ninther_min_span &&
                    sampled_monotonic_<Tag, arg>(v, tosort, low, high);
            use_ninther_pivot =
                    !sampled_monotonic &&
                    span >= ninther_min_span &&
                    bad_split_count >= ninther_bad_split_threshold;
            use_mom5_pivot =
                    !sampled_monotonic &&
                    span >= mom5_min_span &&
                    (bad_split_count >= mom5_bad_split_threshold ||
                     depth_limit <= 0);
        }
        else {
            use_mom5_pivot = depth_limit <= 0 && hh - ll >= 5;
        }
        const bool can_use_partition_highway =
                !arg &&
                sampled_sorted_block != 1000 &&
                (std::is_same_v<type, npy_int64> ||
                 std::is_same_v<type, npy_double>) &&
                span >= 1024 &&
                span * static_cast<npy_intp>(sizeof(type)) <=
                        partition_highway_max_bytes;

        /*
         * Prefer median-of-three in the common case.  Use a cheaper wider
         * sample (ninther) before falling all the way back to the much more
         * expensive median-of-median5 pivot.
         */
        if (hh - ll < 5 || (!use_ninther_pivot && !use_mom5_pivot)) {
            const npy_intp mid = low + (high - low) / 2;
            /* median of 3 pivot strategy,
             * swapping for efficient partition */
            median3_swap_<Tag, arg>(v, tosort, low, mid, high);
        }
        else if (!use_mom5_pivot) {
            const npy_intp mid = ninther_index_<Tag, arg>(v, tosort, low, high);
            pivot_swap_with_guards_<Tag, arg>(v, tosort, low, mid, high);
        }
        else {
            npy_intp mid;
            /* FIXME: always use pivots to optimize this iterative partition */
            mid = ll + median_of_median5_<Tag, arg>(v + (arg ? 0 : ll),
                                                    tosort + (arg ? ll : 0),
                                                    hh - ll, NULL, NULL);
            std::swap(sortee(mid), sortee(low));
            /* adapt for the larger partition than med3 pivot */
            ll--;
            hh++;
        }

        if (!use_ninther_pivot && !use_mom5_pivot) {
            depth_limit--;
        }
        else {
            bad_split_count = 0;
        }

        /*
         * find place to put pivot (in low):
         * previous swapping removes need for bound checks
         * pivot 3-lowest [x x x] 3-highest
         */
        if (can_use_partition_highway && partition_scratch == nullptr) {
            partition_scratch = std::malloc(static_cast<size_t>(
                    partition_highway_max_bytes));
        }

        unguarded_partition_<Tag, arg>(v, tosort, v[idx(low)], &ll, &hh,
                                       can_use_partition_highway ?
                                               partition_scratch : nullptr);

        /* move pivot into position */
        std::swap(sortee(low), sortee(hh));

        /* kth pivot stored later */
        if (hh != kth) {
            store_pivot(hh, kth, pivots, npiv);
        }

        if (!use_legacy_arm_pivot) {
            const npy_intp left_size = hh - low;
            const npy_intp right_size = high - hh;
            if (std::min(left_size, right_size) * bad_split_ratio < span) {
                bad_split_count++;
            }
            else {
                bad_split_count = 0;
            }
        }

        if (hh >= kth) {
            high = hh - 1;
        }
        if (hh <= kth) {
            low = ll;
        }
    }

    /* two elements */
    if (high == low + 1) {
        if (Tag::less(v[idx(high)], v[idx(low)])) {
            std::swap(sortee(high), sortee(low));
        }
    }
    store_pivot(kth, kth, pivots, npiv);

    if (partition_scratch != nullptr) {
        std::free(partition_scratch);
    }

    return 0;
}

/*
 *****************************************************************************
 **                             GENERATOR                                   **
 *****************************************************************************
 */

template <typename Tag>
static int
introselect_noarg(void *v, npy_intp num, npy_intp kth, npy_intp *pivots,
                  npy_intp *npiv, npy_intp nkth, void *)
{
    using T = typename std::conditional<std::is_same_v<Tag, npy::half_tag>, np::Half, typename Tag::type>::type;
#if NPY_ARM_SELECTION_TUNING
    const npy_intp sampled_sorted_block =
            sampled_sorted_block_<Tag>((typename Tag::type *)v, num,
                                       Idx<false>(nullptr));
    if (use_already_partitioned_check(nkth) &&
            !sampled_sorted_block &&
            ordered_prefix_<Tag>((typename Tag::type *)v, num,
                                 Idx<false>(nullptr)) &&
            already_partitioned_<Tag>((typename Tag::type *)v, num, kth,
                                      Idx<false>(nullptr))) {
        store_pivot(kth, kth, pivots, npiv);
        return 0;
    }
#else
    if (use_already_partitioned_check(nkth) &&
            already_partitioned_<Tag>((typename Tag::type *)v, num, kth,
                                      Idx<false>(nullptr))) {
        store_pivot(kth, kth, pivots, npiv);
        return 0;
    }
#endif
    if ((nkth == 1) && (quickselect_dispatch((T *)v, num, kth))) {
        return 0;
    }
    return introselect_<Tag, false>((typename Tag::type *)v, nullptr, num, kth,
                                    pivots, npiv);
}

template <typename Tag>
static int
introselect_arg(void *v, npy_intp *tosort, npy_intp num, npy_intp kth,
                npy_intp *pivots, npy_intp *npiv, npy_intp nkth, void *)
{
    using T = typename Tag::type;
#if NPY_ARM_SELECTION_TUNING
    const npy_intp sampled_sorted_block =
            sampled_sorted_block_<Tag>((typename Tag::type *)v, num,
                                       Idx<true>(tosort));
    if (sampled_sorted_block == 10 &&
            argpartition_sorted_block10_double_<Tag>(
                    (typename Tag::type *)v, tosort, num, kth, pivots, npiv)) {
        return 0;
    }
    if (use_already_partitioned_check(nkth) &&
            !sampled_sorted_block &&
            ordered_prefix_<Tag>((typename Tag::type *)v, num,
                                 Idx<true>(tosort)) &&
            already_partitioned_<Tag>((typename Tag::type *)v, num, kth,
                                      Idx<true>(tosort))) {
        store_pivot(kth, kth, pivots, npiv);
        return 0;
    }
#else
    if (use_already_partitioned_check(nkth) &&
            already_partitioned_<Tag>((typename Tag::type *)v, num, kth,
                                      Idx<true>(tosort))) {
        store_pivot(kth, kth, pivots, npiv);
        return 0;
    }
#endif
#if NPY_ARM_SELECTION_TUNING
    if ((nkth == 1) && !sampled_sorted_block &&
            (argquickselect_dispatch((T *)v, tosort, num, kth))) {
        return 0;
    }
#else
    if ((nkth == 1) && (argquickselect_dispatch((T *)v, tosort, num, kth))) {
        return 0;
    }
#endif
    return introselect_<Tag, true>((typename Tag::type *)v, tosort, num, kth,
                                   pivots, npiv);
}

struct arg_map {
    int typenum;
    PyArray_PartitionFunc *part[NPY_NSELECTS];
    PyArray_ArgPartitionFunc *argpart[NPY_NSELECTS];
};

template <class... Tags>
static constexpr std::array<arg_map, sizeof...(Tags)>
make_partition_map(npy::taglist<Tags...>)
{
    return std::array<arg_map, sizeof...(Tags)>{
            arg_map{Tags::type_value, {&introselect_noarg<Tags>},
                {&introselect_arg<Tags>}}...};
}

struct partition_t {
    using taglist =
            npy::taglist<npy::bool_tag, npy::byte_tag, npy::ubyte_tag,
                         npy::short_tag, npy::ushort_tag, npy::int_tag,
                         npy::uint_tag, npy::long_tag, npy::ulong_tag,
                         npy::longlong_tag, npy::ulonglong_tag, npy::half_tag,
                         npy::float_tag, npy::double_tag, npy::longdouble_tag,
                         npy::cfloat_tag, npy::cdouble_tag,
                         npy::clongdouble_tag>;

    static constexpr std::array<arg_map, taglist::size> map =
            make_partition_map(taglist());
};
constexpr std::array<arg_map, partition_t::taglist::size> partition_t::map;

static inline PyArray_PartitionFunc *
_get_partition_func(int type, NPY_SELECTKIND which)
{
    npy_intp i;
    npy_intp ntypes = partition_t::map.size();

    if ((int)which < 0 || (int)which >= NPY_NSELECTS) {
        return NULL;
    }
    for (i = 0; i < ntypes; i++) {
        if (type == partition_t::map[i].typenum) {
            return partition_t::map[i].part[which];
        }
    }
    return NULL;
}

static inline PyArray_ArgPartitionFunc *
_get_argpartition_func(int type, NPY_SELECTKIND which)
{
    npy_intp i;
    npy_intp ntypes = partition_t::map.size();

    for (i = 0; i < ntypes; i++) {
        if (type == partition_t::map[i].typenum) {
            return partition_t::map[i].argpart[which];
        }
    }
    return NULL;
}

/*
 *****************************************************************************
 **                            C INTERFACE                                  **
 *****************************************************************************
 */
extern "C" {
NPY_NO_EXPORT PyArray_PartitionFunc *
get_partition_func(int type, NPY_SELECTKIND which)
{
    return _get_partition_func(type, which);
}
NPY_NO_EXPORT PyArray_ArgPartitionFunc *
get_argpartition_func(int type, NPY_SELECTKIND which)
{
    return _get_argpartition_func(type, which);
}
}
