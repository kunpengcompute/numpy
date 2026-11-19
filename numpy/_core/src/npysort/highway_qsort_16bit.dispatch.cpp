#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#include "highway_qsort.hpp"
#include "highway_qsort_determinism.hpp"
#include "quicksort.hpp"

#include <algorithm>

#include "common.hpp"

namespace np::highway::qsort_simd {
template <typename T>
void NPY_CPU_DISPATCH_CURFX(QSort)(T *arr, npy_intp size)
{
#if VQSORT_ENABLED
    using THwy = std::conditional_t<std::is_same_v<T, Half>, hwy::float16_t, T>;
    hwy::HWY_NAMESPACE::VQSortStatic(reinterpret_cast<THwy*>(arr), size, hwy::SortAscending());
#else
    sort::Quick(arr, size);
#endif
}
#if !HWY_HAVE_FLOAT16
template <>
void NPY_CPU_DISPATCH_CURFX(QSort)<Half>(Half *arr, npy_intp size)
{
    sort::Quick(arr, size);
}
#endif // !HWY_HAVE_FLOAT16

template void NPY_CPU_DISPATCH_CURFX(QSort)<int16_t>(int16_t*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(QSort)<uint16_t>(uint16_t*, npy_intp);
#if HWY_HAVE_FLOAT16
template void NPY_CPU_DISPATCH_CURFX(QSort)<Half>(Half*, npy_intp);
#endif

namespace {

constexpr npy_intp kSmallArgSort16 = 64;

template <typename T>
inline bool ArgLess16(T *arr, npy_intp a, npy_intp b)
{
    if constexpr (std::is_same_v<T, Half>) {
        if (arr[a].IsNaN()) return false;
        if (arr[b].IsNaN()) return true;
    }
    return arr[a] < arr[b];
}

template <typename T>
void ArgInsertionSort16(T *arr, npy_intp *arg, npy_intp size)
{
    for (npy_intp i = 0; i < size; ++i) {
        arg[i] = i;
    }
    for (npy_intp *pi = arg + 1; pi < arg + size; ++pi) {
        npy_intp vi = *pi;
        npy_intp *pj = pi;
        while (pj > arg && ArgLess16<T>(arr, vi, *(pj - 1))) {
            *pj = *(pj - 1);
            --pj;
        }
        *pj = vi;
    }
}

template <typename T>
inline uint32_t ToSortableKey16(T val)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        return static_cast<uint32_t>(val);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        return static_cast<uint32_t>(static_cast<uint16_t>(val) ^ 0x8000U);
    }
    else if constexpr (std::is_same_v<T, Half>) {
        /* IEEE-754 order-preserving key at 16-bit scale: negatives map to
         * [0, 0x7FFF] via the 16-bit complement, positives to
         * [0x8000, 0xFFFF] via the sign bit, so that -x < 0 <= +x.
         * Note the two zeros are adjacent but NOT equal: -0.0 -> 0x7FFF and
         * +0.0 -> 0x8000, so the key order puts -0.0 first; both orders are
         * valid sorted output. Complementing the zero-extended 32/64-bit
         * word or OR-ing 0x80000000 would put negatives *after* positives. */
        uint16_t u = val.Bits();
        if (val.IsNaN()) {
            return 0xFFFFFFFFU;
        }
        if (u & 0x8000U) {
            return static_cast<uint32_t>(static_cast<uint16_t>(~u));
        } else {
            return static_cast<uint32_t>(u) | 0x8000U;
        }
    }
}

template <typename T>
bool CheckSortedReversed16(T *arr, npy_intp size, npy_intp *arg)
{
    if (size <= 1) {
        if (size == 1) arg[0] = 0;
        return true;
    }

    bool is_sorted = true;
    for (npy_intp i = 1; i < size; ++i) {
        if (ArgLess16<T>(arr, i, i - 1)) {
            is_sorted = false;
            break;
        }
    }
    if (is_sorted) {
        for (npy_intp i = 0; i < size; ++i) {
            arg[i] = i;
        }
        return true;
    }

    bool is_reversed = true;
    for (npy_intp i = 1; i < size; ++i) {
        if (!ArgLess16<T>(arr, i, i - 1)) {
            is_reversed = false;
            break;
        }
    }
    if (is_reversed) {
        for (npy_intp i = 0; i < size; ++i) {
            arg[i] = size - 1 - i;
        }
        return true;
    }

    return false;
}

} // anonymous namespace

template <typename T>
void NPY_CPU_DISPATCH_CURFX(ArgQSort)(T *arr, npy_intp* arg, npy_intp size)
{
    if (CheckSortedReversed16<T>(arr, size, arg)) {
        return;
    }

    if (size < kSmallArgSort16) {
        ArgInsertionSort16(arr, arg, size);
        return;
    }

#if VQSORT_ENABLED
    // Deterministic pivot stream and sampling grid; see
    // highway_qsort_determinism.hpp. Must happen before VQSortStatic.
    ResetVQSortRNG();
    if (static_cast<uint64_t>(size) > 0x100000000ULL) {
        // K32V32's 32-bit value field cannot hold indices past 2^32-1.
        for (npy_intp i = 0; i < size; ++i) {
            arg[i] = i;
        }
        std::sort(arg, arg + size, [arr](npy_intp a, npy_intp b) {
            return ArgLess16<T>(arr, a, b);
        });
        return;
    }
    // 64-byte aligned buffer: DrawSamples() aligns its sampling grid to the
    // runtime address of the keys, which must not depend on malloc.
    AlignedPairs<hwy::K32V32> pairs(static_cast<size_t>(size));
    hwy::K32V32 *kv = pairs.data();
    for (npy_intp i = 0; i < size; ++i) {
        kv[i].key = ToSortableKey16(arr[i]);
        kv[i].value = static_cast<uint32_t>(i);
    }
    hwy::HWY_NAMESPACE::VQSortStatic(kv, static_cast<size_t>(size), hwy::SortAscending());
    for (npy_intp i = 0; i < size; ++i) {
        arg[i] = static_cast<npy_intp>(kv[i].value);
    }
#else
    for (npy_intp i = 0; i < size; ++i) {
        arg[i] = i;
    }
    std::sort(arg, arg + size, [arr](npy_intp a, npy_intp b) {
        return ArgLess16<T>(arr, a, b);
    });
#endif
}

template <typename T>
void NPY_CPU_DISPATCH_CURFX(ArgQSelect)(T *arr, npy_intp* arg, npy_intp num, npy_intp kth)
{
    if (num <= 1) {
        if (num == 1) arg[0] = 0;
        return;
    }

    if (num < kSmallArgSort16) {
        for (npy_intp i = 0; i < num; ++i) {
            arg[i] = i;
        }
        for (npy_intp i = 0; i <= kth; ++i) {
            npy_intp min_idx = i;
            for (npy_intp j = i + 1; j < num; ++j) {
                if (ArgLess16<T>(arr, arg[j], arg[min_idx])) {
                    min_idx = j;
                }
            }
            if (min_idx != i) {
                std::swap(arg[i], arg[min_idx]);
            }
        }
        return;
    }

#if VQSORT_ENABLED
    // Deterministic pivot stream and sampling grid; see
    // highway_qsort_determinism.hpp. Must happen before VQSelectStatic.
    ResetVQSortRNG();
    if (static_cast<uint64_t>(num) > 0x100000000ULL) {
        // K32V32's 32-bit value field cannot hold indices past 2^32-1.
        for (npy_intp i = 0; i < num; ++i) {
            arg[i] = i;
        }
        std::nth_element(arg, arg + kth, arg + num, [arr](npy_intp a, npy_intp b) {
            return ArgLess16<T>(arr, a, b);
        });
        return;
    }
    AlignedPairs<hwy::K32V32> pairs(static_cast<size_t>(num));
    hwy::K32V32 *kv = pairs.data();
    for (npy_intp i = 0; i < num; ++i) {
        kv[i].key = ToSortableKey16(arr[i]);
        kv[i].value = static_cast<uint32_t>(i);
    }
    hwy::HWY_NAMESPACE::VQSelectStatic(kv, static_cast<size_t>(num),
                                       static_cast<size_t>(kth), hwy::SortAscending());
    for (npy_intp i = 0; i < num; ++i) {
        arg[i] = static_cast<npy_intp>(kv[i].value);
    }
#else
    for (npy_intp i = 0; i < num; ++i) {
        arg[i] = i;
    }
    std::nth_element(arg, arg + kth, arg + num, [arr](npy_intp a, npy_intp b) {
        return ArgLess16<T>(arr, a, b);
    });
#endif
}

template void NPY_CPU_DISPATCH_CURFX(ArgQSort)<int16_t>(int16_t*, npy_intp*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(ArgQSelect)<int16_t>(int16_t*, npy_intp*, npy_intp, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(ArgQSort)<uint16_t>(uint16_t*, npy_intp*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(ArgQSelect)<uint16_t>(uint16_t*, npy_intp*, npy_intp, npy_intp);
#if HWY_HAVE_FLOAT16
template void NPY_CPU_DISPATCH_CURFX(ArgQSort)<Half>(Half*, npy_intp*, npy_intp);
template void NPY_CPU_DISPATCH_CURFX(ArgQSelect)<Half>(Half*, npy_intp*, npy_intp, npy_intp);
#endif

} // np::highway::qsort_simd