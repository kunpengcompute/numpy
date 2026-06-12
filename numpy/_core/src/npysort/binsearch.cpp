/* -*- c -*- */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"

#include "npy_binsearch.h"
#include "npy_sort.h"
#include "numpy_tag.h"

#include <array>
#include <functional>  // for std::less and std::less_equal
#include <vector>

// Enumerators for the variant of binsearch
enum arg_t
{
    noarg,
    arg
};
enum side_t
{
    left,
    right
};

// Mapping from enumerators to comparators
template <class Tag, side_t side>
struct side_to_cmp;

template <class Tag>
struct side_to_cmp<Tag, left> {
    static constexpr auto value = Tag::less;
};

template <class Tag>
struct side_to_cmp<Tag, right> {
    static constexpr auto value = Tag::less_equal;
};

template <side_t side>
struct side_to_generic_cmp;

template <>
struct side_to_generic_cmp<left> {
    using type = std::less<int>;
};

template <>
struct side_to_generic_cmp<right> {
    using type = std::less_equal<int>;
};

#if defined(__aarch64__)

/*
 * Interleaved multi-query binary search for contiguous arrays on aarch64.
 * Processes BATCH queries simultaneously to overlap DRAM latency via
 * the CPU's out-of-order execution. Prefetches next-level midpoint addresses.
 */
template <class Tag, side_t side, int BATCH>
static void
binsearch_interleaved(const char *arr, const char *key, char *ret,
                      npy_intp arr_len, npy_intp key_len)
{
    using T = typename Tag::type;
    auto cmp = side_to_cmp<Tag, side>::value;
    const T *arr_t = (const T *)arr;
    const T *key_t = (const T *)key;
    npy_intp *ret_t = (npy_intp *)ret;

    npy_intp ki = 0;
    npy_intp batch_end = key_len - (key_len % BATCH);

    for (; ki < batch_end; ki += BATCH) {
        T key_vals[BATCH];
        npy_intp min_idx[BATCH], max_idx[BATCH];
        int active[BATCH];
        int any_active = BATCH;

        for (int b = 0; b < BATCH; b++) {
            key_vals[b] = key_t[ki + b];
            min_idx[b] = 0;
            max_idx[b] = arr_len;
            active[b] = 1;
        }

        /* Interleaved binary search: advance all queries by one step
         * each round, overlapping their midpoint memory accesses */
        while (any_active > 0) {
            for (int b = 0; b < BATCH; b++) {
                if (!active[b]) continue;

                npy_intp mid = min_idx[b] + ((max_idx[b] - min_idx[b]) >> 1);

                /* Prefetch both possible next-step midpoints */
                npy_intp next_left = min_idx[b] + ((mid - min_idx[b]) >> 1);
                npy_intp next_right = mid + 1 + ((max_idx[b] - mid - 1) >> 1);
                __builtin_prefetch(arr_t + next_left, 0, 1);
                __builtin_prefetch(arr_t + next_right, 0, 1);

                T mid_val = arr_t[mid];

                if (cmp(mid_val, key_vals[b])) {
                    min_idx[b] = mid + 1;
                } else {
                    max_idx[b] = mid;
                }

                if (min_idx[b] >= max_idx[b]) {
                    ret_t[ki + b] = min_idx[b];
                    active[b] = 0;
                    any_active--;
                }
            }
        }
    }

    /* Remaining queries: scalar search */
    for (; ki < key_len; ki++) {
        T key_val = key_t[ki];
        npy_intp min_i = 0, max_i = arr_len;
        while (min_i < max_i) {
            npy_intp mid = min_i + ((max_i - min_i) >> 1);
            T mid_val = arr_t[mid];
            if (cmp(mid_val, key_val)) {
                min_i = mid + 1;
            } else {
                max_i = mid;
            }
        }
        ret_t[ki] = min_i;
    }
}

/*
 * Two-level indexed interleaved binary search for large arrays on aarch64.
 * Builds a compact block index (first element of each 4K-entry block = ~32KB)
 * that fits in L1 cache. Coarse search in the index stays in L1 (~12 steps @ 3ns),
 * fine search within identified blocks stays in L2/L3 (~12 steps @ 10-20ns).
 * Interleaved fine search overlaps DRAM latency across BATCH queries.
 */
template <class Tag, side_t side, int BATCH>
static void
binsearch_indexed_interleaved(const char *arr, const char *key, char *ret,
                              npy_intp arr_len, npy_intp key_len)
{
    using T = typename Tag::type;
    auto cmp = side_to_cmp<Tag, side>::value;
    const T *arr_t = (const T *)arr;
    const T *key_t = (const T *)key;
    npy_intp *ret_t = (npy_intp *)ret;

    const npy_intp BLOCK_SIZE = 4096;
    const npy_intp n_blocks = (arr_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Build block index (first element of each block) — fits in L1 */
    std::vector<T> block_index_vec(n_blocks);
    T *block_index = block_index_vec.data();
    for (npy_intp i = 0; i < n_blocks; i++) {
        block_index[i] = arr_t[i * BLOCK_SIZE];
    }

    npy_intp ki = 0;
    npy_intp batch_end = key_len - (key_len % BATCH);

    for (; ki < batch_end; ki += BATCH) {
        T key_vals[BATCH];
        npy_intp block_idx[BATCH];

        /* Load key values and do coarse binary search in block_index */
        for (int b = 0; b < BATCH; b++) {
            key_vals[b] = key_t[ki + b];

            npy_intp lo = 0, hi = n_blocks;
            while (lo < hi) {
                npy_intp mid = lo + ((hi - lo) >> 1);
                if (cmp(block_index[mid], key_vals[b])) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            block_idx[b] = (lo > 0) ? lo - 1 : 0;
        }

        /* Initialize fine search bounds */
        npy_intp fine_min[BATCH], fine_max[BATCH];
        int active[BATCH];
        int any_active = BATCH;

        for (int b = 0; b < BATCH; b++) {
            fine_min[b] = block_idx[b] * BLOCK_SIZE;
            fine_max[b] = fine_min[b] + BLOCK_SIZE;
            if (fine_max[b] > arr_len) fine_max[b] = arr_len;
            active[b] = 1;

            /* Prefetch first midpoint of fine search */
            npy_intp first_mid = fine_min[b] + ((fine_max[b] - fine_min[b]) >> 1);
            __builtin_prefetch(arr_t + first_mid, 0, 1);
        }

        /* Interleaved fine binary search within blocks */
        while (any_active > 0) {
            for (int b = 0; b < BATCH; b++) {
                if (!active[b]) continue;

                npy_intp mid = fine_min[b] + ((fine_max[b] - fine_min[b]) >> 1);

                /* Prefetch both possible next-step midpoints */
                npy_intp next_left = fine_min[b] + ((mid - fine_min[b]) >> 1);
                npy_intp next_right = mid + 1 + ((fine_max[b] - mid - 1) >> 1);
                __builtin_prefetch(arr_t + next_left, 0, 1);
                __builtin_prefetch(arr_t + next_right, 0, 1);

                T mid_val = arr_t[mid];

                if (cmp(mid_val, key_vals[b])) {
                    fine_min[b] = mid + 1;
                } else {
                    fine_max[b] = mid;
                }

                if (fine_min[b] >= fine_max[b]) {
                    ret_t[ki + b] = fine_min[b];
                    active[b] = 0;
                    any_active--;
                }
            }
        }
    }

    /* Remaining queries: scalar search with two-level decomposition */
    for (; ki < key_len; ki++) {
        T key_val = key_t[ki];

        /* Coarse search */
        npy_intp lo = 0, hi = n_blocks;
        while (lo < hi) {
            npy_intp mid = lo + ((hi - lo) >> 1);
            if (cmp(block_index[mid], key_val)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        npy_intp blk = (lo > 0) ? lo - 1 : 0;

        /* Fine search */
        npy_intp fine_lo = blk * BLOCK_SIZE;
        npy_intp fine_hi = fine_lo + BLOCK_SIZE;
        if (fine_hi > arr_len) fine_hi = arr_len;

        __builtin_prefetch(arr_t + fine_lo + ((fine_hi - fine_lo) >> 1), 0, 1);

        while (fine_lo < fine_hi) {
            npy_intp mid = fine_lo + ((fine_hi - fine_lo) >> 1);
            if (cmp(arr_t[mid], key_val)) {
                fine_lo = mid + 1;
            } else {
                fine_hi = mid;
            }
        }
        ret_t[ki] = fine_lo;
    }
}

#endif /* __aarch64__ */

/*
 *****************************************************************************
 **                            NUMERIC SEARCHES                             **
 *****************************************************************************
 */
template <class Tag, side_t side>
static void
binsearch(const char *arr, const char *key, char *ret, npy_intp arr_len,
          npy_intp key_len, npy_intp arr_str, npy_intp key_str,
          npy_intp ret_str, PyArrayObject *)
{
    using T = typename Tag::type;

#if defined(__aarch64__)
    /* Fast path: ARM-optimized search for contiguous arrays */
    if (arr_str == sizeof(T) && key_str == sizeof(T) &&
        ret_str == sizeof(npy_intp)) {
        if (arr_len > 32768 && key_len >= 4) {
            /* Two-level indexed interleaved search (cache optimization) */
            binsearch_indexed_interleaved<Tag, side, 4>(
                arr, key, ret, arr_len, key_len);
            return;
        }
        else if (arr_len > 0 && key_len >= 4) {
            /* Direct interleaved search (small arrays) */
            binsearch_interleaved<Tag, side, 4>(
                arr, key, ret, arr_len, key_len);
            return;
        }
    }
#endif

    auto cmp = side_to_cmp<Tag, side>::value;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    T last_key_val;

    if (key_len == 0) {
        return;
    }
    last_key_val = *(const T *)key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        const T key_val = *(const T *)key;
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        if (cmp(last_key_val, key_val)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key_val = key_val;

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const T mid_val = *(const T *)(arr + mid_idx * arr_str);
            if (cmp(mid_val, key_val)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(npy_intp *)ret = min_idx;
    }
}

template <class Tag, side_t side>
static int
argbinsearch(const char *arr, const char *key, const char *sort, char *ret,
             npy_intp arr_len, npy_intp key_len, npy_intp arr_str,
             npy_intp key_str, npy_intp sort_str, npy_intp ret_str,
             PyArrayObject *)
{
    using T = typename Tag::type;
    auto cmp = side_to_cmp<Tag, side>::value;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    T last_key_val;

    if (key_len == 0) {
        return 0;
    }
    last_key_val = *(const T *)key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        const T key_val = *(const T *)key;
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        if (cmp(last_key_val, key_val)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key_val = key_val;

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const npy_intp sort_idx = *(npy_intp *)(sort + mid_idx * sort_str);
            T mid_val;

            if (sort_idx < 0 || sort_idx >= arr_len) {
                return -1;
            }

            mid_val = *(const T *)(arr + sort_idx * arr_str);

            if (cmp(mid_val, key_val)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(npy_intp *)ret = min_idx;
    }
    return 0;
}

/*
 *****************************************************************************
 **                             GENERIC SEARCH                              **
 *****************************************************************************
 */

template <side_t side>
static void
npy_binsearch(const char *arr, const char *key, char *ret, npy_intp arr_len,
              npy_intp key_len, npy_intp arr_str, npy_intp key_str,
              npy_intp ret_str, PyArrayObject *cmp)
{
    using Cmp = typename side_to_generic_cmp<side>::type;
    PyArray_CompareFunc *compare = PyDataType_GetArrFuncs(PyArray_DESCR(cmp))->compare;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    const char *last_key = key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        if (Cmp{}(compare(last_key, key, cmp), 0)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key = key;

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const char *arr_ptr = arr + mid_idx * arr_str;

            if (Cmp{}(compare(arr_ptr, key, cmp), 0)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(npy_intp *)ret = min_idx;
    }
}

template <side_t side>
static int
npy_argbinsearch(const char *arr, const char *key, const char *sort, char *ret,
                 npy_intp arr_len, npy_intp key_len, npy_intp arr_str,
                 npy_intp key_str, npy_intp sort_str, npy_intp ret_str,
                 PyArrayObject *cmp)
{
    using Cmp = typename side_to_generic_cmp<side>::type;
    PyArray_CompareFunc *compare = PyDataType_GetArrFuncs(PyArray_DESCR(cmp))->compare;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    const char *last_key = key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        if (Cmp{}(compare(last_key, key, cmp), 0)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key = key;

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const npy_intp sort_idx = *(npy_intp *)(sort + mid_idx * sort_str);
            const char *arr_ptr;

            if (sort_idx < 0 || sort_idx >= arr_len) {
                return -1;
            }

            arr_ptr = arr + sort_idx * arr_str;

            if (Cmp{}(compare(arr_ptr, key, cmp), 0)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(npy_intp *)ret = min_idx;
    }
    return 0;
}

/*
 *****************************************************************************
 **                             GENERATOR                                   **
 *****************************************************************************
 */

template <arg_t arg>
struct binsearch_base;

template <>
struct binsearch_base<arg> {
    using function_type = PyArray_ArgBinSearchFunc *;
    struct value_type {
        int typenum;
        function_type binsearch[NPY_NSEARCHSIDES];
    };
    template <class... Tags>
    static constexpr std::array<value_type, sizeof...(Tags)>
    make_binsearch_map(npy::taglist<Tags...>)
    {
        return std::array<value_type, sizeof...(Tags)>{
                value_type{Tags::type_value,
                           {(function_type)&argbinsearch<Tags, left>,
                            (function_type)argbinsearch<Tags, right>}}...};
    }
    static constexpr std::array<function_type, 2> npy_map = {
            (function_type)&npy_argbinsearch<left>,
            (function_type)&npy_argbinsearch<right>};
};
constexpr std::array<binsearch_base<arg>::function_type, 2>
        binsearch_base<arg>::npy_map;

template <>
struct binsearch_base<noarg> {
    using function_type = PyArray_BinSearchFunc *;
    struct value_type {
        int typenum;
        function_type binsearch[NPY_NSEARCHSIDES];
    };
    template <class... Tags>
    static constexpr std::array<value_type, sizeof...(Tags)>
    make_binsearch_map(npy::taglist<Tags...>)
    {
        return std::array<value_type, sizeof...(Tags)>{
                value_type{Tags::type_value,
                           {(function_type)&binsearch<Tags, left>,
                            (function_type)binsearch<Tags, right>}}...};
    }
    static constexpr std::array<function_type, 2> npy_map = {
            (function_type)&npy_binsearch<left>,
            (function_type)&npy_binsearch<right>};
};
constexpr std::array<binsearch_base<noarg>::function_type, 2>
        binsearch_base<noarg>::npy_map;

// Handle generation of all binsearch variants
template <arg_t arg>
struct binsearch_t : binsearch_base<arg> {
    using binsearch_base<arg>::make_binsearch_map;
    using value_type = typename binsearch_base<arg>::value_type;

    using taglist = npy::taglist<
            /* If adding new types, make sure to keep them ordered by type num
             */
            npy::bool_tag, npy::byte_tag, npy::ubyte_tag, npy::short_tag,
            npy::ushort_tag, npy::int_tag, npy::uint_tag, npy::long_tag,
            npy::ulong_tag, npy::longlong_tag, npy::ulonglong_tag,
            npy::float_tag, npy::double_tag, npy::longdouble_tag, 
            npy::cfloat_tag, npy::cdouble_tag, npy::clongdouble_tag, 
            npy::datetime_tag, npy::timedelta_tag, npy::half_tag>;

    static constexpr std::array<value_type, taglist::size> map =
            make_binsearch_map(taglist());
};

template <arg_t arg>
constexpr std::array<typename binsearch_t<arg>::value_type,
                     binsearch_t<arg>::taglist::size>
        binsearch_t<arg>::map;

template <arg_t arg>
static inline typename binsearch_t<arg>::function_type
_get_binsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    using binsearch = binsearch_t<arg>;
    npy_intp nfuncs = binsearch::map.size();
    npy_intp min_idx = 0;
    npy_intp max_idx = nfuncs;
    int type = dtype->type_num;

    if ((int)side >= (int)NPY_NSEARCHSIDES) {
        return NULL;
    }

    /*
     * It seems only fair that a binary search function be searched for
     * using a binary search...
     */
    while (min_idx < max_idx) {
        npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);

        if (binsearch::map[mid_idx].typenum < type) {
            min_idx = mid_idx + 1;
        }
        else {
            max_idx = mid_idx;
        }
    }

    if (min_idx < nfuncs && binsearch::map[min_idx].typenum == type) {
        return binsearch::map[min_idx].binsearch[side];
    }

    if (PyDataType_GetArrFuncs(dtype)->compare) {
        return binsearch::npy_map[side];
    }

    return NULL;
}

/*
 *****************************************************************************
 **                            C INTERFACE                                  **
 *****************************************************************************
 */
extern "C" {
NPY_NO_EXPORT PyArray_BinSearchFunc *
get_binsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    return _get_binsearch_func<noarg>(dtype, side);
}

NPY_NO_EXPORT PyArray_ArgBinSearchFunc *
get_argbinsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    return _get_binsearch_func<arg>(dtype, side);
}
}
