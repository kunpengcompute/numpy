/*
 * C++ template-based strided copy kernels.
 *
 * This file replaces the copy-related kernels previously generated from
 * lowlevel_strided_loops.c.src with native C++ templates, enabling better
 * compiler auto-vectorization and cleaner code.
 *
 * Copyright (c) 2010 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * See LICENSE.txt for the license.
 */

#include <cstring>
#include <type_traits>
#include <cstddef>

#include "lowlevel_strided_loops.h"

#if defined(__aarch64__)

/*
 * NPY_USE_UNALIGNED_ACCESS controls whether the strided copy kernels use
 * typed pointer access (e.g. *(uint64_t*)ptr) which the compiler may
 * optimize assuming the data is aligned to its type size.
 *
 * On x86, the compiler may auto-vectorize using SSE instructions (e.g.
 * movaps) that require 16-byte alignment. If the data is not properly
 * aligned, this causes a fault. So on x86 this flag must be 0 to use
 * the safe unaligned (memcpy-based) path when alignment is not guaranteed.
 *
 * On aarch64, NEON load/store instructions (ld1/st1) and scalar ldr/str
 * natively support unaligned memory access without faulting. The compiler
 * can safely use typed pointer access regardless of alignment, so we
 * enable this flag to always use the optimized aligned kernels.
 */
#define NPY_USE_UNALIGNED_ACCESS 1

namespace {
/*
 * Helper: unsigned integer type of given size.
 */
template<int N>
struct uint_of_size;

struct copyuint128 { npy_uint64 a; npy_uint64 b; };

template<> struct uint_of_size<1>  { using type = npy_uint8; };
template<> struct uint_of_size<2>  { using type = npy_uint16; };
template<> struct uint_of_size<4>  { using type = npy_uint32; };
template<> struct uint_of_size<8>  { using type = npy_uint64; };
template<> struct uint_of_size<16> { using type = copyuint128; };

template<int N>
using uint_of_size_t = typename uint_of_size<N>::type;

/*
 * ============================================================================
 * Broadcast copy kernels (src_stride = 0)
 *
 * Each combination of (aligned, dst_contig) gets its own function to avoid
 * if constexpr branches inside the loop, which prevents loop unrolling and
 * auto-vectorization on GCC 12.
 * ============================================================================
 */

/* Aligned + Contiguous destination */
template<int N>
NPY_GCC_OPT_3 NPY_GCC_UNROLL_LOOPS int
broadcast_copy_ac_impl(const char *src, char *dst, npy_intp count) {
    using T = uint_of_size_t<N>;
    T temp = *(const T*)src;
    while (count > 0) {
        *(T*)dst = temp;
        dst += N;
        --count;
    }
    return 0;
}

/*
 * Specialization for N=8 (64-bit broadcast, aligned + contiguous dst).
 *
 * The generic template's while-loop with pointer increment (dst += N)
 * causes the compiler to generate suboptimal auto-vectorized code on
 * aarch64: each store uses a separate lsl+add offset calculation with
 * individual str q0, resulting in ~80% of cycles spent in prologue
 * (ldr + cmp) before the first store.
 *
 * Explicit 8-way unrolling with array indexing (arr[0]..arr[7]) gives
 * the compiler a clear view of the contiguous memory pattern, enabling
 * efficient stp q0,q0 paired stores (32 bytes per instruction) with a
 * single add to advance the pointer. This shifts the bottleneck from
 * prologue overhead to actual store throughput.
 */
template<>
int broadcast_copy_ac_impl<8>(const char *src, char *dst, npy_intp count) {
    using T = uint_of_size_t<8>;
    T temp = *(const T*)src;
    T* arr = (T*)dst;

    constexpr npy_intp UNROLL = 8;
    for (; count >= UNROLL; count -= UNROLL, arr += UNROLL) {
        arr[0] = temp;
        arr[1] = temp;
        arr[2] = temp;
        arr[3] = temp;
        arr[4] = temp;
        arr[5] = temp;
        arr[6] = temp;
        arr[7] = temp;
    }

    for (; count > 0; --count, ++arr) {
        *arr = temp;
    }

    return 0;
}

/* Aligned + Strided destination */
template<int N>
NPY_GCC_OPT_3 NPY_GCC_UNROLL_LOOPS int
broadcast_copy_as_impl(const char *src, char *dst, npy_intp count, npy_intp dst_stride) {
    using T = uint_of_size_t<N>;
    T temp = *(const T*)src;
    while (count > 0) {
        *(T*)dst = temp;
        dst += dst_stride;
        --count;
    }
    return 0;
}

#if !NPY_USE_UNALIGNED_ACCESS
/* Unaligned + Contiguous destination */
template<int N>
NPY_GCC_UNROLL_LOOPS int
broadcast_copy_uc_impl(const char *src, char *dst, npy_intp count) {
    char temp[N];
    std::memcpy(temp, src, N);
    while (count > 0) {
        std::memcpy(dst, temp, N);
        dst += N;
        --count;
    }
    return 0;
}

/* Unaligned + Strided destination */
template<int N>
NPY_GCC_UNROLL_LOOPS int
broadcast_copy_us_impl(const char *src, char *dst, npy_intp count, npy_intp dst_stride) {
    char temp[N];
    std::memcpy(temp, src, N);
    while (count > 0) {
        std::memcpy(dst, temp, N);
        dst += dst_stride;
        --count;
    }
    return 0;
}
#endif

/*
 * Broadcast copy dispatch function.
 * Uses constexpr to select the correct implementation at compile time.
 */
template<int N, bool aligned, bool dst_contig>
int broadcast_copy_dispatch(
    PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
    const npy_intp *dimensions, const npy_intp *strides,
    NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp count = dimensions[0];
    const char *src = args[0];
    char *dst = args[1];

    if (count == 0) {
        return 0;
    }

#if aligned && !NPY_USE_UNALIGNED_ACCESS
    if (count > 0) {
        assert(npy_is_aligned(dst, NPY_ALIGNOF_UINT(uint_of_size_t<N>)));
        assert(npy_is_aligned(src, NPY_ALIGNOF_UINT(uint_of_size_t<N>)));
    }
#endif

    /* Special case: memset for 1-byte contiguous broadcast */
    if constexpr (N == 1 && dst_contig) {
        std::memset(dst, *src, count);
        return 0;
    }

    /* Dispatch to specialized implementation */
    if constexpr (aligned && dst_contig) {
        return broadcast_copy_ac_impl<N>(src, dst, count);
    } else if constexpr (aligned && !dst_contig) {
        return broadcast_copy_as_impl<N>(src, dst, count, strides[1]);
#if !NPY_USE_UNALIGNED_ACCESS
    } else if constexpr (!aligned && dst_contig) {
        return broadcast_copy_uc_impl<N>(src, dst, count);
    } else {
        return broadcast_copy_us_impl<N>(src, dst, count, strides[1]);
#endif
    }
}

/*
 * ============================================================================
 * Strided copy kernels
 *
 * Each combination of (aligned, src_contig, dst_contig) gets its own function.
 * ============================================================================
 */

/* Aligned + Contiguous src + Contiguous dst */
template<int N>
NPY_GCC_UNROLL_LOOPS int
strided_copy_acc_impl(const char *src, char *dst, npy_intp count) {
    using T = uint_of_size_t<N>;
    while (count > 0) {
        *(T*)dst = *(const T*)src;
        dst += N;
        src += N;
        --count;
    }
    return 0;
}

/* Aligned + Contiguous src + Strided dst */
template<int N>
NPY_GCC_UNROLL_LOOPS int
strided_copy_acs_impl(const char *src, char *dst, npy_intp count, npy_intp dst_stride) {
    using T = uint_of_size_t<N>;
    while (count > 0) {
        *(T*)dst = *(const T*)src;
        dst += dst_stride;
        src += N;
        --count;
    }
    return 0;
}

/* Aligned + Strided src + Contiguous dst */
template<int N>
NPY_GCC_UNROLL_LOOPS int
strided_copy_asc_impl(const char *src, char *dst, npy_intp count, npy_intp src_stride) {
    using T = uint_of_size_t<N>;
    while (count > 0) {
        *(T*)dst = *(const T*)src;
        dst += N;
        src += src_stride;
        --count;
    }
    return 0;
}

/* Aligned + Strided src + Strided dst */
template<int N>
NPY_GCC_UNROLL_LOOPS int
strided_copy_ass_impl(const char *src, char *dst, npy_intp count, npy_intp src_stride, npy_intp dst_stride) {
    using T = uint_of_size_t<N>;
    while (count > 0) {
        *(T*)dst = *(const T*)src;
        dst += dst_stride;
        src += src_stride;
        --count;
    }
    return 0;
}

#if !NPY_USE_UNALIGNED_ACCESS
/* Unaligned + Contiguous src + Contiguous dst */
template<int N>
NPY_GCC_UNROLL_LOOPS int
strided_copy_ucc_impl(const char *src, char *dst, npy_intp count) {
    while (count > 0) {
        std::memcpy(dst, src, N);
        dst += N;
        src += N;
        --count;
    }
    return 0;
}

/* Unaligned + Contiguous src + Strided dst */
template<int N>
NPY_GCC_UNROLL_LOOPS int
strided_copy_ucs_impl(const char *src, char *dst, npy_intp count, npy_intp dst_stride) {
    while (count > 0) {
        std::memcpy(dst, src, N);
        dst += dst_stride;
        src += N;
        --count;
    }
    return 0;
}

/* Unaligned + Strided src + Contiguous dst */
template<int N>
NPY_GCC_UNROLL_LOOPS int
strided_copy_usc_impl(const char *src, char *dst, npy_intp count, npy_intp src_stride) {
    while (count > 0) {
        std::memcpy(dst, src, N);
        dst += N;
        src += src_stride;
        --count;
    }
    return 0;
}

/* Unaligned + Strided src + Strided dst */
template<int N>
NPY_GCC_UNROLL_LOOPS int
strided_copy_uss_impl(const char *src, char *dst, npy_intp count, npy_intp src_stride, npy_intp dst_stride) {
    while (count > 0) {
        std::memcpy(dst, src, N);
        dst += dst_stride;
        src += src_stride;
        --count;
    }
    return 0;
}
#endif

/*
 * Strided copy dispatch function.
 */
template<int N, bool aligned, bool src_contig, bool dst_contig>
int strided_copy_dispatch(
    PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
    const npy_intp *dimensions, const npy_intp *strides,
    NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp count = dimensions[0];
    const char *src = args[0];
    char *dst = args[1];

#if aligned && !NPY_USE_UNALIGNED_ACCESS
    if (count > 0) {
        assert(npy_is_aligned(dst, NPY_ALIGNOF_UINT(uint_of_size_t<N>)));
        assert(npy_is_aligned(src, NPY_ALIGNOF_UINT(uint_of_size_t<N>)));
    }
#endif

    /* Dispatch to specialized implementation */
    if constexpr (aligned && src_contig && !dst_contig) {
        return strided_copy_acs_impl<N>(src, dst, count, strides[1]);
    } else if constexpr (aligned && !src_contig && dst_contig) {
        return strided_copy_asc_impl<N>(src, dst, count, strides[0]);
    } else if constexpr (aligned && !src_contig && !dst_contig) {
        return strided_copy_ass_impl<N>(src, dst, count, strides[0], strides[1]);
#if !NPY_USE_UNALIGNED_ACCESS
    } else if constexpr (!aligned && src_contig && dst_contig) {
        return strided_copy_ucc_impl<N>(src, dst, count);
    } else if constexpr (!aligned && src_contig && !dst_contig) {
        return strided_copy_ucs_impl<N>(src, dst, count, strides[1]);
    } else if constexpr (!aligned && !src_contig && dst_contig) {
        return strided_copy_usc_impl<N>(src, dst, count, strides[0]);
    } else {
        return strided_copy_uss_impl<N>(src, dst, count, strides[0], strides[1]);
#endif
    }
}

}

/*
 * Helper macros to define function pointers for specific combinations
 */
#define DEF_KERNEL(N, aligned, src_c, dst_c) \
    &strided_copy_dispatch<N, aligned, src_c, dst_c>

#define DEF_BROADCAST(N, aligned, dst_c) \
    &broadcast_copy_dispatch<N, aligned, dst_c>

extern "C" {

static int
_strided_to_strided(
        PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(data))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_stride = strides[0], dst_stride = strides[1];
    npy_intp src_itemsize = context->descriptors[0]->elsize;

    while (N > 0) {
        memmove(dst, src, src_itemsize);
        dst += dst_stride;
        src += src_stride;
        --N;
    }
    return 0;
}

static int
_contig_to_contig(
        PyArrayMethod_Context *context, char *const *args,
        const npy_intp *dimensions, const npy_intp *NPY_UNUSED(strides),
        NpyAuxData *NPY_UNUSED(data))
{
    npy_intp N = dimensions[0];
    char *src = args[0], *dst = args[1];
    npy_intp src_itemsize = context->descriptors[0]->elsize;

    memmove(dst, src, src_itemsize*N);
    return 0;
}

/*
 * PyArray_GetStridedCopyFn - returns function pointer for copying strided memory.
 */
NPY_NO_EXPORT PyArrayMethod_StridedLoop *
PyArray_GetStridedCopyFn(int aligned, npy_intp src_stride,
                         npy_intp dst_stride, npy_intp itemsize)
{
#if !NPY_USE_UNALIGNED_ACCESS
    if (aligned) {
#endif

        /* contiguous dst */
        if (itemsize != 0 && dst_stride == itemsize) {
            /* constant src (broadcast) */
            if (src_stride == 0) {
                switch (itemsize) {
                    case 1:  return DEF_BROADCAST(1, true, true);
                    case 2:  return DEF_BROADCAST(2, true, true);
                    case 4:  return DEF_BROADCAST(4, true, true);
                    case 8:  return DEF_BROADCAST(8, true, true);
                    case 16: return DEF_BROADCAST(16, true, true);
                }
            }
            /* contiguous src */
            else if (src_stride == itemsize) {
                return &_contig_to_contig;
            }
            /* general src (strided to contiguous) */
            else {
                switch (itemsize) {
                    case 1:  return DEF_KERNEL(1, true, false, true);
                    case 2:  return DEF_KERNEL(2, true, false, true);
                    case 4:  return DEF_KERNEL(4, true, false, true);
                    case 8:  return DEF_KERNEL(8, true, false, true);
                    case 16: return DEF_KERNEL(16, true, false, true);
                }
            }
            return &_strided_to_strided;
        }
        /* general dst */
        else {
            /* constant src (broadcast) */
            if (src_stride == 0) {
                switch (itemsize) {
                    case 1:  return DEF_BROADCAST(1, true, false);
                    case 2:  return DEF_BROADCAST(2, true, false);
                    case 4:  return DEF_BROADCAST(4, true, false);
                    case 8:  return DEF_BROADCAST(8, true, false);
                    case 16: return DEF_BROADCAST(16, true, false);
                }
            }
            /* contiguous src */
            else if (src_stride == itemsize) {
                switch (itemsize) {
                    case 1:  return DEF_KERNEL(1, true, true, false);
                    case 2:  return DEF_KERNEL(2, true, true, false);
                    case 4:  return DEF_KERNEL(4, true, true, false);
                    case 8:  return DEF_KERNEL(8, true, true, false);
                    case 16: return DEF_KERNEL(16, true, true, false);
                }
                return &_strided_to_strided;
            }
            /* general src and dst (strided to strided) */
            else {
                switch (itemsize) {
                    case 1:  return DEF_KERNEL(1, true, false, false);
                    case 2:  return DEF_KERNEL(2, true, false, false);
                    case 4:  return DEF_KERNEL(4, true, false, false);
                    case 8:  return DEF_KERNEL(8, true, false, false);
                    case 16: return DEF_KERNEL(16, true, false, false);
                }
            }
        }

#if !NPY_USE_UNALIGNED_ACCESS
    }
    else {
        /* Unaligned path */
        if (itemsize != 0) {
            if (dst_stride == itemsize) {
                /* contiguous dst */
                if (src_stride == itemsize) {
                    /* contiguous src, dst */
                    return &_contig_to_contig;
                }
                else {
                    /* general src */
                    switch (itemsize) {
                        case 1:  return DEF_KERNEL(1, false, false, true);
                        case 2:  return DEF_KERNEL(2, false, false, true);
                        case 4:  return DEF_KERNEL(4, false, false, true);
                        case 8:  return DEF_KERNEL(8, false, false, true);
                        case 16: return DEF_KERNEL(16, false, false, true);
                    }
                }
                return &_strided_to_strided;
            }
            else if (src_stride == itemsize) {
                /* contiguous src, general dst */
                switch (itemsize) {
                    case 1:  return DEF_KERNEL(1, false, true, false);
                    case 2:  return DEF_KERNEL(2, false, true, false);
                    case 4:  return DEF_KERNEL(4, false, true, false);
                    case 8:  return DEF_KERNEL(8, false, true, false);
                    case 16: return DEF_KERNEL(16, false, true, false);
                }
                return &_strided_to_strided;
            }
        }
        else {
            /* general src, dst */
            switch (itemsize) {
                case 1:  return DEF_KERNEL(1, false, false, false);
                case 2:  return DEF_KERNEL(2, false, false, false);
                case 4:  return DEF_KERNEL(4, false, false, false);
                case 8:  return DEF_KERNEL(8, false, false, false);
                case 16: return DEF_KERNEL(16, false, false, false);
            }
        }
    }
#endif

    return &_strided_to_strided;
}

}

#endif
