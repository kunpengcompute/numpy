#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"
#include "npysort_common.h"

#include "../common/numpy_tag.h"
#include <cstdlib>
#include <type_traits>

/*
 *****************************************************************************
 **                            INTEGER SORTS                                **
 *****************************************************************************
 */

// Reference: https://github.com/eloj/radix-sorting#-key-derivation
template <class T, class UT>
static inline UT
KEY_OF(UT x)
{
    // Floating-point is currently disabled.
    // Floating-point tests succeed for double and float on macOS but not on
    // Windows/Linux. Basic sorting tests succeed but others relying on sort
    // fail. Possibly related to floating-point normalisation or multiple NaN
    // reprs? Not sure.
    if constexpr (std::is_floating_point<T>::value) {
        // For floats, we invert the key if the sign bit is set, else we invert
        // the sign bit.
        return ((x) ^ (-((x) >> (sizeof(T) * 8 - 1)) |
                       ((UT)1 << (sizeof(T) * 8 - 1))));
    }
    else if constexpr (std::is_signed<T>::value) {
        // For signed ints, we flip the sign bit so the negatives are below the
        // positives.
        return ((x) ^ ((UT)1 << (sizeof(UT) * 8 - 1)));
    }
    else {
        return x;
    }
}

template <class T>
static inline npy_ubyte
nth_byte(T key, npy_intp l)
{
    return (key >> (l << 3)) & 0xFF;
}

template <class T, class UT>
static UT *
radixsort0(UT *start, UT *aux, npy_intp num)
{
    npy_intp cnt[sizeof(UT)][1 << 8] = {{0}};
    UT key0 = KEY_OF<T>(start[0]);

    for (npy_intp i = 0; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);

        for (size_t l = 0; l < sizeof(UT); l++) {
            cnt[l][nth_byte(k, l)]++;
        }
    }

    size_t ncols = 0;
    npy_ubyte cols[sizeof(UT)];
    for (size_t l = 0; l < sizeof(UT); l++) {
        if (cnt[l][nth_byte(key0, l)] != num) {
            cols[ncols++] = l;
        }
    }

    for (size_t l = 0; l < ncols; l++) {
        npy_intp a = 0;
        for (npy_intp i = 0; i < 256; i++) {
            npy_intp b = cnt[cols[l]][i];
            cnt[cols[l]][i] = a;
            a += b;
        }
    }

    for (size_t l = 0; l < ncols; l++) {
        UT *temp;
        for (npy_intp i = 0; i < num; i++) {
            UT k = KEY_OF<T>(start[i]);
            npy_intp dst = cnt[cols[l]][nth_byte(k, cols[l])]++;
            aux[dst] = start[i];
        }

        temp = aux;
        aux = start;
        start = temp;
    }

    return start;
}

/* Specialized implementation for radix sort on AArch64 platforms.
 * Provides 20-30% performance improvements on ARM 64-bit systems.
 */
#ifdef __aarch64__
template <class T, class UT>
static UT *
radixsort0_8bit(UT *start, UT *aux, npy_intp num)
{
    npy_intp cnt[256] = {0};

    const UT key0 = KEY_OF<T>(start[0]);

    npy_intp cnt_l0[256] = {0}, cnt_l1[256] = {0};
    npy_intp cnt_l2[256] = {0}, cnt_l3[256] = {0};

    npy_intp i = 0;
    for (; i + 8 <= num; i += 8) {
        UT k0 = KEY_OF<T>(start[i]);
        UT k1 = KEY_OF<T>(start[i + 1]);
        UT k2 = KEY_OF<T>(start[i + 2]);
        UT k3 = KEY_OF<T>(start[i + 3]);
        UT k4 = KEY_OF<T>(start[i + 4]);
        UT k5 = KEY_OF<T>(start[i + 5]);
        UT k6 = KEY_OF<T>(start[i + 6]);
        UT k7 = KEY_OF<T>(start[i + 7]);

        cnt_l0[nth_byte(k0, 0)]++;
        cnt_l1[nth_byte(k1, 0)]++;
        cnt_l2[nth_byte(k2, 0)]++;
        cnt_l3[nth_byte(k3, 0)]++;
        cnt_l0[nth_byte(k4, 0)]++;
        cnt_l1[nth_byte(k5, 0)]++;
        cnt_l2[nth_byte(k6, 0)]++;
        cnt_l3[nth_byte(k7, 0)]++;
    }

    for (int j = 0; j < 256; j++) {
        cnt[j] = cnt_l0[j] + cnt_l1[j] + cnt_l2[j] + cnt_l3[j];
    }

    for (; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);
        cnt[nth_byte(k, 0)]++;
    }

    if (cnt[nth_byte(key0, 0)] == num) {
        return start;
    }

    npy_intp a = 0;
    for (int j = 0; j < 256; j++) {
        npy_intp b = cnt[j];
        cnt[j] = a;
        a += b;
    }

    UT *src = start;
    UT *dst = aux;

    npy_intp j = 0;
    for (; j + 8 <= num; j += 8) {
        UT s0 = src[j], s1 = src[j + 1];
        UT s2 = src[j + 2], s3 = src[j + 3];
        UT s4 = src[j + 4], s5 = src[j + 5];
        UT s6 = src[j + 6], s7 = src[j + 7];
        UT k0 = KEY_OF<T>(s0);
        UT k1 = KEY_OF<T>(s1);
        UT k2 = KEY_OF<T>(s2);
        UT k3 = KEY_OF<T>(s3);
        UT k4 = KEY_OF<T>(s4);
        UT k5 = KEY_OF<T>(s5);
        UT k6 = KEY_OF<T>(s6);
        UT k7 = KEY_OF<T>(s7);

        dst[cnt[nth_byte(k0, 0)]++] = s0;
        dst[cnt[nth_byte(k1, 0)]++] = s1;
        dst[cnt[nth_byte(k2, 0)]++] = s2;
        dst[cnt[nth_byte(k3, 0)]++] = s3;
        dst[cnt[nth_byte(k4, 0)]++] = s4;
        dst[cnt[nth_byte(k5, 0)]++] = s5;
        dst[cnt[nth_byte(k6, 0)]++] = s6;
        dst[cnt[nth_byte(k7, 0)]++] = s7;
    }
    for (; j < num; j++) {
        UT k = KEY_OF<T>(src[j]);
        dst[cnt[nth_byte(k, 0)]++] = src[j];
    }

    return dst;
}

template <class T, class UT>
static UT *
radixsort0_16bit(UT *start, UT *aux, npy_intp num)
{
    npy_intp cnt0[256] = {0};
    npy_intp cnt1[256] = {0};

    const UT key0 = KEY_OF<T>(start[0]);

    npy_intp cnt0_l0[256] = {0}, cnt0_l1[256] = {0};
    npy_intp cnt0_l2[256] = {0}, cnt0_l3[256] = {0};
    npy_intp cnt1_l0[256] = {0}, cnt1_l1[256] = {0};
    npy_intp cnt1_l2[256] = {0}, cnt1_l3[256] = {0};

    npy_intp i = 0;
    for (; i + 8 <= num; i += 8) {
        UT k0 = KEY_OF<T>(start[i]);
        UT k1 = KEY_OF<T>(start[i + 1]);
        UT k2 = KEY_OF<T>(start[i + 2]);
        UT k3 = KEY_OF<T>(start[i + 3]);
        UT k4 = KEY_OF<T>(start[i + 4]);
        UT k5 = KEY_OF<T>(start[i + 5]);
        UT k6 = KEY_OF<T>(start[i + 6]);
        UT k7 = KEY_OF<T>(start[i + 7]);

        cnt0_l0[nth_byte(k0, 0)]++;
        cnt0_l1[nth_byte(k1, 0)]++;
        cnt0_l2[nth_byte(k2, 0)]++;
        cnt0_l3[nth_byte(k3, 0)]++;
        cnt0_l0[nth_byte(k4, 0)]++;
        cnt0_l1[nth_byte(k5, 0)]++;
        cnt0_l2[nth_byte(k6, 0)]++;
        cnt0_l3[nth_byte(k7, 0)]++;

        cnt1_l0[nth_byte(k0, 1)]++;
        cnt1_l1[nth_byte(k1, 1)]++;
        cnt1_l2[nth_byte(k2, 1)]++;
        cnt1_l3[nth_byte(k3, 1)]++;
        cnt1_l0[nth_byte(k4, 1)]++;
        cnt1_l1[nth_byte(k5, 1)]++;
        cnt1_l2[nth_byte(k6, 1)]++;
        cnt1_l3[nth_byte(k7, 1)]++;
    }

    for (int j = 0; j < 256; j++) {
        cnt0[j] = cnt0_l0[j] + cnt0_l1[j] + cnt0_l2[j] + cnt0_l3[j];
        cnt1[j] = cnt1_l0[j] + cnt1_l1[j] + cnt1_l2[j] + cnt1_l3[j];
    }

    for (; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);
        cnt0[nth_byte(k, 0)]++;
        cnt1[nth_byte(k, 1)]++;
    }

    size_t ncols = 0;
    npy_ubyte cols[2];
    if (cnt0[nth_byte(key0, 0)] != num) {
        cols[ncols++] = 0;
    }
    if (cnt1[nth_byte(key0, 1)] != num) {
        cols[ncols++] = 1;
    }

    if (ncols == 0) {
        return start;
    }

    npy_intp *cnt[2] = {cnt0, cnt1};
    for (size_t l = 0; l < ncols; l++) {
        npy_intp a = 0;
        npy_intp *c = cnt[cols[l]];
        for (int j = 0; j < 256; j++) {
            npy_intp b = c[j];
            c[j] = a;
            a += b;
        }
    }

    UT *src = start;
    UT *dst = aux;

    for (size_t l = 0; l < ncols; l++) {
        npy_intp *c = cnt[cols[l]];

        npy_intp j = 0;
        for (; j + 8 <= num; j += 8) {

            UT s0 = src[j], s1 = src[j + 1];
            UT s2 = src[j + 2], s3 = src[j + 3];
            UT s4 = src[j + 4], s5 = src[j + 5];
            UT s6 = src[j + 6], s7 = src[j + 7];
            UT k0 = KEY_OF<T>(s0);
            UT k1 = KEY_OF<T>(s1);
            UT k2 = KEY_OF<T>(s2);
            UT k3 = KEY_OF<T>(s3);
            UT k4 = KEY_OF<T>(s4);
            UT k5 = KEY_OF<T>(s5);
            UT k6 = KEY_OF<T>(s6);
            UT k7 = KEY_OF<T>(s7);

            dst[c[nth_byte(k0, cols[l])]++] = s0;
            dst[c[nth_byte(k1, cols[l])]++] = s1;
            dst[c[nth_byte(k2, cols[l])]++] = s2;
            dst[c[nth_byte(k3, cols[l])]++] = s3;
            dst[c[nth_byte(k4, cols[l])]++] = s4;
            dst[c[nth_byte(k5, cols[l])]++] = s5;
            dst[c[nth_byte(k6, cols[l])]++] = s6;
            dst[c[nth_byte(k7, cols[l])]++] = s7;
        }
        for (; j < num; j++) {
            UT k = KEY_OF<T>(src[j]);
            dst[c[nth_byte(k, cols[l])]++] = src[j];
        }

        UT *tmp = dst;
        dst = src;
        src = tmp;
    }

    return src;
}

template <>
npy_ubyte *
radixsort0<npy_byte, npy_ubyte>(npy_ubyte *start, npy_ubyte *aux, npy_intp num)
{
    return radixsort0_8bit<npy_byte, npy_ubyte>(start, aux, num);
}

template <>
npy_ubyte *
radixsort0<npy_ubyte, npy_ubyte>(npy_ubyte *start, npy_ubyte *aux, npy_intp num)
{
    return radixsort0_8bit<npy_ubyte, npy_ubyte>(start, aux, num);
}

template <>
npy_ushort *
radixsort0<npy_short, npy_ushort>(npy_ushort *start, npy_ushort *aux, npy_intp num)
{
    return radixsort0_16bit<npy_short, npy_ushort>(start, aux, num);
}

template <>
npy_ushort *
radixsort0<npy_ushort, npy_ushort>(npy_ushort *start, npy_ushort *aux, npy_intp num)
{
    return radixsort0_16bit<npy_ushort, npy_ushort>(start, aux, num);
}
#else

template <class T, class UT>
static UT *
radixsort0_8bit(UT *start, UT *aux, npy_intp num)
{
    npy_intp cnt[256] = {0};

    const UT key0 = KEY_OF<T>(start[0]);

    npy_intp cnt_l0[256] = {0}, cnt_l1[256] = {0};
    npy_intp cnt_l2[256] = {0}, cnt_l3[256] = {0};

    npy_intp i = 0;
    for (; i + 8 <= num; i += 8) {
        UT k0 = KEY_OF<T>(start[i]);
        UT k1 = KEY_OF<T>(start[i + 1]);
        UT k2 = KEY_OF<T>(start[i + 2]);
        UT k3 = KEY_OF<T>(start[i + 3]);
        UT k4 = KEY_OF<T>(start[i + 4]);
        UT k5 = KEY_OF<T>(start[i + 5]);
        UT k6 = KEY_OF<T>(start[i + 6]);
        UT k7 = KEY_OF<T>(start[i + 7]);

        cnt_l0[nth_byte(k0, 0)]++;
        cnt_l1[nth_byte(k1, 0)]++;
        cnt_l2[nth_byte(k2, 0)]++;
        cnt_l3[nth_byte(k3, 0)]++;
        cnt_l0[nth_byte(k4, 0)]++;
        cnt_l1[nth_byte(k5, 0)]++;
        cnt_l2[nth_byte(k6, 0)]++;
        cnt_l3[nth_byte(k7, 0)]++;
    }

    for (int j = 0; j < 256; j++) {
        cnt[j] = cnt_l0[j] + cnt_l1[j] + cnt_l2[j] + cnt_l3[j];
    }

    for (; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);
        cnt[nth_byte(k, 0)]++;
    }

    if (cnt[nth_byte(key0, 0)] == num) {
        return start;
    }

    npy_intp a = 0;
    for (int j = 0; j < 256; j++) {
        npy_intp b = cnt[j];
        cnt[j] = a;
        a += b;
    }

    UT *src = start;
    UT *dst = aux;

    npy_intp j = 0;
    for (; j + 8 <= num; j += 8) {
        UT s0 = src[j], s1 = src[j + 1];
        UT s2 = src[j + 2], s3 = src[j + 3];
        UT s4 = src[j + 4], s5 = src[j + 5];
        UT s6 = src[j + 6], s7 = src[j + 7];
        UT k0 = KEY_OF<T>(s0);
        UT k1 = KEY_OF<T>(s1);
        UT k2 = KEY_OF<T>(s2);
        UT k3 = KEY_OF<T>(s3);
        UT k4 = KEY_OF<T>(s4);
        UT k5 = KEY_OF<T>(s5);
        UT k6 = KEY_OF<T>(s6);
        UT k7 = KEY_OF<T>(s7);

        dst[cnt[nth_byte(k0, 0)]++] = s0;
        dst[cnt[nth_byte(k1, 0)]++] = s1;
        dst[cnt[nth_byte(k2, 0)]++] = s2;
        dst[cnt[nth_byte(k3, 0)]++] = s3;
        dst[cnt[nth_byte(k4, 0)]++] = s4;
        dst[cnt[nth_byte(k5, 0)]++] = s5;
        dst[cnt[nth_byte(k6, 0)]++] = s6;
        dst[cnt[nth_byte(k7, 0)]++] = s7;
    }
    for (; j < num; j++) {
        UT k = KEY_OF<T>(src[j]);
        dst[cnt[nth_byte(k, 0)]++] = src[j];
    }

    return dst;
}

template <class T, class UT>
static UT *
radixsort0_16bit(UT *start, UT *aux, npy_intp num)
{
    npy_intp cnt0[256] = {0};
    npy_intp cnt1[256] = {0};

    const UT key0 = KEY_OF<T>(start[0]);

    npy_intp cnt0_l0[256] = {0}, cnt0_l1[256] = {0};
    npy_intp cnt0_l2[256] = {0}, cnt0_l3[256] = {0};
    npy_intp cnt1_l0[256] = {0}, cnt1_l1[256] = {0};
    npy_intp cnt1_l2[256] = {0}, cnt1_l3[256] = {0};

    npy_intp i = 0;
    for (; i + 8 <= num; i += 8) {
        UT k0 = KEY_OF<T>(start[i]);
        UT k1 = KEY_OF<T>(start[i + 1]);
        UT k2 = KEY_OF<T>(start[i + 2]);
        UT k3 = KEY_OF<T>(start[i + 3]);
        UT k4 = KEY_OF<T>(start[i + 4]);
        UT k5 = KEY_OF<T>(start[i + 5]);
        UT k6 = KEY_OF<T>(start[i + 6]);
        UT k7 = KEY_OF<T>(start[i + 7]);

        cnt0_l0[nth_byte(k0, 0)]++;
        cnt0_l1[nth_byte(k1, 0)]++;
        cnt0_l2[nth_byte(k2, 0)]++;
        cnt0_l3[nth_byte(k3, 0)]++;
        cnt0_l0[nth_byte(k4, 0)]++;
        cnt0_l1[nth_byte(k5, 0)]++;
        cnt0_l2[nth_byte(k6, 0)]++;
        cnt0_l3[nth_byte(k7, 0)]++;

        cnt1_l0[nth_byte(k0, 1)]++;
        cnt1_l1[nth_byte(k1, 1)]++;
        cnt1_l2[nth_byte(k2, 1)]++;
        cnt1_l3[nth_byte(k3, 1)]++;
        cnt1_l0[nth_byte(k4, 1)]++;
        cnt1_l1[nth_byte(k5, 1)]++;
        cnt1_l2[nth_byte(k6, 1)]++;
        cnt1_l3[nth_byte(k7, 1)]++;
    }

    for (int j = 0; j < 256; j++) {
        cnt0[j] = cnt0_l0[j] + cnt0_l1[j] + cnt0_l2[j] + cnt0_l3[j];
        cnt1[j] = cnt1_l0[j] + cnt1_l1[j] + cnt1_l2[j] + cnt1_l3[j];
    }

    for (; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);
        cnt0[nth_byte(k, 0)]++;
        cnt1[nth_byte(k, 1)]++;
    }

    size_t ncols = 0;
    npy_ubyte cols[2];
    if (cnt0[nth_byte(key0, 0)] != num) {
        cols[ncols++] = 0;
    }
    if (cnt1[nth_byte(key0, 1)] != num) {
        cols[ncols++] = 1;
    }

    if (ncols == 0) {
        return start;
    }

    npy_intp *cnt[2] = {cnt0, cnt1};
    for (size_t l = 0; l < ncols; l++) {
        npy_intp a = 0;
        npy_intp *c = cnt[cols[l]];
        for (int j = 0; j < 256; j++) {
            npy_intp b = c[j];
            c[j] = a;
            a += b;
        }
    }

    UT *src = start;
    UT *dst = aux;

    for (size_t l = 0; l < ncols; l++) {
        npy_intp *c = cnt[cols[l]];

        npy_intp j = 0;
        for (; j + 8 <= num; j += 8) {

            UT s0 = src[j], s1 = src[j + 1];
            UT s2 = src[j + 2], s3 = src[j + 3];
            UT s4 = src[j + 4], s5 = src[j + 5];
            UT s6 = src[j + 6], s7 = src[j + 7];
            UT k0 = KEY_OF<T>(s0);
            UT k1 = KEY_OF<T>(s1);
            UT k2 = KEY_OF<T>(s2);
            UT k3 = KEY_OF<T>(s3);
            UT k4 = KEY_OF<T>(s4);
            UT k5 = KEY_OF<T>(s5);
            UT k6 = KEY_OF<T>(s6);
            UT k7 = KEY_OF<T>(s7);

            dst[c[nth_byte(k0, cols[l])]++] = s0;
            dst[c[nth_byte(k1, cols[l])]++] = s1;
            dst[c[nth_byte(k2, cols[l])]++] = s2;
            dst[c[nth_byte(k3, cols[l])]++] = s3;
            dst[c[nth_byte(k4, cols[l])]++] = s4;
            dst[c[nth_byte(k5, cols[l])]++] = s5;
            dst[c[nth_byte(k6, cols[l])]++] = s6;
            dst[c[nth_byte(k7, cols[l])]++] = s7;
        }
        for (; j < num; j++) {
            UT k = KEY_OF<T>(src[j]);
            dst[c[nth_byte(k, cols[l])]++] = src[j];
        }

        UT *tmp = dst;
        dst = src;
        src = tmp;
    }

    return src;
}
#endif

template <class T, class UT>
static int
radixsort_(UT *start, npy_intp num)
{
    npy_intp i;
    T* v_strat = (T *)start;
    T k_prev;
    npy_bool all_sorted = 1;
    if (num < 2) {
        return 0;
    }

    k_prev = v_strat[0];
    for (i = 1; i + 8 <= num; i += 8) {
        T k0 = v_strat[i];
        T k1 = v_strat[i + 1];
        T k2 = v_strat[i + 2];
        T k3 = v_strat[i + 3];
        T k4 = v_strat[i + 4];
        T k5 = v_strat[i + 5];
        T k6 = v_strat[i + 6];
        T k7 = v_strat[i + 7];
        if (k_prev > k0 || k0 > k1 || k1 > k2 || k2 > k3 || k3 > k4 ||
            k4 > k5 || k5 > k6 || k6 > k7) {
            all_sorted = 0;
            break;
        }
        k_prev = k7;
    }
    for (; i < num && all_sorted; i++) {
        T k0 = v_strat[i];
        if (k_prev > k0) {
            all_sorted = 0;
            break;
        }
        k_prev = k0;
    }
    if (all_sorted) {
        return 0;
    }

    UT *aux = (UT *)malloc(num * sizeof(UT));
    if (aux == nullptr) {
        return -NPY_ENOMEM;
    }

    UT *sorted;
    if constexpr (sizeof(UT) == 1) {
        sorted = radixsort0_8bit<T>(start, aux, num);
    }
    else if constexpr (sizeof(UT) == 2) {
        sorted = radixsort0_16bit<T>(start, aux, num);
    }
    else {
        sorted = radixsort0<T>(start, aux, num);
    }
    if (sorted != start) {
        memcpy(start, sorted, num * sizeof(UT));
    }

    free(aux);
    return 0;
}

template <class T>
static int
radixsort(void *start, npy_intp num)
{
    using UT = typename std::make_unsigned<T>::type;
    return radixsort_<T>((UT *)start, num);
}

template <class T, class UT>
static npy_intp *
aradixsort0(UT *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    npy_intp cnt[sizeof(UT)][1 << 8] = {{0}};
    UT key0 = KEY_OF<T>(start[0]);

    for (npy_intp i = 0; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);

        for (size_t l = 0; l < sizeof(UT); l++) {
            cnt[l][nth_byte(k, l)]++;
        }
    }

    size_t ncols = 0;
    npy_ubyte cols[sizeof(UT)];
    for (size_t l = 0; l < sizeof(UT); l++) {
        if (cnt[l][nth_byte(key0, l)] != num) {
            cols[ncols++] = l;
        }
    }

    for (size_t l = 0; l < ncols; l++) {
        npy_intp a = 0;
        for (npy_intp i = 0; i < 256; i++) {
            npy_intp b = cnt[cols[l]][i];
            cnt[cols[l]][i] = a;
            a += b;
        }
    }

    for (size_t l = 0; l < ncols; l++) {
        npy_intp *temp;
        for (npy_intp i = 0; i < num; i++) {
            UT k = KEY_OF<T>(start[tosort[i]]);
            npy_intp dst = cnt[cols[l]][nth_byte(k, cols[l])]++;
            aux[dst] = tosort[i];
        }

        temp = aux;
        aux = tosort;
        tosort = temp;
    }

    return tosort;
}

/* Specialized implementation for radix sort on AArch64 platforms.
 * Provides 20-30% performance improvements on ARM 64-bit systems.
 */
#ifdef __aarch64__
template <class T, class UT>
static npy_intp *
aradixsort0_8bit(UT *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    npy_intp cnt[256] = {0};

    const UT key0 = KEY_OF<T>(start[0]);

    npy_intp cnt_l0[256] = {0}, cnt_l1[256] = {0};
    npy_intp cnt_l2[256] = {0}, cnt_l3[256] = {0};

    npy_intp i = 0;
    for (; i + 8 <= num; i += 8) {
        UT k0 = KEY_OF<T>(start[i]);
        UT k1 = KEY_OF<T>(start[i + 1]);
        UT k2 = KEY_OF<T>(start[i + 2]);
        UT k3 = KEY_OF<T>(start[i + 3]);
        UT k4 = KEY_OF<T>(start[i + 4]);
        UT k5 = KEY_OF<T>(start[i + 5]);
        UT k6 = KEY_OF<T>(start[i + 6]);
        UT k7 = KEY_OF<T>(start[i + 7]);

        cnt_l0[nth_byte(k0, 0)]++;
        cnt_l1[nth_byte(k1, 0)]++;
        cnt_l2[nth_byte(k2, 0)]++;
        cnt_l3[nth_byte(k3, 0)]++;
        cnt_l0[nth_byte(k4, 0)]++;
        cnt_l1[nth_byte(k5, 0)]++;
        cnt_l2[nth_byte(k6, 0)]++;
        cnt_l3[nth_byte(k7, 0)]++;
    }

    for (int j = 0; j < 256; j++) {
        cnt[j] = cnt_l0[j] + cnt_l1[j] + cnt_l2[j] + cnt_l3[j];
    }

    for (; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);
        cnt[nth_byte(k, 0)]++;
    }

    if (cnt[nth_byte(key0, 0)] == num) {
        return tosort;
    }

    npy_intp a = 0;
    for (int j = 0; j < 256; j++) {
        npy_intp b = cnt[j];
        cnt[j] = a;
        a += b;
    }

    npy_intp *src = tosort;
    npy_intp *dst = aux;

    npy_intp j = 0;
    for (; j + 8 <= num; j += 8) {
        npy_intp s0 = src[j], s1 = src[j + 1];
        npy_intp s2 = src[j + 2], s3 = src[j + 3];
        npy_intp s4 = src[j + 4], s5 = src[j + 5];
        npy_intp s6 = src[j + 6], s7 = src[j + 7];

        UT k0 = KEY_OF<T>(start[s0]);
        UT k1 = KEY_OF<T>(start[s1]);
        UT k2 = KEY_OF<T>(start[s2]);
        UT k3 = KEY_OF<T>(start[s3]);
        UT k4 = KEY_OF<T>(start[s4]);
        UT k5 = KEY_OF<T>(start[s5]);
        UT k6 = KEY_OF<T>(start[s6]);
        UT k7 = KEY_OF<T>(start[s7]);

        dst[cnt[nth_byte(k0, 0)]++] = s0;
        dst[cnt[nth_byte(k1, 0)]++] = s1;
        dst[cnt[nth_byte(k2, 0)]++] = s2;
        dst[cnt[nth_byte(k3, 0)]++] = s3;
        dst[cnt[nth_byte(k4, 0)]++] = s4;
        dst[cnt[nth_byte(k5, 0)]++] = s5;
        dst[cnt[nth_byte(k6, 0)]++] = s6;
        dst[cnt[nth_byte(k7, 0)]++] = s7;
    }
    for (; j < num; j++) {
        UT k = KEY_OF<T>(start[src[j]]);
        dst[cnt[nth_byte(k, 0)]++] = src[j];
    }

    return dst;
}

template <class T, class UT>
static npy_intp *
aradixsort0_16bit(UT *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    npy_intp cnt0[256] = {0};
    npy_intp cnt1[256] = {0};

    const UT key0 = KEY_OF<T>(start[0]);

    npy_intp cnt0_l0[256] = {0}, cnt0_l1[256] = {0};
    npy_intp cnt0_l2[256] = {0}, cnt0_l3[256] = {0};
    npy_intp cnt1_l0[256] = {0}, cnt1_l1[256] = {0};
    npy_intp cnt1_l2[256] = {0}, cnt1_l3[256] = {0};

    npy_intp i = 0;
    for (; i + 8 <= num; i += 8) {
        UT k0 = KEY_OF<T>(start[i]);
        UT k1 = KEY_OF<T>(start[i + 1]);
        UT k2 = KEY_OF<T>(start[i + 2]);
        UT k3 = KEY_OF<T>(start[i + 3]);
        UT k4 = KEY_OF<T>(start[i + 4]);
        UT k5 = KEY_OF<T>(start[i + 5]);
        UT k6 = KEY_OF<T>(start[i + 6]);
        UT k7 = KEY_OF<T>(start[i + 7]);

        cnt0_l0[nth_byte(k0, 0)]++;
        cnt0_l1[nth_byte(k1, 0)]++;
        cnt0_l2[nth_byte(k2, 0)]++;
        cnt0_l3[nth_byte(k3, 0)]++;
        cnt0_l0[nth_byte(k4, 0)]++;
        cnt0_l1[nth_byte(k5, 0)]++;
        cnt0_l2[nth_byte(k6, 0)]++;
        cnt0_l3[nth_byte(k7, 0)]++;

        cnt1_l0[nth_byte(k0, 1)]++;
        cnt1_l1[nth_byte(k1, 1)]++;
        cnt1_l2[nth_byte(k2, 1)]++;
        cnt1_l3[nth_byte(k3, 1)]++;
        cnt1_l0[nth_byte(k4, 1)]++;
        cnt1_l1[nth_byte(k5, 1)]++;
        cnt1_l2[nth_byte(k6, 1)]++;
        cnt1_l3[nth_byte(k7, 1)]++;
    }

    for (int j = 0; j < 256; j++) {
        cnt0[j] = cnt0_l0[j] + cnt0_l1[j] + cnt0_l2[j] + cnt0_l3[j];
        cnt1[j] = cnt1_l0[j] + cnt1_l1[j] + cnt1_l2[j] + cnt1_l3[j];
    }

    for (; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);
        cnt0[nth_byte(k, 0)]++;
        cnt1[nth_byte(k, 1)]++;
    }

    size_t ncols = 0;
    npy_ubyte cols[2];
    if (cnt0[nth_byte(key0, 0)] != num) {
        cols[ncols++] = 0;
    }
    if (cnt1[nth_byte(key0, 1)] != num) {
        cols[ncols++] = 1;
    }

    if (ncols == 0) {
        return tosort;
    }

    npy_intp *cnt[2] = {cnt0, cnt1};
    for (size_t l = 0; l < ncols; l++) {
        npy_intp a = 0;
        npy_intp *c = cnt[cols[l]];
        for (int j = 0; j < 256; j++) {
            npy_intp b = c[j];
            c[j] = a;
            a += b;
        }
    }

    npy_intp *src = tosort;
    npy_intp *dst = aux;

    for (size_t l = 0; l < ncols; l++) {
        npy_intp *c = cnt[cols[l]];

        npy_intp j = 0;
        for (; j + 8 <= num; j += 8) {
            npy_intp s0 = src[j], s1 = src[j + 1];
            npy_intp s2 = src[j + 2], s3 = src[j + 3];
            npy_intp s4 = src[j + 4], s5 = src[j + 5];
            npy_intp s6 = src[j + 6], s7 = src[j + 7];

            UT k0 = KEY_OF<T>(start[s0]);
            UT k1 = KEY_OF<T>(start[s1]);
            UT k2 = KEY_OF<T>(start[s2]);
            UT k3 = KEY_OF<T>(start[s3]);
            UT k4 = KEY_OF<T>(start[s4]);
            UT k5 = KEY_OF<T>(start[s5]);
            UT k6 = KEY_OF<T>(start[s6]);
            UT k7 = KEY_OF<T>(start[s7]);

            dst[c[nth_byte(k0, cols[l])]++] = s0;
            dst[c[nth_byte(k1, cols[l])]++] = s1;
            dst[c[nth_byte(k2, cols[l])]++] = s2;
            dst[c[nth_byte(k3, cols[l])]++] = s3;
            dst[c[nth_byte(k4, cols[l])]++] = s4;
            dst[c[nth_byte(k5, cols[l])]++] = s5;
            dst[c[nth_byte(k6, cols[l])]++] = s6;
            dst[c[nth_byte(k7, cols[l])]++] = s7;
        }
        for (; j < num; j++) {
            UT k = KEY_OF<T>(start[src[j]]);
            dst[c[nth_byte(k, cols[l])]++] = src[j];
        }

        npy_intp *tmp = dst;
        dst = src;
        src = tmp;
    }

    return src;
}

template <>
npy_intp *
aradixsort0<npy_byte, npy_ubyte>(npy_ubyte *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    return aradixsort0_8bit<npy_byte, npy_ubyte>(start, aux, tosort, num);
}

template <>
npy_intp *
aradixsort0<npy_ubyte, npy_ubyte>(npy_ubyte *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    return aradixsort0_8bit<npy_ubyte, npy_ubyte>(start, aux, tosort, num);
}

template <>
npy_intp *
aradixsort0<npy_short, npy_ushort>(npy_ushort *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    return aradixsort0_16bit<npy_short, npy_ushort>(start, aux, tosort, num);
}

template <>
npy_intp *
aradixsort0<npy_ushort, npy_ushort>(npy_ushort *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    return aradixsort0_16bit<npy_ushort, npy_ushort>(start, aux, tosort, num);
}
#else

template <class T, class UT>
static npy_intp *
aradixsort0_8bit(UT *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    npy_intp cnt[256] = {0};

    const UT key0 = KEY_OF<T>(start[0]);

    npy_intp cnt_l0[256] = {0}, cnt_l1[256] = {0};
    npy_intp cnt_l2[256] = {0}, cnt_l3[256] = {0};

    npy_intp i = 0;
    for (; i + 8 <= num; i += 8) {
        UT k0 = KEY_OF<T>(start[i]);
        UT k1 = KEY_OF<T>(start[i + 1]);
        UT k2 = KEY_OF<T>(start[i + 2]);
        UT k3 = KEY_OF<T>(start[i + 3]);
        UT k4 = KEY_OF<T>(start[i + 4]);
        UT k5 = KEY_OF<T>(start[i + 5]);
        UT k6 = KEY_OF<T>(start[i + 6]);
        UT k7 = KEY_OF<T>(start[i + 7]);

        cnt_l0[nth_byte(k0, 0)]++;
        cnt_l1[nth_byte(k1, 0)]++;
        cnt_l2[nth_byte(k2, 0)]++;
        cnt_l3[nth_byte(k3, 0)]++;
        cnt_l0[nth_byte(k4, 0)]++;
        cnt_l1[nth_byte(k5, 0)]++;
        cnt_l2[nth_byte(k6, 0)]++;
        cnt_l3[nth_byte(k7, 0)]++;
    }

    for (int j = 0; j < 256; j++) {
        cnt[j] = cnt_l0[j] + cnt_l1[j] + cnt_l2[j] + cnt_l3[j];
    }

    for (; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);
        cnt[nth_byte(k, 0)]++;
    }

    if (cnt[nth_byte(key0, 0)] == num) {
        return tosort;
    }

    npy_intp a = 0;
    for (int j = 0; j < 256; j++) {
        npy_intp b = cnt[j];
        cnt[j] = a;
        a += b;
    }

    npy_intp *src = tosort;
    npy_intp *dst = aux;

    npy_intp j = 0;
    for (; j + 8 <= num; j += 8) {
        npy_intp s0 = src[j], s1 = src[j + 1];
        npy_intp s2 = src[j + 2], s3 = src[j + 3];
        npy_intp s4 = src[j + 4], s5 = src[j + 5];
        npy_intp s6 = src[j + 6], s7 = src[j + 7];

        UT k0 = KEY_OF<T>(start[s0]);
        UT k1 = KEY_OF<T>(start[s1]);
        UT k2 = KEY_OF<T>(start[s2]);
        UT k3 = KEY_OF<T>(start[s3]);
        UT k4 = KEY_OF<T>(start[s4]);
        UT k5 = KEY_OF<T>(start[s5]);
        UT k6 = KEY_OF<T>(start[s6]);
        UT k7 = KEY_OF<T>(start[s7]);

        dst[cnt[nth_byte(k0, 0)]++] = s0;
        dst[cnt[nth_byte(k1, 0)]++] = s1;
        dst[cnt[nth_byte(k2, 0)]++] = s2;
        dst[cnt[nth_byte(k3, 0)]++] = s3;
        dst[cnt[nth_byte(k4, 0)]++] = s4;
        dst[cnt[nth_byte(k5, 0)]++] = s5;
        dst[cnt[nth_byte(k6, 0)]++] = s6;
        dst[cnt[nth_byte(k7, 0)]++] = s7;
    }
    for (; j < num; j++) {
        UT k = KEY_OF<T>(start[src[j]]);
        dst[cnt[nth_byte(k, 0)]++] = src[j];
    }

    return dst;
}

template <class T, class UT>
static npy_intp *
aradixsort0_16bit(UT *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    npy_intp cnt0[256] = {0};
    npy_intp cnt1[256] = {0};

    const UT key0 = KEY_OF<T>(start[0]);

    npy_intp cnt0_l0[256] = {0}, cnt0_l1[256] = {0};
    npy_intp cnt0_l2[256] = {0}, cnt0_l3[256] = {0};
    npy_intp cnt1_l0[256] = {0}, cnt1_l1[256] = {0};
    npy_intp cnt1_l2[256] = {0}, cnt1_l3[256] = {0};

    npy_intp i = 0;
    for (; i + 8 <= num; i += 8) {
        UT k0 = KEY_OF<T>(start[i]);
        UT k1 = KEY_OF<T>(start[i + 1]);
        UT k2 = KEY_OF<T>(start[i + 2]);
        UT k3 = KEY_OF<T>(start[i + 3]);
        UT k4 = KEY_OF<T>(start[i + 4]);
        UT k5 = KEY_OF<T>(start[i + 5]);
        UT k6 = KEY_OF<T>(start[i + 6]);
        UT k7 = KEY_OF<T>(start[i + 7]);

        cnt0_l0[nth_byte(k0, 0)]++;
        cnt0_l1[nth_byte(k1, 0)]++;
        cnt0_l2[nth_byte(k2, 0)]++;
        cnt0_l3[nth_byte(k3, 0)]++;
        cnt0_l0[nth_byte(k4, 0)]++;
        cnt0_l1[nth_byte(k5, 0)]++;
        cnt0_l2[nth_byte(k6, 0)]++;
        cnt0_l3[nth_byte(k7, 0)]++;

        cnt1_l0[nth_byte(k0, 1)]++;
        cnt1_l1[nth_byte(k1, 1)]++;
        cnt1_l2[nth_byte(k2, 1)]++;
        cnt1_l3[nth_byte(k3, 1)]++;
        cnt1_l0[nth_byte(k4, 1)]++;
        cnt1_l1[nth_byte(k5, 1)]++;
        cnt1_l2[nth_byte(k6, 1)]++;
        cnt1_l3[nth_byte(k7, 1)]++;
    }

    for (int j = 0; j < 256; j++) {
        cnt0[j] = cnt0_l0[j] + cnt0_l1[j] + cnt0_l2[j] + cnt0_l3[j];
        cnt1[j] = cnt1_l0[j] + cnt1_l1[j] + cnt1_l2[j] + cnt1_l3[j];
    }

    for (; i < num; i++) {
        UT k = KEY_OF<T>(start[i]);
        cnt0[nth_byte(k, 0)]++;
        cnt1[nth_byte(k, 1)]++;
    }

    size_t ncols = 0;
    npy_ubyte cols[2];
    if (cnt0[nth_byte(key0, 0)] != num) {
        cols[ncols++] = 0;
    }
    if (cnt1[nth_byte(key0, 1)] != num) {
        cols[ncols++] = 1;
    }

    if (ncols == 0) {
        return tosort;
    }

    npy_intp *cnt[2] = {cnt0, cnt1};
    for (size_t l = 0; l < ncols; l++) {
        npy_intp a = 0;
        npy_intp *c = cnt[cols[l]];
        for (int j = 0; j < 256; j++) {
            npy_intp b = c[j];
            c[j] = a;
            a += b;
        }
    }

    npy_intp *src = tosort;
    npy_intp *dst = aux;

    for (size_t l = 0; l < ncols; l++) {
        npy_intp *c = cnt[cols[l]];

        npy_intp j = 0;
        for (; j + 8 <= num; j += 8) {
            npy_intp s0 = src[j], s1 = src[j + 1];
            npy_intp s2 = src[j + 2], s3 = src[j + 3];
            npy_intp s4 = src[j + 4], s5 = src[j + 5];
            npy_intp s6 = src[j + 6], s7 = src[j + 7];

            UT k0 = KEY_OF<T>(start[s0]);
            UT k1 = KEY_OF<T>(start[s1]);
            UT k2 = KEY_OF<T>(start[s2]);
            UT k3 = KEY_OF<T>(start[s3]);
            UT k4 = KEY_OF<T>(start[s4]);
            UT k5 = KEY_OF<T>(start[s5]);
            UT k6 = KEY_OF<T>(start[s6]);
            UT k7 = KEY_OF<T>(start[s7]);

            dst[c[nth_byte(k0, cols[l])]++] = s0;
            dst[c[nth_byte(k1, cols[l])]++] = s1;
            dst[c[nth_byte(k2, cols[l])]++] = s2;
            dst[c[nth_byte(k3, cols[l])]++] = s3;
            dst[c[nth_byte(k4, cols[l])]++] = s4;
            dst[c[nth_byte(k5, cols[l])]++] = s5;
            dst[c[nth_byte(k6, cols[l])]++] = s6;
            dst[c[nth_byte(k7, cols[l])]++] = s7;
        }
        for (; j < num; j++) {
            UT k = KEY_OF<T>(start[src[j]]);
            dst[c[nth_byte(k, cols[l])]++] = src[j];
        }

        npy_intp *tmp = dst;
        dst = src;
        src = tmp;
    }

    return src;
}
#endif

template <class T, class UT>
static int
aradixsort_(UT *start, npy_intp *tosort, npy_intp num)
{
    npy_intp *sorted;
    npy_intp *aux;
    npy_intp i = 1;
    T *v_start = (T *)start;
    T k_prev;
    npy_bool all_sorted = 1;

    if (num < 2) {
        return 0;
    }

    k_prev = v_start[tosort[0]];
    for (i = 1; i + 8 <= num; i += 8) {
        T k0 = v_start[tosort[i]];
        T k1 = v_start[tosort[i + 1]];
        T k2 = v_start[tosort[i + 2]];
        T k3 = v_start[tosort[i + 3]];
        T k4 = v_start[tosort[i + 4]];
        T k5 = v_start[tosort[i + 5]];
        T k6 = v_start[tosort[i + 6]];
        T k7 = v_start[tosort[i + 7]];
        if (k_prev > k0 || k0 > k1 || k1 > k2 || k2 > k3 || k3 > k4 ||
            k4 > k5 || k5 > k6 || k6 > k7) {
            all_sorted = 0;
            break;
        }
        k_prev = k7;
    }

    for (; i < num && all_sorted; i++) {
        T k0 = v_start[tosort[i]];
        if (k_prev > k0) {
            all_sorted = 0;
            break;
        }
        k_prev = k0;
    }

    if (all_sorted) {
        return 0;
    }

    aux = (npy_intp *)malloc(num * sizeof(npy_intp));
    if (aux == NULL) {
        return -NPY_ENOMEM;
    }

    if constexpr (sizeof(UT) == 1) {
        sorted = aradixsort0_8bit<T>(start, aux, tosort, num);
    }
    else if constexpr (sizeof(UT) == 2) {
        sorted = aradixsort0_16bit<T>(start, aux, tosort, num);
    }
    else {
        sorted = aradixsort0<T>(start, aux, tosort, num);
    }
    if (sorted != tosort) {
        memcpy(tosort, sorted, num * sizeof(npy_intp));
    }

    free(aux);
    return 0;
}

template <class T>
static int
aradixsort(void *start, npy_intp *tosort, npy_intp num)
{
    using UT = typename std::make_unsigned<T>::type;
    return aradixsort_<T>((UT *)start, tosort, num);
}

extern "C" {
NPY_NO_EXPORT int
radixsort_bool(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_bool>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_byte(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_byte>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_ubyte(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_ubyte>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_short(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_short>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_ushort(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_ushort>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_int(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_int>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_uint(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_uint>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_long(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_long>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_ulong(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_ulong>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_longlong(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_longlong>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_ulonglong(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_ulonglong>(vec, cnt);
}
NPY_NO_EXPORT int
aradixsort_bool(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    return aradixsort<npy_bool>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_byte(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    return aradixsort<npy_byte>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_ubyte(void *vec, npy_intp *ind, npy_intp cnt,
                 void *NPY_UNUSED(null))
{
    return aradixsort<npy_ubyte>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_short(void *vec, npy_intp *ind, npy_intp cnt,
                 void *NPY_UNUSED(null))
{
    return aradixsort<npy_short>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_ushort(void *vec, npy_intp *ind, npy_intp cnt,
                  void *NPY_UNUSED(null))
{
    return aradixsort<npy_ushort>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_int(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    return aradixsort<npy_int>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_uint(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    return aradixsort<npy_uint>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_long(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    return aradixsort<npy_long>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_ulong(void *vec, npy_intp *ind, npy_intp cnt,
                 void *NPY_UNUSED(null))
{
    return aradixsort<npy_ulong>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_longlong(void *vec, npy_intp *ind, npy_intp cnt,
                    void *NPY_UNUSED(null))
{
    return aradixsort<npy_longlong>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_ulonglong(void *vec, npy_intp *ind, npy_intp cnt,
                     void *NPY_UNUSED(null))
{
    return aradixsort<npy_ulonglong>(vec, ind, cnt);
}
}
