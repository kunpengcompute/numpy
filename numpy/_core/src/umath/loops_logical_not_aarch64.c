#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdint.h>

#include <numpy/ndarraytypes.h>

#include "fast_loop_macros.h"
#include "loops_logical_not_aarch64.h"

#if defined(__aarch64__)
    #include <arm_neon.h>
#endif

#if defined(__aarch64__)

static inline int
aarch64_logical_not_contig_ok(char **args, npy_intp const *steps,
        size_t elsize, npy_intp len)
{
    return len > 0 &&
            steps[0] == (npy_intp)elsize &&
            steps[1] == (npy_intp)sizeof(npy_bool) &&
            abs_ptrdiff(args[1], args[0]) >= AUTOVEC_OVERLAP_SIZE;
}

static inline int
aarch64_logical_not_strided_ok(char **args, npy_intp const *steps,
        size_t elsize, npy_intp len)
{
    return len >= 16 &&
            (steps[0] == (npy_intp)elsize ||
             steps[0] == (npy_intp)(4 * elsize)) &&
            (steps[1] == (npy_intp)sizeof(npy_bool) ||
             steps[1] == (npy_intp)(2 * sizeof(npy_bool))) &&
            abs_ptrdiff(args[1], args[0]) >= AUTOVEC_OVERLAP_SIZE;
}

static void
half_logical_not_aarch64_contig(char **args, npy_intp len)
{
    const uint16_t *ip = (const uint16_t *)args[0];
    npy_bool *op = (npy_bool *)args[1];
    const uint16x8_t sign_mask = vdupq_n_u16((uint16_t)0x7fff);
    const uint16x8_t zero = vdupq_n_u16(0);
    const uint8x8_t one = vdup_n_u8(1);

    for (; len >= 8; len -= 8, ip += 8, op += 8) {
        uint16x8_t mag = vandq_u16(vld1q_u16(ip), sign_mask);
        uint16x8_t cmp = vceqq_u16(mag, zero);
        vst1_u8(op, vand_u8(vmovn_u16(cmp), one));
    }

    for (; len > 0; --len, ++ip, ++op) {
        *op = (npy_bool)((*ip & 0x7fffu) == 0);
    }
}

static void
float_logical_not_aarch64_contig(char **args, npy_intp len)
{
    const float *ip = (const float *)args[0];
    npy_bool *op = (npy_bool *)args[1];
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const uint8x8_t one = vdup_n_u8(1);

    for (; len >= 8; len -= 8, ip += 8, op += 8) {
        uint32x4_t cmp0 = vceqq_f32(vld1q_f32(ip), zero);
        uint32x4_t cmp1 = vceqq_f32(vld1q_f32(ip + 4), zero);
        uint16x4_t lo = vmovn_u32(cmp0);
        uint16x4_t hi = vmovn_u32(cmp1);
        vst1_u8(op, vand_u8(vmovn_u16(vcombine_u16(lo, hi)), one));
    }

    for (; len > 0; --len, ++ip, ++op) {
        *op = (npy_bool)(*ip == 0.0f);
    }
}

static void
double_logical_not_aarch64_contig(char **args, npy_intp len)
{
    const double *ip = (const double *)args[0];
    npy_bool *op = (npy_bool *)args[1];
    const float64x2_t zero = vdupq_n_f64(0.0);
    const uint8x8_t one = vdup_n_u8(1);

    for (; len >= 8; len -= 8, ip += 8, op += 8) {
        uint64x2_t cmp0 = vceqq_f64(vld1q_f64(ip), zero);
        uint64x2_t cmp1 = vceqq_f64(vld1q_f64(ip + 2), zero);
        uint64x2_t cmp2 = vceqq_f64(vld1q_f64(ip + 4), zero);
        uint64x2_t cmp3 = vceqq_f64(vld1q_f64(ip + 6), zero);
        uint32x2_t p0 = vmovn_u64(cmp0);
        uint32x2_t p1 = vmovn_u64(cmp1);
        uint32x2_t p2 = vmovn_u64(cmp2);
        uint32x2_t p3 = vmovn_u64(cmp3);
        uint16x4_t lo = vmovn_u32(vcombine_u32(p0, p1));
        uint16x4_t hi = vmovn_u32(vcombine_u32(p2, p3));
        vst1_u8(op, vand_u8(vmovn_u16(vcombine_u16(lo, hi)), one));
    }

    for (; len > 0; --len, ++ip, ++op) {
        *op = (npy_bool)(*ip == 0.0);
    }
}

static void
half_logical_not_aarch64_strided(char **args,
        npy_intp const *steps, npy_intp len)
{
    const char *ip = args[0];
    char *op = args[1];
    const npy_intp is = steps[0];
    const npy_intp os = steps[1];

    for (; len >= 8; len -= 8, ip += 8 * is, op += 8 * os) {
        *((npy_bool *)(op + 0 * os)) = (npy_bool)((*(const uint16_t *)(ip + 0 * is) & 0x7fffu) == 0);
        *((npy_bool *)(op + 1 * os)) = (npy_bool)((*(const uint16_t *)(ip + 1 * is) & 0x7fffu) == 0);
        *((npy_bool *)(op + 2 * os)) = (npy_bool)((*(const uint16_t *)(ip + 2 * is) & 0x7fffu) == 0);
        *((npy_bool *)(op + 3 * os)) = (npy_bool)((*(const uint16_t *)(ip + 3 * is) & 0x7fffu) == 0);
        *((npy_bool *)(op + 4 * os)) = (npy_bool)((*(const uint16_t *)(ip + 4 * is) & 0x7fffu) == 0);
        *((npy_bool *)(op + 5 * os)) = (npy_bool)((*(const uint16_t *)(ip + 5 * is) & 0x7fffu) == 0);
        *((npy_bool *)(op + 6 * os)) = (npy_bool)((*(const uint16_t *)(ip + 6 * is) & 0x7fffu) == 0);
        *((npy_bool *)(op + 7 * os)) = (npy_bool)((*(const uint16_t *)(ip + 7 * is) & 0x7fffu) == 0);
    }
    for (; len > 0; --len, ip += is, op += os) {
        *(npy_bool *)op = (npy_bool)((*(const uint16_t *)ip & 0x7fffu) == 0);
    }
}

static void
float_logical_not_aarch64_strided(char **args,
        npy_intp const *steps, npy_intp len)
{
    const char *ip = args[0];
    char *op = args[1];
    const npy_intp is = steps[0];
    const npy_intp os = steps[1];

    for (; len >= 8; len -= 8, ip += 8 * is, op += 8 * os) {
        *((npy_bool *)(op + 0 * os)) = (npy_bool)(*(const float *)(ip + 0 * is) == 0.0f);
        *((npy_bool *)(op + 1 * os)) = (npy_bool)(*(const float *)(ip + 1 * is) == 0.0f);
        *((npy_bool *)(op + 2 * os)) = (npy_bool)(*(const float *)(ip + 2 * is) == 0.0f);
        *((npy_bool *)(op + 3 * os)) = (npy_bool)(*(const float *)(ip + 3 * is) == 0.0f);
        *((npy_bool *)(op + 4 * os)) = (npy_bool)(*(const float *)(ip + 4 * is) == 0.0f);
        *((npy_bool *)(op + 5 * os)) = (npy_bool)(*(const float *)(ip + 5 * is) == 0.0f);
        *((npy_bool *)(op + 6 * os)) = (npy_bool)(*(const float *)(ip + 6 * is) == 0.0f);
        *((npy_bool *)(op + 7 * os)) = (npy_bool)(*(const float *)(ip + 7 * is) == 0.0f);
    }
    for (; len > 0; --len, ip += is, op += os) {
        *(npy_bool *)op = (npy_bool)(*(const float *)ip == 0.0f);
    }
}

static void
double_logical_not_aarch64_strided(char **args,
        npy_intp const *steps, npy_intp len)
{
    const char *ip = args[0];
    char *op = args[1];
    const npy_intp is = steps[0];
    const npy_intp os = steps[1];

    for (; len >= 8; len -= 8, ip += 8 * is, op += 8 * os) {
        *((npy_bool *)(op + 0 * os)) = (npy_bool)(*(const double *)(ip + 0 * is) == 0.0);
        *((npy_bool *)(op + 1 * os)) = (npy_bool)(*(const double *)(ip + 1 * is) == 0.0);
        *((npy_bool *)(op + 2 * os)) = (npy_bool)(*(const double *)(ip + 2 * is) == 0.0);
        *((npy_bool *)(op + 3 * os)) = (npy_bool)(*(const double *)(ip + 3 * is) == 0.0);
        *((npy_bool *)(op + 4 * os)) = (npy_bool)(*(const double *)(ip + 4 * is) == 0.0);
        *((npy_bool *)(op + 5 * os)) = (npy_bool)(*(const double *)(ip + 5 * is) == 0.0);
        *((npy_bool *)(op + 6 * os)) = (npy_bool)(*(const double *)(ip + 6 * is) == 0.0);
        *((npy_bool *)(op + 7 * os)) = (npy_bool)(*(const double *)(ip + 7 * is) == 0.0);
    }
    for (; len > 0; --len, ip += is, op += os) {
        *(npy_bool *)op = (npy_bool)(*(const double *)ip == 0.0);
    }
}

#endif

int
npy_half_logical_not_aarch64(
        char **args, npy_intp const *dimensions, npy_intp const *steps)
{
#if defined(__aarch64__)
    if (aarch64_logical_not_contig_ok(
            args, steps, sizeof(npy_uint16), dimensions[0])) {
        half_logical_not_aarch64_contig(args, dimensions[0]);
        return 1;
    }
    if (aarch64_logical_not_strided_ok(
            args, steps, sizeof(npy_uint16), dimensions[0])) {
        half_logical_not_aarch64_strided(args, steps, dimensions[0]);
        return 1;
    }
#else
    (void)args;
    (void)dimensions;
    (void)steps;
#endif
    return 0;
}

int
npy_float_logical_not_aarch64(
        char **args, npy_intp const *dimensions, npy_intp const *steps,
        int itemsize)
{
#if defined(__aarch64__)
    if (!aarch64_logical_not_contig_ok(
            args, steps, (size_t)itemsize, dimensions[0]) &&
            !aarch64_logical_not_strided_ok(
                    args, steps, (size_t)itemsize, dimensions[0])) {
        return 0;
    }
    if (itemsize == (int)sizeof(npy_float)) {
        if (steps[0] == (npy_intp)sizeof(npy_float) &&
                steps[1] == (npy_intp)sizeof(npy_bool)) {
            float_logical_not_aarch64_contig(args, dimensions[0]);
        }
        else {
            float_logical_not_aarch64_strided(args, steps, dimensions[0]);
        }
        return 1;
    }
    if (itemsize == (int)sizeof(npy_double)) {
        if (steps[0] == (npy_intp)sizeof(npy_double) &&
                steps[1] == (npy_intp)sizeof(npy_bool)) {
            double_logical_not_aarch64_contig(args, dimensions[0]);
        }
        else {
            double_logical_not_aarch64_strided(args, steps, dimensions[0]);
        }
        return 1;
    }
#else
    (void)args;
    (void)dimensions;
    (void)steps;
    (void)itemsize;
#endif
    return 0;
}
