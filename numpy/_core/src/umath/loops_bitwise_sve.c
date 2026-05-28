#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "loops_bitwise_sve.h"
#include "loops_sve_utils.h"

#define NPY_SVE_OP_AND 0
#define NPY_SVE_OP_OR 1
#define NPY_SVE_OP_XOR 2

int
npy_bitwise_sve_available(void)
{
    return npy_sve_intrinsics_available();
}

#if NPY_HAVE_ARM_SVE_INTRINSICS

static inline NPY_SVE_TARGET svuint8_t
npy_bitwise_op_u8(svbool_t pg, svuint8_t a, svuint8_t b, int op)
{
    if (op == NPY_SVE_OP_AND) {
        return svand_u8_x(pg, a, b);
    }
    if (op == NPY_SVE_OP_OR) {
        return svorr_u8_x(pg, a, b);
    }
    return sveor_u8_x(pg, a, b);
}

static inline NPY_SVE_TARGET svuint16_t
npy_bitwise_op_u16(svbool_t pg, svuint16_t a, svuint16_t b, int op)
{
    if (op == NPY_SVE_OP_AND) {
        return svand_u16_x(pg, a, b);
    }
    if (op == NPY_SVE_OP_OR) {
        return svorr_u16_x(pg, a, b);
    }
    return sveor_u16_x(pg, a, b);
}

static inline NPY_SVE_TARGET svuint32_t
npy_bitwise_op_u32(svbool_t pg, svuint32_t a, svuint32_t b, int op)
{
    if (op == NPY_SVE_OP_AND) {
        return svand_u32_x(pg, a, b);
    }
    if (op == NPY_SVE_OP_OR) {
        return svorr_u32_x(pg, a, b);
    }
    return sveor_u32_x(pg, a, b);
}

static inline NPY_SVE_TARGET svuint64_t
npy_bitwise_op_u64(svbool_t pg, svuint64_t a, svuint64_t b, int op)
{
    if (op == NPY_SVE_OP_AND) {
        return svand_u64_x(pg, a, b);
    }
    if (op == NPY_SVE_OP_OR) {
        return svorr_u64_x(pg, a, b);
    }
    return sveor_u64_x(pg, a, b);
}

#endif

NPY_SVE_TARGET void
npy_bitwise_and_sve_i64(char **args, npy_intp len)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    const int64_t *in0 = (const int64_t *)args[0];
    const int64_t *in1 = (const int64_t *)args[1];
    int64_t *out = (int64_t *)args[2];
    npy_intp i = 0;

    for (; i < len; ) {
        svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)len);
        svint64_t a = svld1_s64(pg, in0 + i);
        svint64_t b = svld1_s64(pg, in1 + i);
        svst1_s64(pg, out + i, svand_s64_x(pg, a, b));
        i += (npy_intp)svcntd();
    }
#else
    (void)args;
    (void)len;
#endif
}

NPY_SVE_TARGET void
npy_bitwise_sve_contig(char **args, npy_intp len, int itemsize, int op)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    npy_intp i = 0;

    switch (itemsize) {
        case 1: {
            const uint8_t *in0 = (const uint8_t *)args[0];
            const uint8_t *in1 = (const uint8_t *)args[1];
            uint8_t *out = (uint8_t *)args[2];
            for (; i < len; ) {
                svbool_t pg = svwhilelt_b8((uint64_t)i, (uint64_t)len);
                svuint8_t a = svld1_u8(pg, in0 + i);
                svuint8_t b = svld1_u8(pg, in1 + i);
                svst1_u8(pg, out + i, npy_bitwise_op_u8(pg, a, b, op));
                i += (npy_intp)svcntb();
            }
            return;
        }
        case 2: {
            const uint16_t *in0 = (const uint16_t *)args[0];
            const uint16_t *in1 = (const uint16_t *)args[1];
            uint16_t *out = (uint16_t *)args[2];
            for (; i < len; ) {
                svbool_t pg = svwhilelt_b16((uint64_t)i, (uint64_t)len);
                svuint16_t a = svld1_u16(pg, in0 + i);
                svuint16_t b = svld1_u16(pg, in1 + i);
                svst1_u16(pg, out + i, npy_bitwise_op_u16(pg, a, b, op));
                i += (npy_intp)svcnth();
            }
            return;
        }
        case 4: {
            const uint32_t *in0 = (const uint32_t *)args[0];
            const uint32_t *in1 = (const uint32_t *)args[1];
            uint32_t *out = (uint32_t *)args[2];
            for (; i < len; ) {
                svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)len);
                svuint32_t a = svld1_u32(pg, in0 + i);
                svuint32_t b = svld1_u32(pg, in1 + i);
                svst1_u32(pg, out + i, npy_bitwise_op_u32(pg, a, b, op));
                i += (npy_intp)svcntw();
            }
            return;
        }
        case 8: {
            const uint64_t *in0 = (const uint64_t *)args[0];
            const uint64_t *in1 = (const uint64_t *)args[1];
            uint64_t *out = (uint64_t *)args[2];
            for (; i < len; ) {
                svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)len);
                svuint64_t a = svld1_u64(pg, in0 + i);
                svuint64_t b = svld1_u64(pg, in1 + i);
                svst1_u64(pg, out + i, npy_bitwise_op_u64(pg, a, b, op));
                i += (npy_intp)svcntd();
            }
            return;
        }
        default:
            return;
    }
#else
    (void)args;
    (void)len;
    (void)itemsize;
    (void)op;
#endif
}

NPY_SVE_TARGET void
npy_bitwise_sve_scalar_in0(char **args, npy_intp len, int itemsize, int op)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    npy_intp i = 0;

    switch (itemsize) {
        case 1: {
            const svuint8_t a = svdup_u8(*(const uint8_t *)args[0]);
            const uint8_t *in1 = (const uint8_t *)args[1];
            uint8_t *out = (uint8_t *)args[2];
            for (; i < len; ) {
                svbool_t pg = svwhilelt_b8((uint64_t)i, (uint64_t)len);
                svuint8_t b = svld1_u8(pg, in1 + i);
                svst1_u8(pg, out + i, npy_bitwise_op_u8(pg, a, b, op));
                i += (npy_intp)svcntb();
            }
            return;
        }
        case 2: {
            const svuint16_t a = svdup_u16(*(const uint16_t *)args[0]);
            const uint16_t *in1 = (const uint16_t *)args[1];
            uint16_t *out = (uint16_t *)args[2];
            for (; i < len; ) {
                svbool_t pg = svwhilelt_b16((uint64_t)i, (uint64_t)len);
                svuint16_t b = svld1_u16(pg, in1 + i);
                svst1_u16(pg, out + i, npy_bitwise_op_u16(pg, a, b, op));
                i += (npy_intp)svcnth();
            }
            return;
        }
        case 4: {
            const svuint32_t a = svdup_u32(*(const uint32_t *)args[0]);
            const uint32_t *in1 = (const uint32_t *)args[1];
            uint32_t *out = (uint32_t *)args[2];
            for (; i < len; ) {
                svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)len);
                svuint32_t b = svld1_u32(pg, in1 + i);
                svst1_u32(pg, out + i, npy_bitwise_op_u32(pg, a, b, op));
                i += (npy_intp)svcntw();
            }
            return;
        }
        case 8: {
            const svuint64_t a = svdup_u64(*(const uint64_t *)args[0]);
            const uint64_t *in1 = (const uint64_t *)args[1];
            uint64_t *out = (uint64_t *)args[2];
            for (; i < len; ) {
                svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)len);
                svuint64_t b = svld1_u64(pg, in1 + i);
                svst1_u64(pg, out + i, npy_bitwise_op_u64(pg, a, b, op));
                i += (npy_intp)svcntd();
            }
            return;
        }
        default:
            return;
    }
#else
    (void)args;
    (void)len;
    (void)itemsize;
    (void)op;
#endif
}

NPY_SVE_TARGET void
npy_bitwise_sve_scalar_in1(char **args, npy_intp len, int itemsize, int op)
{
    char *tmp = args[0];

    args[0] = args[1];
    args[1] = tmp;
    npy_bitwise_sve_scalar_in0(args, len, itemsize, op);
    tmp = args[0];
    args[0] = args[1];
    args[1] = tmp;
}

NPY_SVE_TARGET void
npy_logical_xor_sve_u8(char **args, npy_intp len)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    const uint8_t *in0 = (const uint8_t *)args[0];
    const uint8_t *in1 = (const uint8_t *)args[1];
    uint8_t *out = (uint8_t *)args[2];
    const svuint8_t ones = svdup_n_u8(1);
    const svuint8_t zeros = svdup_n_u8(0);
    npy_intp i = 0;

    for (; i < len; ) {
        svbool_t pg = svwhilelt_b8((uint64_t)i, (uint64_t)len);
        svuint8_t a = svld1_u8(pg, in0 + i);
        svuint8_t b = svld1_u8(pg, in1 + i);
        svbool_t pa = svcmpne_n_u8(pg, a, 0);
        svbool_t pb = svcmpne_n_u8(pg, b, 0);
        svst1_u8(pg, out + i, svsel_u8(sveor_b_z(pg, pa, pb), ones, zeros));
        i += (npy_intp)svcntb();
    }
#else
    (void)args;
    (void)len;
#endif
}

NPY_SVE_TARGET void
npy_logical_xor_sve_contig(char **args, npy_intp len, int itemsize)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    const svuint8_t ones8 = svdup_n_u8(1);
    const svuint8_t zeros8 = svdup_n_u8(0);
    npy_intp i = 0;

    switch (itemsize) {
        case 1:
            npy_logical_xor_sve_u8(args, len);
            return;
        case 2: {
            const uint16_t *in0 = (const uint16_t *)args[0];
            const uint16_t *in1 = (const uint16_t *)args[1];
            uint8_t *out = (uint8_t *)args[2];
            for (; i < len; ) {
                svbool_t pg = svwhilelt_b16((uint64_t)i, (uint64_t)len);
                svuint16_t a = svld1_u16(pg, in0 + i);
                svuint16_t b = svld1_u16(pg, in1 + i);
                svbool_t px = sveor_b_z(
                        pg, svcmpne_n_u16(pg, a, 0), svcmpne_n_u16(pg, b, 0));
                svst1b_u16(pg, out + i, svsel_u16(px, svdup_n_u16(1), svdup_n_u16(0)));
                i += (npy_intp)svcnth();
            }
            return;
        }
        case 4: {
            const uint32_t *in0 = (const uint32_t *)args[0];
            const uint32_t *in1 = (const uint32_t *)args[1];
            uint8_t *out = (uint8_t *)args[2];
            for (; i < len; ) {
                svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)len);
                svuint32_t a = svld1_u32(pg, in0 + i);
                svuint32_t b = svld1_u32(pg, in1 + i);
                svbool_t px = sveor_b_z(
                        pg, svcmpne_n_u32(pg, a, 0), svcmpne_n_u32(pg, b, 0));
                svst1b_u32(pg, out + i, svsel_u32(px, svdup_n_u32(1), svdup_n_u32(0)));
                i += (npy_intp)svcntw();
            }
            return;
        }
        case 8: {
            const uint64_t *in0 = (const uint64_t *)args[0];
            const uint64_t *in1 = (const uint64_t *)args[1];
            uint8_t *out = (uint8_t *)args[2];
            for (; i < len; ) {
                svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)len);
                svuint64_t a = svld1_u64(pg, in0 + i);
                svuint64_t b = svld1_u64(pg, in1 + i);
                svbool_t px = sveor_b_z(
                        pg, svcmpne_n_u64(pg, a, 0), svcmpne_n_u64(pg, b, 0));
                svst1b_u64(pg, out + i, svsel_u64(px, svdup_n_u64(1), svdup_n_u64(0)));
                i += (npy_intp)svcntd();
            }
            return;
        }
        default:
            (void)ones8;
            (void)zeros8;
            return;
    }
#else
    (void)args;
    (void)len;
    (void)itemsize;
#endif
}
