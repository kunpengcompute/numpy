#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "loops_shift_sve.h"
#include "loops_sve_utils.h"

int
npy_shift_sve_scalar_available(void)
{
    return npy_sve_intrinsics_available();
}

#if NPY_HAVE_ARM_SVE_INTRINSICS

static inline NPY_SVE_TARGET svint8_t
LSHIFT_s8(svbool_t pg, svint8_t a, svuint8_t count)
{
    return svlsl_s8_x(pg, a, count);
}

static inline NPY_SVE_TARGET svint16_t
LSHIFT_s16(svbool_t pg, svint16_t a, svuint16_t count)
{
    return svlsl_s16_x(pg, a, count);
}

static inline NPY_SVE_TARGET svint32_t
LSHIFT_s32(svbool_t pg, svint32_t a, svuint32_t count)
{
    return svlsl_s32_x(pg, a, count);
}

static inline NPY_SVE_TARGET svint64_t
LSHIFT_s64(svbool_t pg, svint64_t a, svuint64_t count)
{
    return svlsl_s64_x(pg, a, count);
}

static inline NPY_SVE_TARGET svuint8_t
LSHIFT_u8(svbool_t pg, svuint8_t a, svuint8_t count)
{
    return svlsl_u8_x(pg, a, count);
}

static inline NPY_SVE_TARGET svuint16_t
LSHIFT_u16(svbool_t pg, svuint16_t a, svuint16_t count)
{
    return svlsl_u16_x(pg, a, count);
}

static inline NPY_SVE_TARGET svuint32_t
LSHIFT_u32(svbool_t pg, svuint32_t a, svuint32_t count)
{
    return svlsl_u32_x(pg, a, count);
}

static inline NPY_SVE_TARGET svuint64_t
LSHIFT_u64(svbool_t pg, svuint64_t a, svuint64_t count)
{
    return svlsl_u64_x(pg, a, count);
}

static inline NPY_SVE_TARGET svint8_t
RSHIFT_s8(svbool_t pg, svint8_t a, svuint8_t count)
{
    return svasr_s8_x(pg, a, count);
}

static inline NPY_SVE_TARGET svint16_t
RSHIFT_s16(svbool_t pg, svint16_t a, svuint16_t count)
{
    return svasr_s16_x(pg, a, count);
}

static inline NPY_SVE_TARGET svint32_t
RSHIFT_s32(svbool_t pg, svint32_t a, svuint32_t count)
{
    return svasr_s32_x(pg, a, count);
}

static inline NPY_SVE_TARGET svint64_t
RSHIFT_s64(svbool_t pg, svint64_t a, svuint64_t count)
{
    return svasr_s64_x(pg, a, count);
}

static inline NPY_SVE_TARGET svuint8_t
RSHIFT_u8(svbool_t pg, svuint8_t a, svuint8_t count)
{
    return svlsr_u8_x(pg, a, count);
}

static inline NPY_SVE_TARGET svuint16_t
RSHIFT_u16(svbool_t pg, svuint16_t a, svuint16_t count)
{
    return svlsr_u16_x(pg, a, count);
}

static inline NPY_SVE_TARGET svuint32_t
RSHIFT_u32(svbool_t pg, svuint32_t a, svuint32_t count)
{
    return svlsr_u32_x(pg, a, count);
}

static inline NPY_SVE_TARGET svuint64_t
RSHIFT_u64(svbool_t pg, svuint64_t a, svuint64_t count)
{
    return svlsr_u64_x(pg, a, count);
}

#define svdup_svint8_t svdup_s8
#define svdup_svint16_t svdup_s16
#define svdup_svint32_t svdup_s32
#define svdup_svint64_t svdup_s64
#define svdup_svuint8_t svdup_u8
#define svdup_svuint16_t svdup_u16
#define svdup_svuint32_t svdup_u32
#define svdup_svuint64_t svdup_u64

#define svld1_svint8_t svld1_s8
#define svld1_svint16_t svld1_s16
#define svld1_svint32_t svld1_s32
#define svld1_svint64_t svld1_s64
#define svld1_svuint8_t svld1_u8
#define svld1_svuint16_t svld1_u16
#define svld1_svuint32_t svld1_u32
#define svld1_svuint64_t svld1_u64

#define svst1_svint8_t svst1_s8
#define svst1_svint16_t svst1_s16
#define svst1_svint32_t svst1_s32
#define svst1_svint64_t svst1_s64
#define svst1_svuint8_t svst1_u8
#define svst1_svuint16_t svst1_u16
#define svst1_svuint32_t svst1_u32
#define svst1_svuint64_t svst1_u64

#define svsel_svint8_t svsel_s8
#define svsel_svint16_t svsel_s16
#define svsel_svint32_t svsel_s32
#define svsel_svint64_t svsel_s64
#define svsel_svuint8_t svsel_u8
#define svsel_svuint16_t svsel_u16
#define svsel_svuint32_t svsel_u32
#define svsel_svuint64_t svsel_u64

#define svreinterpret_u8_svint8_t svreinterpret_u8_s8
#define svreinterpret_u16_svint16_t svreinterpret_u16_s16
#define svreinterpret_u32_svint32_t svreinterpret_u32_s32
#define svreinterpret_u64_svint64_t svreinterpret_u64_s64

#define svcmplt_svint8_t svcmplt_s8
#define svcmplt_svint16_t svcmplt_s16
#define svcmplt_svint32_t svcmplt_s32
#define svcmplt_svint64_t svcmplt_s64

#define DEFINE_LSHIFT_CONTIG_UNSIGNED(BITS, PRED_B, CNT_FN)                         \
static NPY_SVE_TARGET void                                                          \
npy_left_shift_contig_u##BITS(char **args, npy_intp len)                            \
{                                                                                    \
    const uint##BITS##_t *in0 = (const uint##BITS##_t *)args[0];                    \
    const uint##BITS##_t *in1 = (const uint##BITS##_t *)args[1];                    \
    uint##BITS##_t *out = (uint##BITS##_t *)args[2];                                \
    const svuint##BITS##_t vmax = svdup_u##BITS((BITS) - 1);                        \
    npy_intp i = 0;                                                                  \
    for (; i < len; i += CNT_FN()) {                                                 \
        svbool_t pg = svwhilelt_b##PRED_B((uint64_t)i, (uint64_t)len);               \
        svuint##BITS##_t a = svld1_u##BITS(pg, in0 + i);                            \
        svuint##BITS##_t count = svld1_u##BITS(pg, in1 + i);                        \
        svbool_t ok = svcmple_u##BITS(pg, count, vmax);                             \
        svuint##BITS##_t shifted = svlsl_u##BITS##_x(pg, a, count);                 \
        svst1_u##BITS(pg, out + i, svsel_u##BITS(ok, shifted, svdup_u##BITS(0)));   \
    }                                                                                \
}

#define DEFINE_RSHIFT_CONTIG_UNSIGNED(BITS, PRED_B, CNT_FN)                         \
static NPY_SVE_TARGET void                                                          \
npy_right_shift_contig_u##BITS(char **args, npy_intp len)                           \
{                                                                                    \
    const uint##BITS##_t *in0 = (const uint##BITS##_t *)args[0];                    \
    const uint##BITS##_t *in1 = (const uint##BITS##_t *)args[1];                    \
    uint##BITS##_t *out = (uint##BITS##_t *)args[2];                                \
    const svuint##BITS##_t vmax = svdup_u##BITS((BITS) - 1);                        \
    npy_intp i = 0;                                                                  \
    for (; i < len; i += CNT_FN()) {                                                 \
        svbool_t pg = svwhilelt_b##PRED_B((uint64_t)i, (uint64_t)len);               \
        svuint##BITS##_t a = svld1_u##BITS(pg, in0 + i);                            \
        svuint##BITS##_t count = svld1_u##BITS(pg, in1 + i);                        \
        svbool_t ok = svcmple_u##BITS(pg, count, vmax);                             \
        svuint##BITS##_t shifted = svlsr_u##BITS##_x(pg, a, count);                 \
        svst1_u##BITS(pg, out + i, svsel_u##BITS(ok, shifted, svdup_u##BITS(0)));   \
    }                                                                                \
}

#define DEFINE_LSHIFT_CONTIG_SIGNED(BITS, PRED_B, CNT_FN)                           \
static NPY_SVE_TARGET void                                                          \
npy_left_shift_contig_s##BITS(char **args, npy_intp len)                            \
{                                                                                    \
    const int##BITS##_t *in0 = (const int##BITS##_t *)args[0];                      \
    const int##BITS##_t *in1 = (const int##BITS##_t *)args[1];                      \
    int##BITS##_t *out = (int##BITS##_t *)args[2];                                  \
    const svuint##BITS##_t vmax = svdup_u##BITS((BITS) - 1);                        \
    npy_intp i = 0;                                                                  \
    for (; i < len; i += CNT_FN()) {                                                 \
        svbool_t pg = svwhilelt_b##PRED_B((uint64_t)i, (uint64_t)len);               \
        svint##BITS##_t a = svld1_s##BITS(pg, in0 + i);                             \
        svint##BITS##_t b = svld1_s##BITS(pg, in1 + i);                             \
        svuint##BITS##_t count = svreinterpret_u##BITS##_s##BITS(b);                \
        svbool_t ok = svcmple_u##BITS(pg, count, vmax);                             \
        svint##BITS##_t shifted = svlsl_s##BITS##_x(pg, a, count);                  \
        svst1_s##BITS(pg, out + i, svsel_s##BITS(ok, shifted, svdup_s##BITS(0)));   \
    }                                                                                \
}

#define DEFINE_RSHIFT_CONTIG_SIGNED(BITS, PRED_B, CNT_FN)                           \
static NPY_SVE_TARGET void                                                          \
npy_right_shift_contig_s##BITS(char **args, npy_intp len)                           \
{                                                                                    \
    const int##BITS##_t *in0 = (const int##BITS##_t *)args[0];                      \
    const int##BITS##_t *in1 = (const int##BITS##_t *)args[1];                      \
    int##BITS##_t *out = (int##BITS##_t *)args[2];                                  \
    const svuint##BITS##_t vmax = svdup_u##BITS((BITS) - 1);                        \
    npy_intp i = 0;                                                                  \
    for (; i < len; i += CNT_FN()) {                                                 \
        svbool_t pg = svwhilelt_b##PRED_B((uint64_t)i, (uint64_t)len);               \
        svint##BITS##_t a = svld1_s##BITS(pg, in0 + i);                             \
        svint##BITS##_t b = svld1_s##BITS(pg, in1 + i);                             \
        svuint##BITS##_t count = svreinterpret_u##BITS##_s##BITS(b);                \
        svbool_t ok = svcmple_u##BITS(pg, count, vmax);                             \
        svint##BITS##_t shifted = svasr_s##BITS##_x(pg, a, count);                  \
        svint##BITS##_t fill = svdup_s##BITS(0);                                    \
        svbool_t neg = svcmplt_n_s##BITS(pg, a, 0);                                 \
        fill = svsel_s##BITS(neg, svdup_s##BITS(-1), fill);                         \
        svst1_s##BITS(pg, out + i, svsel_s##BITS(ok, shifted, fill));               \
    }                                                                                \
}

DEFINE_LSHIFT_CONTIG_SIGNED(8, 8, svcntb)
DEFINE_LSHIFT_CONTIG_UNSIGNED(8, 8, svcntb)
DEFINE_LSHIFT_CONTIG_SIGNED(16, 16, svcnth)
DEFINE_LSHIFT_CONTIG_UNSIGNED(16, 16, svcnth)
DEFINE_LSHIFT_CONTIG_SIGNED(32, 32, svcntw)
DEFINE_LSHIFT_CONTIG_UNSIGNED(32, 32, svcntw)
DEFINE_LSHIFT_CONTIG_SIGNED(64, 64, svcntd)
DEFINE_LSHIFT_CONTIG_UNSIGNED(64, 64, svcntd)

DEFINE_RSHIFT_CONTIG_SIGNED(8, 8, svcntb)
DEFINE_RSHIFT_CONTIG_UNSIGNED(8, 8, svcntb)
DEFINE_RSHIFT_CONTIG_SIGNED(16, 16, svcnth)
DEFINE_RSHIFT_CONTIG_UNSIGNED(16, 16, svcnth)
DEFINE_RSHIFT_CONTIG_SIGNED(32, 32, svcntw)
DEFINE_RSHIFT_CONTIG_UNSIGNED(32, 32, svcntw)
DEFINE_RSHIFT_CONTIG_SIGNED(64, 64, svcntd)
DEFINE_RSHIFT_CONTIG_UNSIGNED(64, 64, svcntd)

#define DEFINE_SIGNED_LSHIFT_SCALAR_IN0(TYPE, CTYPE, SVT, UBITS, PRED_B, CNT_FN, OP) \
NPY_NO_EXPORT NPY_SVE_TARGET void                                                   \
TYPE##_left_shift_sve_scalar_in0(char **args, npy_intp len)                         \
{                                                                                    \
    const CTYPE scalar = *(const CTYPE *)args[0];                                    \
    const SVT a = svdup_##SVT(scalar);                                               \
    const CTYPE *in1 = (const CTYPE *)args[1];                                       \
    CTYPE *out = (CTYPE *)args[2];                                                   \
    const svuint##UBITS##_t vmax = svdup_u##UBITS((UBITS) - 1);                      \
    npy_intp i = 0;                                                                  \
    for (; i < len; i += CNT_FN()) {                                                 \
        svbool_t pg = svwhilelt_b##PRED_B((uint64_t)i, (uint64_t)len);               \
        SVT b = svld1_##SVT(pg, &in1[i]);                                            \
        svuint##UBITS##_t count = svreinterpret_u##UBITS##_##SVT(b);                 \
        svbool_t ok = svcmple_u##UBITS(pg, count, vmax);                             \
        SVT shifted = OP(pg, a, count);                                              \
        svst1_##SVT(pg, &out[i], svsel_##SVT(ok, shifted, svdup_##SVT(0)));          \
    }                                                                                \
}

#define DEFINE_UNSIGNED_LSHIFT_SCALAR_IN0(TYPE, CTYPE, SVT, UBITS, PRED_B, CNT_FN, OP) \
NPY_NO_EXPORT NPY_SVE_TARGET void                                                     \
TYPE##_left_shift_sve_scalar_in0(char **args, npy_intp len)                           \
{                                                                                      \
    const CTYPE scalar = *(const CTYPE *)args[0];                                      \
    const SVT a = svdup_##SVT(scalar);                                                 \
    const CTYPE *in1 = (const CTYPE *)args[1];                                         \
    CTYPE *out = (CTYPE *)args[2];                                                     \
    const svuint##UBITS##_t vmax = svdup_u##UBITS((UBITS) - 1);                        \
    npy_intp i = 0;                                                                    \
    for (; i < len; i += CNT_FN()) {                                                   \
        svbool_t pg = svwhilelt_b##PRED_B((uint64_t)i, (uint64_t)len);                 \
        SVT count = svld1_##SVT(pg, &in1[i]);                                          \
        svbool_t ok = svcmple_u##UBITS(pg, count, vmax);                               \
        SVT shifted = OP(pg, a, count);                                                \
        svst1_##SVT(pg, &out[i], svsel_##SVT(ok, shifted, svdup_##SVT(0)));            \
    }                                                                                  \
}

#define DEFINE_SIGNED_RSHIFT_SCALAR_IN0(TYPE, CTYPE, SVT, UBITS, PRED_B, CNT_FN, OP) \
NPY_NO_EXPORT NPY_SVE_TARGET void                                                   \
TYPE##_right_shift_sve_scalar_in0(char **args, npy_intp len)                        \
{                                                                                    \
    const CTYPE scalar = *(const CTYPE *)args[0];                                    \
    const SVT a = svdup_##SVT(scalar);                                               \
    const SVT fill = scalar < 0 ? svdup_##SVT(-1) : svdup_##SVT(0);                  \
    const CTYPE *in1 = (const CTYPE *)args[1];                                       \
    CTYPE *out = (CTYPE *)args[2];                                                   \
    const svuint##UBITS##_t vmax = svdup_u##UBITS((UBITS) - 1);                      \
    npy_intp i = 0;                                                                  \
    for (; i < len; i += CNT_FN()) {                                                 \
        svbool_t pg = svwhilelt_b##PRED_B((uint64_t)i, (uint64_t)len);               \
        SVT b = svld1_##SVT(pg, &in1[i]);                                            \
        svuint##UBITS##_t count = svreinterpret_u##UBITS##_##SVT(b);                 \
        svbool_t ok = svcmple_u##UBITS(pg, count, vmax);                             \
        SVT shifted = OP(pg, a, count);                                              \
        svst1_##SVT(pg, &out[i], svsel_##SVT(ok, shifted, fill));                    \
    }                                                                                \
}

#define DEFINE_UNSIGNED_RSHIFT_SCALAR_IN0(TYPE, CTYPE, SVT, UBITS, PRED_B, CNT_FN, OP) \
NPY_NO_EXPORT NPY_SVE_TARGET void                                                     \
TYPE##_right_shift_sve_scalar_in0(char **args, npy_intp len)                          \
{                                                                                      \
    const CTYPE scalar = *(const CTYPE *)args[0];                                      \
    const SVT a = svdup_##SVT(scalar);                                                 \
    const CTYPE *in1 = (const CTYPE *)args[1];                                         \
    CTYPE *out = (CTYPE *)args[2];                                                     \
    const svuint##UBITS##_t vmax = svdup_u##UBITS((UBITS) - 1);                        \
    npy_intp i = 0;                                                                    \
    for (; i < len; i += CNT_FN()) {                                                   \
        svbool_t pg = svwhilelt_b##PRED_B((uint64_t)i, (uint64_t)len);                 \
        SVT count = svld1_##SVT(pg, &in1[i]);                                          \
        svbool_t ok = svcmple_u##UBITS(pg, count, vmax);                               \
        SVT shifted = OP(pg, a, count);                                                \
        svst1_##SVT(pg, &out[i], svsel_##SVT(ok, shifted, svdup_##SVT(0)));            \
    }                                                                                  \
}

DEFINE_SIGNED_LSHIFT_SCALAR_IN0(BYTE, npy_int8, svint8_t, 8, 8, svcntb, LSHIFT_s8)
DEFINE_UNSIGNED_LSHIFT_SCALAR_IN0(UBYTE, npy_uint8, svuint8_t, 8, 8, svcntb, LSHIFT_u8)
DEFINE_SIGNED_LSHIFT_SCALAR_IN0(SHORT, npy_int16, svint16_t, 16, 16, svcnth, LSHIFT_s16)
DEFINE_UNSIGNED_LSHIFT_SCALAR_IN0(USHORT, npy_uint16, svuint16_t, 16, 16, svcnth, LSHIFT_u16)

DEFINE_SIGNED_RSHIFT_SCALAR_IN0(BYTE, npy_int8, svint8_t, 8, 8, svcntb, RSHIFT_s8)
DEFINE_UNSIGNED_RSHIFT_SCALAR_IN0(UBYTE, npy_uint8, svuint8_t, 8, 8, svcntb, RSHIFT_u8)
DEFINE_SIGNED_RSHIFT_SCALAR_IN0(SHORT, npy_int16, svint16_t, 16, 16, svcnth, RSHIFT_s16)
DEFINE_UNSIGNED_RSHIFT_SCALAR_IN0(USHORT, npy_uint16, svuint16_t, 16, 16, svcnth, RSHIFT_u16)
DEFINE_SIGNED_RSHIFT_SCALAR_IN0(LONG, npy_long, svint64_t, 64, 64, svcntd, RSHIFT_s64)

NPY_NO_EXPORT NPY_SVE_TARGET void
LONG_left_shift_sve_scalar_in0(char **args, npy_intp len)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    const npy_long scalar = *(const npy_long *)args[0];
    const svint64_t a = svdup_s64((int64_t)scalar);
    const npy_long *in1 = (const npy_long *)args[1];
    npy_long *out = (npy_long *)args[2];
    const svuint64_t vmax = svdup_u64(63);
    npy_intp i = 0;

    for (; i < len; i += svcntd()) {
        svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)len);
        svint64_t b = svld1_s64(pg, (const int64_t *)&in1[i]);
        svuint64_t count = svreinterpret_u64_s64(b);
        svbool_t ok = svcmple_u64(pg, count, vmax);
        svint64_t shifted = svlsl_s64_x(pg, a, count);
        svst1_s64(pg, (int64_t *)&out[i], svsel_s64(ok, shifted, svdup_s64(0)));
    }
#else
    (void)args;
    (void)len;
#endif
}

NPY_NO_EXPORT NPY_SVE_TARGET void
ULONG_left_shift_sve_scalar_in0(char **args, npy_intp len)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    const npy_ulong scalar = *(const npy_ulong *)args[0];
    const svuint64_t a = svdup_u64((uint64_t)scalar);
    const npy_ulong *in1 = (const npy_ulong *)args[1];
    npy_ulong *out = (npy_ulong *)args[2];
    const svuint64_t vmax = svdup_u64(63);
    npy_intp i = 0;

    for (; i < len; i += svcntd()) {
        svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)len);
        svuint64_t count = svld1_u64(pg, (const uint64_t *)&in1[i]);
        svbool_t ok = svcmple_u64(pg, count, vmax);
        svuint64_t shifted = svlsl_u64_x(pg, a, count);
        svst1_u64(pg, (uint64_t *)&out[i], svsel_u64(ok, shifted, svdup_u64(0)));
    }
#else
    (void)args;
    (void)len;
#endif
}

NPY_NO_EXPORT NPY_SVE_TARGET void
ULONG_right_shift_sve_scalar_in0(char **args, npy_intp len)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    const npy_ulong scalar = *(const npy_ulong *)args[0];
    const svuint64_t a = svdup_u64((uint64_t)scalar);
    const npy_ulong *in1 = (const npy_ulong *)args[1];
    npy_ulong *out = (npy_ulong *)args[2];
    const svuint64_t vmax = svdup_u64(63);
    npy_intp i = 0;

    for (; i < len; i += svcntd()) {
        svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)len);
        svuint64_t count = svld1_u64(pg, (const uint64_t *)&in1[i]);
        svbool_t ok = svcmple_u64(pg, count, vmax);
        svuint64_t shifted = svlsr_u64_x(pg, a, count);
        svst1_u64(pg, (uint64_t *)&out[i], svsel_u64(ok, shifted, svdup_u64(0)));
    }
#else
    (void)args;
    (void)len;
#endif
}

NPY_NO_EXPORT NPY_SVE_TARGET void
LONGLONG_left_shift_sve_scalar_in0(char **args, npy_intp len)
{
    LONG_left_shift_sve_scalar_in0(args, len);
}

NPY_NO_EXPORT NPY_SVE_TARGET void
LONGLONG_right_shift_sve_scalar_in0(char **args, npy_intp len)
{
    LONG_right_shift_sve_scalar_in0(args, len);
}

NPY_NO_EXPORT NPY_SVE_TARGET void
ULONGLONG_left_shift_sve_scalar_in0(char **args, npy_intp len)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    const npy_ulonglong scalar = *(const npy_ulonglong *)args[0];
    const svuint64_t a = svdup_u64((uint64_t)scalar);
    const npy_ulonglong *in1 = (const npy_ulonglong *)args[1];
    npy_ulonglong *out = (npy_ulonglong *)args[2];
    const svuint64_t vmax = svdup_u64(63);
    npy_intp i = 0;

    for (; i < len; i += svcntd()) {
        svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)len);
        svuint64_t count = svld1_u64(pg, (const uint64_t *)&in1[i]);
        svbool_t ok = svcmple_u64(pg, count, vmax);
        svuint64_t shifted = svlsl_u64_x(pg, a, count);
        svst1_u64(pg, (uint64_t *)&out[i], svsel_u64(ok, shifted, svdup_u64(0)));
    }
#else
    (void)args;
    (void)len;
#endif
}

NPY_NO_EXPORT NPY_SVE_TARGET void
ULONGLONG_right_shift_sve_scalar_in0(char **args, npy_intp len)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    const npy_ulonglong scalar = *(const npy_ulonglong *)args[0];
    const svuint64_t a = svdup_u64((uint64_t)scalar);
    const npy_ulonglong *in1 = (const npy_ulonglong *)args[1];
    npy_ulonglong *out = (npy_ulonglong *)args[2];
    const svuint64_t vmax = svdup_u64(63);
    npy_intp i = 0;

    for (; i < len; i += svcntd()) {
        svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)len);
        svuint64_t count = svld1_u64(pg, (const uint64_t *)&in1[i]);
        svbool_t ok = svcmple_u64(pg, count, vmax);
        svuint64_t shifted = svlsr_u64_x(pg, a, count);
        svst1_u64(pg, (uint64_t *)&out[i], svsel_u64(ok, shifted, svdup_u64(0)));
    }
#else
    (void)args;
    (void)len;
#endif
}

#endif

void
npy_left_shift_sve_contig(char **args, npy_intp len, int itemsize, int is_signed)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    switch (itemsize) {
        case 1:
            if (is_signed) {
                npy_left_shift_contig_s8(args, len);
            }
            else {
                npy_left_shift_contig_u8(args, len);
            }
            return;
        case 2:
            if (is_signed) {
                npy_left_shift_contig_s16(args, len);
            }
            else {
                npy_left_shift_contig_u16(args, len);
            }
            return;
        case 4:
            if (is_signed) {
                npy_left_shift_contig_s32(args, len);
            }
            else {
                npy_left_shift_contig_u32(args, len);
            }
            return;
        case 8:
            if (is_signed) {
                npy_left_shift_contig_s64(args, len);
            }
            else {
                npy_left_shift_contig_u64(args, len);
            }
            return;
        default:
            return;
    }
#else
    (void)args;
    (void)len;
    (void)itemsize;
    (void)is_signed;
#endif
}

void
npy_right_shift_sve_contig(char **args, npy_intp len, int itemsize, int is_signed)
{
#if NPY_HAVE_ARM_SVE_INTRINSICS
    switch (itemsize) {
        case 1:
            if (is_signed) {
                npy_right_shift_contig_s8(args, len);
            }
            else {
                npy_right_shift_contig_u8(args, len);
            }
            return;
        case 2:
            if (is_signed) {
                npy_right_shift_contig_s16(args, len);
            }
            else {
                npy_right_shift_contig_u16(args, len);
            }
            return;
        case 4:
            if (is_signed) {
                npy_right_shift_contig_s32(args, len);
            }
            else {
                npy_right_shift_contig_u32(args, len);
            }
            return;
        case 8:
            if (is_signed) {
                npy_right_shift_contig_s64(args, len);
            }
            else {
                npy_right_shift_contig_u64(args, len);
            }
            return;
        default:
            return;
    }
#else
    (void)args;
    (void)len;
    (void)itemsize;
    (void)is_signed;
#endif
}
