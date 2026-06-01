#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "loops_logical_and_sve.h"
#include "loops_sve_utils.h"

#include <string.h>

int
npy_logical_and_sve_scalar_available(void)
{
    return npy_sve_intrinsics_available();
}

#if NPY_HAVE_ARM_SVE_INTRINSICS

static inline int
npy_scalar_nonzero(const char *src, int itemsize)
{
    for (int i = 0; i < itemsize; ++i) {
        if (src[i] != 0) {
            return 1;
        }
    }
    return 0;
}

static inline NPY_SVE_TARGET void
npy_store_bool_u32(svbool_t pg, npy_bool *out, svbool_t pred_true)
{
    const svuint32_t ones = svdup_n_u32(1);
    const svuint32_t zeros = svdup_n_u32(0);
    svst1b_u32(pg, (uint8_t *)out, svsel_u32(pred_true, ones, zeros));
}

static inline NPY_SVE_TARGET void
npy_store_bool_u64(svbool_t pg, npy_bool *out, svbool_t pred_true)
{
    const svuint64_t ones = svdup_n_u64(1);
    const svuint64_t zeros = svdup_n_u64(0);
    svst1b_u64(pg, (uint8_t *)out, svsel_u64(pred_true, ones, zeros));
}

static NPY_SVE_TARGET void
npy_logical_truth_u32(const void *src, npy_bool *out, npy_intp n)
{
    const uint32_t *in = (const uint32_t *)src;
    npy_intp i = 0;

    for (; i < n; ) {
        svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)n);
        svuint32_t v = svld1_u32(pg, in + i);
        npy_store_bool_u32(pg, out + i, svcmpne_n_u32(pg, v, 0));
        i += (npy_intp)svcntw();
    }
}

static NPY_SVE_TARGET void
npy_logical_truth_u64(const void *src, npy_bool *out, npy_intp n)
{
    const uint64_t *in = (const uint64_t *)src;
    npy_intp i = 0;

    for (; i < n; ) {
        svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)n);
        svuint64_t v = svld1_u64(pg, in + i);
        npy_store_bool_u64(pg, out + i, svcmpne_n_u64(pg, v, 0));
        i += (npy_intp)svcntd();
    }
}

#endif

void
npy_logical_and_sve_scalar_in0(
        char **args, npy_intp const *dimensions, npy_intp const *steps,
        int itemsize)
{
    (void)steps;

#if NPY_HAVE_ARM_SVE_INTRINSICS
    const npy_intp n = dimensions[0];
    npy_bool *out = (npy_bool *)args[2];

    if (!npy_scalar_nonzero(args[0], itemsize)) {
        memset(out, 0, (size_t)n);
        return;
    }

    switch (itemsize) {
        case 4:
            npy_logical_truth_u32(args[1], out, n);
            return;
        case 8:
            npy_logical_truth_u64(args[1], out, n);
            return;
        default:
            return;
    }
#else
    (void)args;
    (void)dimensions;
    (void)itemsize;
#endif
}

void
npy_logical_and_sve_scalar_in1(
        char **args, npy_intp const *dimensions, npy_intp const *steps,
        int itemsize)
{
    char *tmp = args[0];

    args[0] = args[1];
    args[1] = tmp;
    npy_logical_and_sve_scalar_in0(args, dimensions, steps, itemsize);
    tmp = args[0];
    args[0] = args[1];
    args[1] = tmp;
}
