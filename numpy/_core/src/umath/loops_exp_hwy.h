#ifndef _NPY_UMATH_LOOPS_EXP_HWY_H_
#define _NPY_UMATH_LOOPS_EXP_HWY_H_
/*
 * Shared Highway load/store helpers for the exp dispatch kernels.
 *
 * Included inside the SIMD_ARM section of loops_exp.dispatch.cpp (f64)
 * and loops_exp_half.dispatch.cpp (f16) after
 * `namespace hn = hwy::HWY_NAMESPACE;` is defined.
 *
 * 32/64-bit lanes use Highway gather/scatter for strided runs (the f64
 * wrapper guards element alignment, overlap and the int64 index bound);
 * 16-bit lanes fall back to scalar lane insert/extract because SVE gather
 * is 32/64-bit only. With stride 1 both branches use plain LoadU/StoreU
 * (LoadN/StoreN tails).
 */
template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE vtype
exp_load_vector(type_t *src, npy_intp ssrc, npy_intp len)
{
    auto D = hn::DFromV<vtype>();
    if constexpr (sizeof(type_t) == 2) {
        /* 16-bit lanes: SVE gather is 32/64-bit only, so build the vector
         * with scalar inserts (portable across NEON and SVE). */
        const int nlanes = hn::Lanes(D);
        if (ssrc == 1) {
            if (len < nlanes) return hn::LoadN(D, src, len);
            return hn::LoadU(D, src);
        }
        vtype v = hn::Zero(D);
        const int n = (len < nlanes) ? static_cast<int>(len) : nlanes;
        for (int i = 0; i < n; i++) v = hn::InsertLane(v, i, src[i * ssrc]);
        return v;
    } else {
        using DI = hn::RebindToSigned<decltype(D)>;
        DI di;
        auto indices = hn::Mul(hn::Iota(di, 0), hn::Set(di, ssrc));
        if (len < hn::Lanes(D)) {
            if (ssrc == 1) return hn::LoadN(D, src, len);
            return hn::GatherIndexN(D, src, indices, len);
        }
        if (ssrc == 1) return hn::LoadU(D, src);
        return hn::GatherIndex(D, src, indices);
    }
}

template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE void
exp_store_vector(vtype vec, type_t *dst, npy_intp sdst, npy_intp len)
{
    auto D = hn::DFromV<vtype>();
    if constexpr (sizeof(type_t) == 2) {
        const int nlanes = hn::Lanes(D);
        if (sdst == 1) {
            if (len < nlanes) { hn::StoreN(vec, D, dst, len); return; }
            hn::StoreU(vec, D, dst);
            return;
        }
        const int n = (len < nlanes) ? static_cast<int>(len) : nlanes;
        for (int i = 0; i < n; i++) dst[i * sdst] = hn::ExtractLane(vec, i);
    } else {
        using DI = hn::RebindToSigned<decltype(D)>;
        DI di;
        auto indices = hn::Mul(hn::Iota(di, 0), hn::Set(di, sdst));
        if (len < hn::Lanes(D)) {
            if (sdst == 1) { hn::StoreN(vec, D, dst, len); return; }
            hn::ScatterIndexN(vec, D, dst, indices, len);
            return;
        }
        if (sdst == 1) { hn::StoreU(vec, D, dst); return; }
        hn::ScatterIndex(vec, D, dst, indices);
    }
}
#endif // _NPY_UMATH_LOOPS_EXP_HWY_H_