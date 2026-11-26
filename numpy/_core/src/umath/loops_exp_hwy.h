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
 * wrapper guards element alignment, overlap and zero output stride).
 * 16-bit lanes use scalar lane InsertLane/ExtractLane for strided runs:
 * SVE gather is 32/64-bit only, so no gather hardware is involved and the
 * Kunpeng 920 SVE gather bottleneck does not apply. f16 vectorization is
 * SVE-target only in this build (NPY_HWY_F16 needs ASIMDHP, absent from the
 * min baseline), where a runtime lane index lowers to cheap predicate ops
 * (InsertLane = cmpeq+sel, ExtractLane = whilelt+lasta), not a stack spill.
 * The f16 wrapper guards element alignment, overlap, zero output stride and
 * the int64 lane-index bound. With stride 1 both branches use plain
 * LoadU/StoreU (LoadN/StoreN tails).
 */
template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE vtype
exp_load_vector(type_t *src, npy_intp ssrc, npy_intp len)
{
    auto D = hn::DFromV<vtype>();
    if constexpr (sizeof(type_t) == 2) {
        /* 16-bit lanes: SVE gather is 32/64-bit only, so strided runs
         * build the vector with scalar lane inserts (no gather hardware). */
        const int nlanes = hn::Lanes(D);
        if (ssrc == 1) {
            if (len < nlanes) return hn::LoadN(D, src, len);
            return hn::LoadU(D, src);
        }
        vtype v = hn::Zero(D);
        const int n = (len < nlanes) ? static_cast<int>(len) : nlanes;
        for (int i = 0; i < n; i++) {
            v = hn::InsertLane(v, i, src[i * ssrc]);
        }
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
        /* Scalar lane extracts (see load); ascending lane order keeps the
         * write order deterministic. */
        const int n = (len < nlanes) ? static_cast<int>(len) : nlanes;
        for (int i = 0; i < n; i++) {
            dst[i * sdst] = hn::ExtractLane(vec, i);
        }
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