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
 * Kunpeng 920 SVE gather bottleneck does not apply. The f16 wrapper
 * guards element alignment, overlap, zero output stride and the int64
 * lane-index bound. With stride 1 both branches use plain LoadU/StoreU
 * (LoadN/StoreN tails).
 */

/* Compile-time lane-index insert/extract for the full-vector strided case
 * on fixed-width targets (NEON: MaxLanes == Lanes). A runtime lane index
 * would make Highway's NEON InsertLane/ExtractLane spill the whole vector
 * to the stack per lane; constant indices lower to vset_lane / vget_lane.
 * SVE (MaxLanes > 16 here) uses a runtime loop instead: its lane access is
 * cheap predicate ops (cmpeq+sel / whilelt+lasta), no spill. */
template <int kLane, typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE vtype
exp_insert_lanes_full(vtype v, const type_t *src, npy_intp ssrc)
{
    v = hn::InsertLane(v, kLane, src[kLane * ssrc]);
    if constexpr (kLane + 1 < hn::MaxLanes(hn::DFromV<vtype>())) {
        return exp_insert_lanes_full<kLane + 1>(v, src, ssrc);
    }
    return v;
}
template <int kLane, typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE void
exp_extract_lanes_full(vtype v, type_t *dst, npy_intp sdst)
{
    dst[kLane * sdst] = hn::ExtractLane(v, kLane);
    if constexpr (kLane + 1 < hn::MaxLanes(hn::DFromV<vtype>())) {
        exp_extract_lanes_full<kLane + 1>(v, dst, sdst);
    }
}

template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE vtype
exp_load_vector(type_t *src, npy_intp ssrc, npy_intp len)
{
    auto D = hn::DFromV<vtype>();
    if constexpr (sizeof(type_t) == 2) {
        /* 16-bit lanes: SVE gather is 32/64-bit only, so strided runs
         * build the vector with scalar lane inserts (portable across
         * NEON and SVE; no gather hardware involved). */
        const int nlanes = hn::Lanes(D);
        if (ssrc == 1) {
            if (len < nlanes) return hn::LoadN(D, src, len);
            return hn::LoadU(D, src);
        }
        if (len >= nlanes) {
            if constexpr (hn::MaxLanes(D) <= 16) {
                return exp_insert_lanes_full<0>(hn::Zero(D), src, ssrc);
            } else {
                vtype v = hn::Zero(D);
                for (int i = 0; i < nlanes; i++) {
                    v = hn::InsertLane(v, i, src[i * ssrc]);
                }
                return v;
            }
        }
        vtype v = hn::Zero(D);
        for (int i = 0; i < (int)len; i++) {
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
        if (len >= nlanes) {
            if constexpr (hn::MaxLanes(D) <= 16) {
                exp_extract_lanes_full<0>(vec, dst, sdst);
            } else {
                for (int i = 0; i < nlanes; i++) {
                    dst[i * sdst] = hn::ExtractLane(vec, i);
                }
            }
            return;
        }
        for (int i = 0; i < (int)len; i++) {
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