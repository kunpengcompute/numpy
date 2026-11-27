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
 * 16-bit lanes: SVE gather is 32/64-bit only, so no gather hardware is
 * involved and the Kunpeng 920 SVE gather bottleneck does not apply.
 * Stride 1 uses plain LoadU/StoreU (LoadN/StoreN tails); other strided
 * runs scalar-gather/scatter the owned elements through a contiguous
 * stack temp + one ld1h/st1h (pipelineable memory ops, no per-lane FSU
 * splice chain). The staging loops use asm ldrh/strh so GCC does not
 * auto-vectorize them into a slow SVE gather.
 */

template <typename vtype, typename type_t>
HWY_ATTR NPY_FINLINE vtype
exp_load_vector(type_t *src, npy_intp ssrc, npy_intp len)
{
    auto D = hn::DFromV<vtype>();
    if constexpr (sizeof(type_t) == 2) {
        /* 16-bit lanes: SVE gather is 32/64-bit only (no gather hardware).
         * Stride 1 is contiguous; other strides stage through a stack temp
         * (scalar gather/scatter + one ld1h/st1h), tails use lane ops. */
        const int nlanes = hn::Lanes(D);
        if (ssrc == 1) {
            if (len < nlanes) return hn::LoadN(D, src, len);
            return hn::LoadU(D, src);
        }
        if (len >= nlanes) {
            /* General strided fallback: scalar-gather the owned elements
             * into a contiguous stack temp, then one contiguous vector
             * load. Replaces the per-lane InsertLane FSU chain (serial
             * ld1rh+cmpeq+dup+merge on Kunpeng 920) with pipelineable
             * scalar memory ops + ld1h. Reads only owned elements (no
             * overread). The staging loop
             * is kept scalar (asm ldrh) so GCC does not auto-vectorize it
             * into a slow SVE gather. */
            HWY_ALIGN uint16_t temp[hn::MaxLanes(D)];
            const uint16_t *s = reinterpret_cast<const uint16_t *>(src);
            for (int i = 0; i < nlanes; i++) {
                uint16_t v;
                __asm__("ldrh %w0, [%1]" : "=r"(v) : "r"(s));
                temp[i] = v;
                s += ssrc;
            }
            return hn::LoadU(D, reinterpret_cast<const hwy::float16_t *>(temp));
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
        /* General strided fallback: one contiguous vector store to a stack
         * temp, then scalar-scatter the owned elements to dst. Replaces the
         * per-lane ExtractLane (whilelo+lasta+str) loop with st1h +
         * pipelineable scalar memory ops. Writes only owned elements. The
         * scatter loop is kept scalar (asm strh) to avoid auto-vectorization
         * into a slow SVE scatter. */
        if (len >= nlanes) {
            HWY_ALIGN uint16_t temp[hn::MaxLanes(D)];
            hn::StoreU(vec, D, reinterpret_cast<hwy::float16_t *>(temp));
            uint16_t *d = reinterpret_cast<uint16_t *>(dst);
            for (int i = 0; i < nlanes; i++) {
                uint16_t v = temp[i];
                __asm__ volatile("strh %w0, [%1]"
                                 : : "r"(v), "r"(d) : "memory");
                d += sdst;
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