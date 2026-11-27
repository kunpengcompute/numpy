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
             * overread). temp is float16_t (repo staging convention, see
             * load_f16 in loops_log2); scalar bits cross via aliasing-safe
             * BitCastScalar, not a type pun. */
            HWY_ALIGN hwy::float16_t temp[hn::MaxLanes(D)];
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
            /* SVE target: keep the gather scalar via asm ldrh so GCC does
             * not auto-vectorize it into a slow microcoded SVE gather
             * (Kunpeng 920 SVE1). The "m" input operand declares the memory
             * read without a full "memory" barrier (a barrier serializes the
             * loop and costs ~6% on strided special); the wrapper guarantees
             * src/dst do not overlap and src is not written in this loop. */
            const uint16_t *s = reinterpret_cast<const uint16_t *>(src);
            for (int i = 0; i < nlanes; i++) {
                uint16_t v;
                __asm__("ldrh %w0, [%1]"
                        : "=r"(v) : "r"(s), "m"(*(const uint16_t *)s));
                temp[i] = hwy::BitCastScalar<hwy::float16_t>(v);
                s += ssrc;
            }
#else
            /* Portable C scalar staging (non-SVE / non-AArch64): no SVE
             * gather auto-vectorization risk, plain element copies. */
            for (int i = 0; i < nlanes; i++) {
                temp[i] = src[i * ssrc];
            }
#endif
            return hn::LoadU(D, temp);
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
         * pipelineable scalar memory ops. Writes only owned elements. temp
         * is float16_t (repo staging convention); scalar bits cross via
         * aliasing-safe BitCastScalar, not a type pun. */
        if (len >= nlanes) {
            HWY_ALIGN hwy::float16_t temp[hn::MaxLanes(D)];
            hn::StoreU(vec, D, temp);
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
            /* SVE target: keep the scatter scalar via asm strh (volatile +
             * "memory" so the stores are not reordered/eliminated) to avoid
             * auto-vectorization into a slow SVE scatter. */
            uint16_t *d = reinterpret_cast<uint16_t *>(dst);
            for (int i = 0; i < nlanes; i++) {
                uint16_t v = hwy::BitCastScalar<uint16_t>(temp[i]);
                __asm__ volatile("strh %w0, [%1]"
                                 : : "r"(v), "r"(d) : "memory");
                d += sdst;
            }
#else
            /* Portable C scalar staging (non-SVE / non-AArch64). */
            for (int i = 0; i < nlanes; i++) {
                dst[i * sdst] = temp[i];
            }
#endif
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