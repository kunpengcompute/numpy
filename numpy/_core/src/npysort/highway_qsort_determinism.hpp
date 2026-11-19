#ifndef NUMPY_CORE_SRC_NPYSORT_HIGHWAY_QSORT_DETERMINISM_HPP
#define NUMPY_CORE_SRC_NPYSORT_HIGHWAY_QSORT_DETERMINISM_HPP

/*
 * Deterministic arg-sort / arg-select on top of Highway vqsort.
 *
 * vqsort draws its pivot candidates from positions produced by a
 * thread-local Xorshift128+ generator (hwy::detail::GetGeneratorStateStatic).
 * In the VQSORT_ONLY_STATIC mode used by NumPy the generator is seeded from
 * stack/code addresses and clock(), and its state advances from call to
 * call.  For value sorts that is invisible: equal keys are interchangeable,
 * so the sorted output is identical either way.  Arg-sort and arg-select,
 * however, emit the *order of equal keys* as their result, and vqsort only
 * compares keys, never values (TraitsKV in hwy/contrib/sort/traits-inl.h),
 * so the permutation would change from call to call.  NumPy guarantees that
 * repeated sorts of the same input return identical results (the built-in
 * introsort, and x86-simd-sort on x86, are both deterministic).
 *
 * Two inputs to pivot selection must be pinned down for reproducibility:
 *
 *   1. the generator stream: reset it to fixed constants before every
 *      arg-sort/arg-select call (ResetVQSortRNG below);
 *   2. the sampling grid: DrawSamples() skips up to one 64-byte chunk so
 *      that sampling starts at a 64-byte-aligned address, so the key/value
 *      buffer passed to VQSortStatic/VQSelectStatic must itself have a
 *      64-byte aligned base (AlignedPairs below; a plain std::vector only
 *      guarantees 8/16-byte alignment).
 *
 * With both fixed, the pivot sequence - and therefore the resulting
 * permutation - is a pure function of (input values, array length).  It
 * still is *not* guaranteed to match the built-in introsort, nor to be
 * stable, and it may differ between vector lengths (SVE 128/256-bit) or
 * Highway versions.
 *
 * Value sort/partition (QSort/QSelect) deliberately keep the entropy-seeded
 * generator: their output does not depend on the sampling positions.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hwy/highway.h"
/*
 * Match the integration mode of the NumPy sort dispatchers.  Including
 * vqsort-inl.h is a no-op when the including translation unit already did
 * so (normal include guard for the target-independent prelude).
 */
#ifndef VQSORT_ONLY_STATIC
#define VQSORT_ONLY_STATIC 1
#endif
#include "hwy/contrib/sort/vqsort-inl.h"

namespace np { namespace highway { namespace qsort_simd {

/*
 * Re-seed the thread-local vqsort generator with fixed, non-zero constants.
 * O(1); the first call also performs the generator's lazy initialization,
 * after which state[2] stays non-zero and is never re-seeded from entropy.
 *
 * NOTE: all three words must be reset. hwy::RandomBits (shared-inl.h)
 * derives its output from state[0] ^ (state[2] + 1) and writes all three
 * words back, i.e. state[2] doubles as a monotonically advancing counter
 * that is part of the random stream. Leaving it untouched would keep
 * consecutive arg-sort calls nondeterministic (only the very first call
 * per process would match). It must be reset to 1, not 0: zero means
 * "not yet initialized" to GetGeneratorStateStatic and would trigger
 * re-seeding from entropy.
 *
 * hwy::detail is a private namespace: this relies on the exact state
 * layout verified against the vendored Highway submodule at commit
 * ee36c837129310be19c17c9108c6dc3f6ae06942 (uint64_t state[3] via
 * GetGeneratorStateStatic(), state[2]==0 as the uninitialized sentinel,
 * xorshift output in shared-inl.h RandomBits). Re-validate this
 * sub-contract whenever the Highway submodule is upgraded; if Highway
 * ever grows a public fixed-seed / set-state API, prefer it.
 */
inline void ResetVQSortRNG()
{
    uint64_t* state = hwy::detail::GetGeneratorStateStatic();
    state[0] = 0x9E3779B97F4A7C15ULL;
    state[1] = 0xBF58476D1CE4E5B9ULL;
    state[2] = 1;
}

/*
 * RAII storage for key/value pairs whose data() is aligned to kChunkBytes
 * (64), the chunk size DrawSamples() aligns its pivot-sampling grid to.
 * The padding covers the round-up distance for any KV element size:
 * data() may consume up to kChunkBytes-1 bytes, so we need
 * ceil((kChunkBytes-1)/sizeof(KV)) extra elements; the expression below
 * is that bound plus one, exact for the current instantiations.
 */
template <typename KV>
class AlignedPairs {
  public:
    static constexpr size_t kChunkBytes = 64;

    explicit AlignedPairs(size_t count)
        : storage_(count + (kChunkBytes - 1) / sizeof(KV) + 1) {}

    AlignedPairs(const AlignedPairs&) = delete;
    AlignedPairs& operator=(const AlignedPairs&) = delete;

    KV* data()
    {
        return reinterpret_cast<KV*>(
                (reinterpret_cast<uintptr_t>(storage_.data()) +
                 kChunkBytes - 1) &
                ~(static_cast<uintptr_t>(kChunkBytes) - 1));
    }

  private:
    std::vector<KV> storage_;
};

}}}  // namespace np::highway::qsort_simd

#endif  /* NUMPY_CORE_SRC_NPYSORT_HIGHWAY_QSORT_DETERMINISM_HPP */
