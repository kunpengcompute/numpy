#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <cstdint>
#include <type_traits>

#include <numpy/ndarraytypes.h>

#include "npy_cpu_dispatch.h"
#include "cast_hwy.dispatch.h"

#include <hwy/highway.h>

#include "cast_hwy.h"


HWY_BEFORE_NAMESPACE();

namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;


template <typename Src, typename Dst>
struct CastTraits
{
    using Primary = std::conditional_t<(sizeof(Src) >= sizeof(Dst)), Src, Dst>;
    using PrimaryTag = hn::ScalableTag<Primary>;
    using SrcTag = hn::Rebind<Src, PrimaryTag>;
    using DstTag = hn::Rebind<Dst, PrimaryTag>;
    using SrcVec = hn::Vec<SrcTag>;
    using DstVec = hn::Vec<DstTag>;
    using SrcByteTag = hn::Repartition<uint8_t, SrcTag>;
    using DstByteTag = hn::Repartition<uint8_t, DstTag>;

    static HWY_INLINE npy_intp
    Lanes()
    {
        return static_cast<npy_intp>(hn::Lanes(SrcTag()));
    }

    static HWY_INLINE SrcVec
    Load(const char *src)
    {
        const SrcByteTag bytes;
        return hn::BitCast(SrcTag(), hn::LoadU(
                bytes, reinterpret_cast<const uint8_t *>(src)));
    }

    static HWY_INLINE SrcVec
    LoadN(const char *src, npy_intp count)
    {
        const SrcByteTag bytes;
        return hn::BitCast(SrcTag(), hn::LoadN(
                bytes, reinterpret_cast<const uint8_t *>(src),
                static_cast<size_t>(count) * sizeof(Src)));
    }

    static HWY_INLINE void
    Store(DstVec value, char *dst)
    {
        const DstByteTag bytes;
        hn::StoreU(hn::BitCast(bytes, value), bytes,
                reinterpret_cast<uint8_t *>(dst));
    }

    static HWY_INLINE void
    StoreN(DstVec value, char *dst, npy_intp count)
    {
        const DstByteTag bytes;
        hn::StoreN(hn::BitCast(bytes, value), bytes,
                reinterpret_cast<uint8_t *>(dst),
                static_cast<size_t>(count) * sizeof(Dst));
    }

    static HWY_INLINE DstVec
    Convert(SrcVec value)
    {
        const DstTag dst_tag;

        if constexpr (std::is_integral_v<Src> &&
                      std::is_integral_v<Dst>) {
            if constexpr (sizeof(Src) < sizeof(Dst)) {
                return hn::PromoteTo(dst_tag, value);
            }
            else {
                using UnsignedSrc = std::make_unsigned_t<Src>;
                using UnsignedDst = std::make_unsigned_t<Dst>;
                const hn::Rebind<UnsignedSrc, PrimaryTag> unsigned_src_tag;
                const hn::Rebind<UnsignedDst, PrimaryTag> unsigned_dst_tag;
                auto unsigned_value = hn::BitCast(unsigned_src_tag, value);
                auto truncated = hn::TruncateTo(
                        unsigned_dst_tag, unsigned_value);
                return hn::BitCast(dst_tag, truncated);
            }
        }
        else if constexpr (std::is_integral_v<Src> &&
                           std::is_floating_point_v<Dst>) {
            if constexpr (sizeof(Src) == sizeof(Dst)) {
                return hn::ConvertTo(dst_tag, value);
            }
            else if constexpr (sizeof(Src) == 4 && sizeof(Dst) == 8) {
                return hn::PromoteTo(dst_tag, value);
            }
            else {
                using WideInt = std::conditional_t<
                        sizeof(Dst) == 4, int32_t, int64_t>;
                const hn::Rebind<WideInt, PrimaryTag> wide_int_tag;
                return hn::ConvertTo(
                        dst_tag, hn::PromoteTo(wide_int_tag, value));
            }
        }
        else {
            return hn::PromoteTo(dst_tag, value);
        }
    }
};


template <typename Traits, int Row, int Rows>
HWY_INLINE void
CastFullRows(const char *src, npy_intp src_outer_stride,
        char *dst, npy_intp dst_outer_stride, npy_intp offset)
{
    using Src = hn::TFromD<typename Traits::SrcTag>;
    using Dst = hn::TFromD<typename Traits::DstTag>;

    auto value = Traits::Load(
            src + Row * src_outer_stride + offset * sizeof(Src));
    Traits::Store(Traits::Convert(value),
            dst + Row * dst_outer_stride + offset * sizeof(Dst));
    if constexpr (Row + 1 < Rows) {
        CastFullRows<Traits, Row + 1, Rows>(
                src, src_outer_stride, dst, dst_outer_stride, offset);
    }
}


template <typename Traits, int Row, int Rows>
HWY_INLINE void
CastTailRows(const char *src, npy_intp src_outer_stride,
        char *dst, npy_intp dst_outer_stride,
        npy_intp offset, npy_intp count)
{
    using Src = hn::TFromD<typename Traits::SrcTag>;
    using Dst = hn::TFromD<typename Traits::DstTag>;

    auto value = Traits::LoadN(
            src + Row * src_outer_stride + offset * sizeof(Src), count);
    Traits::StoreN(Traits::Convert(value),
            dst + Row * dst_outer_stride + offset * sizeof(Dst), count);
    if constexpr (Row + 1 < Rows) {
        CastTailRows<Traits, Row + 1, Rows>(
                src, src_outer_stride, dst, dst_outer_stride, offset, count);
    }
}


template <typename Src, typename Dst, int Rows>
HWY_INLINE void
CastRowBatch(const char *src, npy_intp src_outer_stride,
        char *dst, npy_intp dst_outer_stride, npy_intp inner_count)
{
    using Traits = CastTraits<Src, Dst>;
    const npy_intp lanes = Traits::Lanes();
    npy_intp offset = 0;

    for (; offset + lanes <= inner_count; offset += lanes) {
        CastFullRows<Traits, 0, Rows>(
                src, src_outer_stride, dst, dst_outer_stride, offset);
    }
    if (offset < inner_count) {
        CastTailRows<Traits, 0, Rows>(
                src, src_outer_stride, dst, dst_outer_stride,
                offset, inner_count - offset);
    }
}


template <typename D>
HWY_INLINE void
StoreBytes(D tag, hn::Vec<D> value, char *dst)
{
    const hn::Repartition<uint8_t, D> bytes;
    hn::StoreU(hn::BitCast(bytes, value), bytes,
            reinterpret_cast<uint8_t *>(dst));
}


template <typename Current, typename Dst, typename CurrentTag>
HWY_INLINE void
WidenConvertStore(CurrentTag current_tag, hn::Vec<CurrentTag> value, char *dst)
{
    if constexpr (sizeof(Current) == sizeof(Dst)) {
        if constexpr (std::is_same_v<Current, Dst>) {
            StoreBytes(current_tag, value, dst);
        }
        else {
            const hn::Rebind<Dst, CurrentTag> dst_tag;
            StoreBytes(dst_tag, hn::ConvertTo(dst_tag, value), dst);
        }
    }
    else {
        using Next = std::conditional_t<
                std::is_same_v<Current, int16_t>, int32_t, Dst>;
        const hn::Repartition<Next, CurrentTag> next_tag;

        WidenConvertStore<Next, Dst>(
                next_tag, hn::PromoteLowerTo(next_tag, value), dst);
        WidenConvertStore<Next, Dst>(
                next_tag, hn::PromoteUpperTo(next_tag, value),
                dst + hn::Lanes(next_tag) * sizeof(Dst));
    }
}


template <typename Src, typename Dst, int Row, int Rows>
HWY_INLINE void
CastWidenFullRows(const char *src, npy_intp src_outer_stride,
        char *dst, npy_intp dst_outer_stride, npy_intp offset)
{
    using SrcTag = hn::ScalableTag<Src>;
    const SrcTag src_tag;
    const hn::Repartition<uint8_t, SrcTag> src_bytes;

    auto value = hn::BitCast(src_tag, hn::LoadU(
            src_bytes, reinterpret_cast<const uint8_t *>(
                    src + Row * src_outer_stride + offset * sizeof(Src))));
    WidenConvertStore<Src, Dst>(src_tag, value,
            dst + Row * dst_outer_stride + offset * sizeof(Dst));

    if constexpr (Row + 1 < Rows) {
        CastWidenFullRows<Src, Dst, Row + 1, Rows>(
                src, src_outer_stride, dst, dst_outer_stride, offset);
    }
}


template <typename Src, typename Dst, int Rows>
HWY_INLINE void
CastWidenRowBatch(const char *src, npy_intp src_outer_stride,
        char *dst, npy_intp dst_outer_stride, npy_intp inner_count)
{
    const hn::ScalableTag<Src> src_tag;
    const npy_intp lanes = static_cast<npy_intp>(hn::Lanes(src_tag));
    npy_intp offset = 0;

    for (; offset + lanes <= inner_count; offset += lanes) {
        CastWidenFullRows<Src, Dst, 0, Rows>(
                src, src_outer_stride, dst, dst_outer_stride, offset);
    }
    if (offset < inner_count) {
        CastRowBatch<Src, Dst, Rows>(
                src + offset * sizeof(Src), src_outer_stride,
                dst + offset * sizeof(Dst), dst_outer_stride,
                inner_count - offset);
    }
}


template <typename Src, typename Dst, int Rows>
HWY_INLINE void
CastBestRowBatch(const char *src, npy_intp src_outer_stride,
        char *dst, npy_intp dst_outer_stride, npy_intp inner_count)
{
    if constexpr (sizeof(Src) < sizeof(Dst)) {
        CastWidenRowBatch<Src, Dst, Rows>(
                src, src_outer_stride, dst, dst_outer_stride, inner_count);
    }
    else {
        CastRowBatch<Src, Dst, Rows>(
                src, src_outer_stride, dst, dst_outer_stride, inner_count);
    }
}


template <typename Src, typename Dst, int Rows>
HWY_ATTR int
Cast2D(const char *src, npy_intp src_outer_stride,
        char *dst, npy_intp dst_outer_stride,
        npy_intp inner_count, npy_intp outer_count)
{
    npy_intp row = 0;
    for (; row + Rows <= outer_count; row += Rows) {
        CastBestRowBatch<Src, Dst, Rows>(
                src + row * src_outer_stride, src_outer_stride,
                dst + row * dst_outer_stride, dst_outer_stride, inner_count);
    }
    for (; row < outer_count; ++row) {
        CastBestRowBatch<Src, Dst, 1>(
                src + row * src_outer_stride, src_outer_stride,
                dst + row * dst_outer_stride, dst_outer_stride, inner_count);
    }
    return 0;
}

}  // namespace HWY_NAMESPACE

HWY_AFTER_NAMESPACE();


namespace {

enum class CastKind
{
    unsupported,
    int16_to_float,
    int16_to_double,
    int16_to_int32,
    int16_to_int64,
    int32_to_int16,
    int32_to_float,
    int32_to_double,
    int32_to_int64,
    int64_to_int16,
    int64_to_int32,
    int64_to_double,
    float_to_double,
    float_to_int64,
};


CastKind
GetCastKind(int src_type, int dst_type)
{
    switch (src_type) {
        case NPY_INT16:
            switch (dst_type) {
                case NPY_FLOAT: return CastKind::int16_to_float;
                case NPY_DOUBLE: return CastKind::int16_to_double;
                case NPY_INT32: return CastKind::int16_to_int32;
                case NPY_INT64: return CastKind::int16_to_int64;
            }
            break;
        case NPY_INT32:
            switch (dst_type) {
                case NPY_INT16: return CastKind::int32_to_int16;
                case NPY_FLOAT: return CastKind::int32_to_float;
                case NPY_DOUBLE: return CastKind::int32_to_double;
                case NPY_INT64: return CastKind::int32_to_int64;
            }
            break;
        case NPY_INT64:
            switch (dst_type) {
                case NPY_INT16: return CastKind::int64_to_int16;
                case NPY_INT32: return CastKind::int64_to_int32;
                case NPY_DOUBLE: return CastKind::int64_to_double;
            }
            break;
        case NPY_FLOAT:
            switch (dst_type) {
                case NPY_DOUBLE: return CastKind::float_to_double;
                case NPY_INT64: return CastKind::float_to_int64;
            }
            break;
    }
    return CastKind::unsupported;
}


template <typename Src, typename Dst, int Rows = 1>
int
RunCast(const char *src, npy_intp src_outer_stride,
        char *dst, npy_intp dst_outer_stride,
        npy_intp inner_count, npy_intp outer_count)
{
    return HWY_NAMESPACE::Cast2D<Src, Dst, Rows>(
            src, src_outer_stride, dst, dst_outer_stride,
            inner_count, outer_count);
}

}  // namespace


extern "C" NPY_VISIBILITY_HIDDEN int
NPY_CPU_DISPATCH_CURFX(npy_cast_hwy_supports)(int src_type, int dst_type)
{
    return GetCastKind(src_type, dst_type) != CastKind::unsupported;
}


extern "C" NPY_VISIBILITY_HIDDEN int
NPY_CPU_DISPATCH_CURFX(npy_cast_hwy_2d)(int src_type, int dst_type,
        const char *src, npy_intp src_outer_stride,
        char *dst, npy_intp dst_outer_stride,
        npy_intp inner_count, npy_intp outer_count)
{
    switch (GetCastKind(src_type, dst_type)) {
        case CastKind::int16_to_float:
            return RunCast<int16_t, float>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::int16_to_double:
            return RunCast<int16_t, double, 2>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::int16_to_int32:
            return RunCast<int16_t, int32_t, 2>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::int16_to_int64:
            return RunCast<int16_t, int64_t>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::int32_to_int16:
            return RunCast<int32_t, int16_t>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::int32_to_float:
            return RunCast<int32_t, float, 4>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::int32_to_double:
            return RunCast<int32_t, double, 2>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::int32_to_int64:
            return RunCast<int32_t, int64_t, 2>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::int64_to_int16:
            return RunCast<int64_t, int16_t>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::int64_to_int32:
            return RunCast<int64_t, int32_t>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::int64_to_double:
            return RunCast<int64_t, double, 2>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::float_to_double:
            return RunCast<float, double>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::float_to_int64:
            return RunCast<float, int64_t, 2>(src, src_outer_stride,
                    dst, dst_outer_stride, inner_count, outer_count);
        case CastKind::unsupported:
            return -1;
    }
    return -1;
}
