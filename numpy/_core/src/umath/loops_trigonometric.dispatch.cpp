#include "fast_loop_macros.h"
#include "loops.h"
#include "loops_utils.h"
#include "numpy/npy_math.h"

#include "simd/simd.h"
#include "simd/simd.hpp" 
#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>
#include <cstring>
#include <fenv.h>
#include "npy_svml.h"

namespace hn = hwy::HWY_NAMESPACE;

/*
 * Vectorized approximate sine/cosine algorithms: The following code is a
 * vectorized version of the algorithm presented here:
 * https://stackoverflow.com/questions/30463616/payne-hanek-algorithm-implementation-in-c/30465751#30465751
 * (1) Load data in registers and generate mask for elements that are within
 * range [-71476.0625f, 71476.0625f] for cosine and [-117435.992f, 117435.992f]
 * for sine.
 * (2) For elements within range, perform range reduction using
 * Cody-Waite's method: x* = x - y*PI/2, where y = rint(x*2/PI). x* \in [-PI/4,
 * PI/4].
 * (3) Map cos(x) to (+/-)sine or (+/-)cosine of x* based on the
 * quadrant k = int(y).
 * (4) For elements outside that range, Cody-Waite
 * reduction performs poorly leading to catastrophic cancellation. We compute
 * cosine by calling glibc in a scalar fashion.
 * (5) Vectorized implementation
 * has a max ULP of 1.49 and performs at least 5-7x(x86) - 2.5-3x(Power) -
 * 1-2x(Arm) faster than scalar implementations when magnitude of all elements
 * in the array < 71476.0625f (117435.992f for sine).  Worst case performance
 * is when all the elements are large leading to about 1-2% reduction in
 * performance.
 * TODO: use vectorized version of Payne-Hanek style reduction for large
 * elements or when there's no native FUSED support instead of fallback to libc
 */

#if NPY_SIMD_FMA3  // native support
typedef enum {
    SIMD_COMPUTE_SIN,
    SIMD_COMPUTE_COS
} SIMD_TRIG_OP;

const hn::ScalableTag<float> f32;
const hn::ScalableTag<int32_t> s32;
using vec_f32 = hn::Vec<decltype(f32)>;
using vec_s32 = hn::Vec<decltype(s32)>;
using opmask_t = hn::Mask<decltype(f32)>;

HWY_INLINE HWY_ATTR vec_f32
simd_range_reduction_f32(vec_f32 &x, vec_f32 &y, const vec_f32 &c1,
                         const vec_f32 &c2, const vec_f32 &c3)
{
    vec_f32 reduced_x = hn::MulAdd(y, c1, x);
    reduced_x = hn::MulAdd(y, c2, reduced_x);
    reduced_x = hn::MulAdd(y, c3, reduced_x);
    return reduced_x;
}

HWY_INLINE HWY_ATTR vec_f32
simd_cosine_poly_f32(vec_f32 &x2)
{
    const vec_f32 invf8 = hn::Set(f32, 0x1.98e616p-16f);
    const vec_f32 invf6 = hn::Set(f32, -0x1.6c06dcp-10f);
    const vec_f32 invf4 = hn::Set(f32, 0x1.55553cp-05f);
    const vec_f32 invf2 = hn::Set(f32, -0x1.000000p-01f);
    const vec_f32 invf0 = hn::Set(f32, 0x1.000000p+00f);

    vec_f32 r = hn::MulAdd(invf8, x2, invf6);
    r = hn::MulAdd(r, x2, invf4);
    r = hn::MulAdd(r, x2, invf2);
    r = hn::MulAdd(r, x2, invf0);
    return r;
}
/*
 * Approximate sine algorithm for x \in [-PI/4, PI/4]
 * Maximum ULP across all 32-bit floats = 0.647
 * Polynomial approximation based on unpublished work by T. Myklebust
 */
HWY_INLINE HWY_ATTR vec_f32
simd_sine_poly_f32(vec_f32 &x, vec_f32 &x2)
{
    const vec_f32 invf9 = hn::Set(f32, 0x1.7d3bbcp-19f);
    const vec_f32 invf7 = hn::Set(f32, -0x1.a06bbap-13f);
    const vec_f32 invf5 = hn::Set(f32, 0x1.11119ap-07f);
    const vec_f32 invf3 = hn::Set(f32, -0x1.555556p-03f);

    vec_f32 r = hn::MulAdd(invf9, x2, invf7);
    r = hn::MulAdd(r, x2, invf5);
    r = hn::MulAdd(r, x2, invf3);
    r = hn::MulAdd(r, x2, hn::Zero(f32));
    r = hn::MulAdd(r, x, x);
    return r;
}

static void HWY_ATTR SIMD_MSVC_NOINLINE
simd_sincos_f32(const float *src, npy_intp ssrc, float *dst, npy_intp sdst,
                npy_intp len, SIMD_TRIG_OP trig_op)
{
    // Load up frequently used constants
    const vec_f32 zerosf = hn::Zero(f32);
    const vec_s32 ones = hn::Set(s32, 1);
    const vec_s32 twos = hn::Set(s32, 2);
    const vec_f32 two_over_pi = hn::Set(f32, 0x1.45f306p-1f);
    const vec_f32 codyw_pio2_highf = hn::Set(f32, -0x1.921fb0p+00f);
    const vec_f32 codyw_pio2_medf = hn::Set(f32, -0x1.5110b4p-22f);
    const vec_f32 codyw_pio2_lowf = hn::Set(f32, -0x1.846988p-48f);
    const vec_f32 rint_cvt_magic = hn::Set(f32, 0x1.800000p+23f);
    // Cody-Waite's range
    float max_codi = 117435.992f;
    if (trig_op == SIMD_COMPUTE_COS) {
        max_codi = 71476.0625f;
    }
    const vec_f32 max_cody = hn::Set(f32, max_codi);

    const int lanes = hn::Lanes(f32);
    const vec_s32 src_index = hn::Mul(hn::Iota(s32, 0), hn::Set(s32, ssrc));
    const vec_s32 dst_index = hn::Mul(hn::Iota(s32, 0), hn::Set(s32, sdst));

    for (; len > 0; len -= lanes, src += ssrc * lanes, dst += sdst * lanes) {
        vec_f32 x_in;
        if (ssrc == 1) {
            x_in = hn::LoadN(f32, src, len);
        }
        else {
            x_in = hn::GatherIndexN(f32, src, src_index, len);
        }
        opmask_t nnan_mask = hn::Not(hn::IsNaN(x_in));
        // Eliminate NaN to avoid FP invalid exception
        x_in = hn::IfThenElse(nnan_mask, x_in, zerosf);
        opmask_t simd_mask = hn::Le(hn::Abs(x_in), max_cody);
        /*
         * For elements outside of this range, Cody-Waite's range reduction
         * becomes inaccurate and we will call libc to compute cosine for
         * these numbers
         */
        if (!hn::AllFalse(f32, simd_mask)) {
            vec_f32 x = hn::IfThenElse(hn::And(nnan_mask, simd_mask), x_in,
                                       zerosf);

            vec_f32 quadrant = hn::Mul(x, two_over_pi);
            // round to nearest, -0.0f -> +0.0f, and |a| must be <= 0x1.0p+22
            quadrant = hn::Add(quadrant, rint_cvt_magic);
            quadrant = hn::Sub(quadrant, rint_cvt_magic);

            // Cody-Waite's range reduction algorithm
            vec_f32 reduced_x =
                    simd_range_reduction_f32(x, quadrant, codyw_pio2_highf,
                                             codyw_pio2_medf, codyw_pio2_lowf);
            vec_f32 reduced_x2 = hn::Mul(reduced_x, reduced_x);

            // compute cosine and sine
            vec_f32 cos = simd_cosine_poly_f32(reduced_x2);
            vec_f32 sin = simd_sine_poly_f32(reduced_x, reduced_x2);

            vec_s32 iquadrant = hn::NearestInt(quadrant);
            if (trig_op == SIMD_COMPUTE_COS) {
                iquadrant = hn::Add(iquadrant, ones);
            }
            // blend sin and cos based on the quadrant
            opmask_t sine_mask = hn::RebindMask(
                    f32, hn::Eq(hn::And(iquadrant, ones), hn::Zero(s32)));
            cos = hn::IfThenElse(sine_mask, sin, cos);

            // multiply by -1 for appropriate elements
            opmask_t negate_mask = hn::RebindMask(
                    f32, hn::Eq(hn::And(iquadrant, twos), twos));
            cos = hn::MaskedSubOr(cos, negate_mask, zerosf, cos);
            cos = hn::IfThenElse(nnan_mask, cos, hn::Set(f32, NPY_NANF));

            if (sdst == 1) {
                hn::StoreN(cos, f32, dst, len);
            }
            else {
                hn::ScatterIndexN(cos, f32, dst, dst_index, len);
            }
        }
        if (!hn::AllTrue(f32, simd_mask)) {
            static_assert(hn::MaxLanes(f32) <= 64,
                          "The following fallback is not applicable for "
                          "SIMD widths larger than 2048 bits, or for scalable "
                          "SIMD in general.");
            npy_uint64 simd_maski;
            hn::StoreMaskBits(f32, simd_mask, (uint8_t *)&simd_maski);
#if HWY_IS_BIG_ENDIAN
            static_assert(hn::MaxLanes(f32) <= 8,
                          "This conversion is not supported for SIMD widths "
                          "larger than 256 bits.");
            simd_maski = ((uint8_t *)&simd_maski)[0];
#endif
            float NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) ip_fback[hn::MaxLanes(f32)];
            hn::Store(x_in, f32, ip_fback);

            // process elements using libc for large elements
            if (trig_op == SIMD_COMPUTE_COS) {
                for (unsigned i = 0; i < hn::Lanes(f32); ++i) {
                    if ((simd_maski >> i) & 1) {
                        continue;
                    }
                    dst[sdst * i] = npy_cosf(ip_fback[i]);
                }
            }
            else {
                for (unsigned i = 0; i < hn::Lanes(f32); ++i) {
                    if ((simd_maski >> i) & 1) {
                        continue;
                    }
                    dst[sdst * i] = npy_sinf(ip_fback[i]);
                }
            }
        }
        npyv_cleanup();
    }
}
#endif  // NPY_SIMD_FMA3

template <typename T>
struct TypeToInt {
    using Type = void;
};

template <>
struct TypeToInt<float> {
    using Type = int32_t;
};

template <>
struct TypeToInt<double> {
    using Type = int64_t;
};

/*
 * Vectorized sine/cosine for double on ARM platforms using Highway
 * Highway sin/cos have valid range [-39000, +39000] and ULP <= 3
 * NaN: replaced with 0 before CallSin/CallCos to avoid spurious FP exceptions
 * Inf/out-of-range: handled by Highway (produces NaN, triggers FP exception)
 * X86 platforms use scalar libm implementation
 */
#if defined(__aarch64__) && NPY_SIMD && NPY_SIMD_F64 && HWY_HAVE_FLOAT64

/* Keep this block behind the SIMD/F64 guards above.  The smoke-test build uses
 * -Dcpu-baseline=none, where NPY_SIMD_WIDTH can be 0; the overlap loops below
 * advance by NPY_SIMD_WIDTH / sizeof(npy_double) and would otherwise never
 * make progress. */

template <typename T>
struct OpSin {
static constexpr T c_inv_pi = 0x1.45f306dc9c883p-2;
static constexpr T c_pi_1 = 0x1.921fb54442d18p+1;
static constexpr T c_pi_2 = 0x1.1a62633145c06p-53;
static constexpr T c_pi_3 = 0x1.c1cd129024e09p-106;
static constexpr T c_range_val = 0x1p23;
static constexpr T c_tiny_threshold = 1e-100;
static constexpr T c_pi_threshold = 1e-10;
static constexpr T c_poly0 = -0x1.555555555547bp-3;
static constexpr T c_poly1 = 0x1.1111111108a4dp-7;
static constexpr T c_poly2 = -0x1.a01a019936f27p-13;
static constexpr T c_poly3 = 0x1.71de37a97d93ep-19;
static constexpr T c_poly4 = -0x1.ae633919987c6p-26;
static constexpr T c_poly5 = 0x1.60e277ae07cecp-33;
static constexpr T c_poly6 = -0x1.9e9540300a1p-41;

static inline bool has_special_value(T val) {
    uint64_t bits;
    memcpy(&bits, &val, sizeof(T));
    uint64_t exp = (bits >> 52) & 0x7FF;
    return exp == 0x7FF;
}

static inline bool needs_scalar_path(T val) {
    uint64_t bits;
    memcpy(&bits, &val, sizeof(T));
    uint64_t exp = (bits >> 52) & 0x7FF;
    if (exp == 0x7FF) return true;
    if (exp > 23 + 1023) return true;
    T abs_val = std::fabs(val);
    if (abs_val < c_tiny_threshold) return false;
    T k = std::round(abs_val * c_inv_pi);
    T k_pi = k * c_pi_1;
    T delta = std::fabs(abs_val - k_pi);
    return delta < c_pi_threshold;
}

static inline T scalar_call(T val) {
    return std::isnan(val) ? NPY_NAN : npy_sin(val);
}

template <typename V, typename = std::enable_if_t<np::simd::kSupportLane<T>>>
inline V operator()(const V &x) const
{
    const hn::ScalableTag<T> d;
    const hn::Rebind<uint64_t, decltype(d)> du64;
    const hn::Rebind<int64_t, decltype(d)> di64;
    
    const auto inv_pi = hn::Set(d, c_inv_pi);
    const auto pi_1 = hn::Set(d, c_pi_1);
    const auto pi_2 = hn::Set(d, c_pi_2);
    const auto pi_3 = hn::Set(d, c_pi_3);
    const auto range_val = hn::Set(d, c_range_val);
    const auto tiny_threshold = hn::Set(d, c_tiny_threshold);
    const auto pi_threshold = hn::Set(d, c_pi_threshold);
    const auto zeros = hn::Zero(d);
    const auto nan_val = hn::Set(d, NPY_NAN);
    
    auto is_nan = hn::IsNaN(x);
    auto is_inf = hn::IsInf(x);
    auto x_clean = hn::IfThenElse(is_nan, zeros, x);
    
    auto abs_x = hn::Abs(x_clean);
    auto is_tiny = hn::Lt(abs_x, tiny_threshold);
    auto is_large = hn::Gt(abs_x, range_val);
    
    if (__builtin_expect(hn::AllTrue(d, hn::Or(is_nan, is_tiny)), 0)) {
        return hn::IfThenElse(is_nan, nan_val, x);
    }
    
    auto k = hn::Round(hn::Mul(abs_x, inv_pi));
    auto k_pi = hn::Mul(k, pi_1);
    auto delta = hn::Abs(hn::Sub(abs_x, k_pi));
    auto is_boundary = hn::Lt(delta, pi_threshold);
    
    auto needs_scalar = hn::Or(is_large, is_boundary);
    needs_scalar = hn::Or(needs_scalar, is_inf);
    
    if (__builtin_expect(!hn::AllFalse(d, needs_scalar), 0)) {
        T NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) input_arr[hn::MaxLanes(d)];
        T NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) result_arr[hn::MaxLanes(d)];
        hn::Store(x, d, input_arr);
        
        size_t lanes_count = hn::Lanes(d);
        for (size_t i = 0; i < lanes_count; i++) {
            T val = input_arr[i];
            result_arr[i] = std::isnan(val) ? NPY_NAN : npy_sin(val);
        }
        
        return hn::Load(d, result_arr);
    }
    
    auto n = hn::Round(hn::Mul(x_clean, inv_pi));
    auto n_int = hn::ConvertTo(di64, n);
    auto odd = hn::ShiftLeft<63>(hn::BitCast(du64, n_int));
    
    auto r = hn::NegMulAdd(n, pi_1, x_clean);
    r = hn::NegMulAdd(n, pi_2, r);
    r = hn::NegMulAdd(n, pi_3, r);
    
    auto r2 = hn::Mul(r, r);
    auto r3 = hn::Mul(r2, r);
    auto r4 = hn::Mul(r2, r2);
    
    const auto c0 = hn::Set(d, c_poly0);
    const auto c1 = hn::Set(d, c_poly1);
    const auto c2 = hn::Set(d, c_poly2);
    const auto c3 = hn::Set(d, c_poly3);
    const auto c4 = hn::Set(d, c_poly4);
    const auto c5 = hn::Set(d, c_poly5);
    const auto c6 = hn::Set(d, c_poly6);
    
    auto p01 = hn::MulAdd(r2, c1, c0);
    auto p23 = hn::MulAdd(r2, c3, c2);
    auto p45 = hn::MulAdd(r2, c5, c4);
    auto p46 = hn::MulAdd(r4, c6, p45);
    
    auto p26 = hn::MulAdd(r4, p46, p23);
    auto p06 = hn::MulAdd(r4, p26, p01);
    auto y = hn::MulAdd(r3, p06, r);
    
    auto simd_result = hn::BitCast(d, hn::Xor(hn::BitCast(du64, y), odd));
    simd_result = hn::IfThenElse(is_tiny, x, simd_result);
    simd_result = hn::IfThenElse(is_nan, nan_val, simd_result);
    
    return simd_result;
}
};

template <typename T>
struct OpCos {
static constexpr T c_inv_pi = 0x1.45f306dc9c883p-2;
static constexpr T c_pi_1 = 0x1.921fb54442d18p+1;
static constexpr T c_pi_2 = 0x1.1a62633145c06p-53;
static constexpr T c_pi_3 = 0x1.c1cd129024e09p-106;
static constexpr T c_range_val = 0x1p23;
static constexpr T c_tiny_threshold = 1e-100;
static constexpr T c_pi_threshold = 1e-10;
static constexpr T c_half = 0.5;
static constexpr T c_one = 1.0;
static constexpr T c_poly0 = -0x1.555555555547bp-3;
static constexpr T c_poly1 = 0x1.1111111108a4dp-7;
static constexpr T c_poly2 = -0x1.a01a019936f27p-13;
static constexpr T c_poly3 = 0x1.71de37a97d93ep-19;
static constexpr T c_poly4 = -0x1.ae633919987c6p-26;
static constexpr T c_poly5 = 0x1.60e277ae07cecp-33;
static constexpr T c_poly6 = -0x1.9e9540300a1p-41;

static inline bool has_special_value(T val) {
    uint64_t bits;
    memcpy(&bits, &val, sizeof(T));
    uint64_t exp = (bits >> 52) & 0x7FF;
    return exp == 0x7FF;
}

static inline bool needs_scalar_path(T val) {
    uint64_t bits;
    memcpy(&bits, &val, sizeof(T));
    uint64_t exp = (bits >> 52) & 0x7FF;
    if (exp == 0x7FF) return true;
    if (exp > 23 + 1023) return true;
    T abs_val = std::fabs(val);
    if (abs_val < c_tiny_threshold) return false;
    T k = std::round(abs_val * c_inv_pi);
    T k_pi = k * c_pi_1;
    T delta = std::fabs(abs_val - k_pi);
    return delta < c_pi_threshold;
}

static inline T scalar_call(T val) {
    return std::isnan(val) ? NPY_NAN : npy_cos(val);
}

template <typename V, typename = std::enable_if_t<np::simd::kSupportLane<T>>>
inline V operator()(const V &x) const
{
    const hn::ScalableTag<T> d;
    const hn::Rebind<uint64_t, decltype(d)> du64;
    const hn::Rebind<int64_t, decltype(d)> di64;
    
    const auto inv_pi = hn::Set(d, c_inv_pi);
    const auto pi_1 = hn::Set(d, c_pi_1);
    const auto pi_2 = hn::Set(d, c_pi_2);
    const auto pi_3 = hn::Set(d, c_pi_3);
    const auto range_val = hn::Set(d, c_range_val);
    const auto tiny_threshold = hn::Set(d, c_tiny_threshold);
    const auto pi_threshold = hn::Set(d, c_pi_threshold);
    const auto half = hn::Set(d, c_half);
    const auto one = hn::Set(d, c_one);
    const auto zeros = hn::Zero(d);
    const auto nan_val = hn::Set(d, NPY_NAN);
    
auto is_nan = hn::IsNaN(x);
    auto is_inf = hn::IsInf(x);
    auto x_clean = hn::IfThenElse(is_nan, zeros, x);
    
    auto abs_x = hn::Abs(x_clean);
    auto is_tiny = hn::Lt(abs_x, tiny_threshold);
    auto is_large = hn::Gt(abs_x, range_val);
    
    if (__builtin_expect(hn::AllTrue(d, hn::Or(is_nan, is_tiny)), 0)) {
        return hn::IfThenElse(is_nan, nan_val, one);
    }
    
    auto k = hn::Round(hn::Mul(abs_x, inv_pi));
    auto k_pi = hn::Mul(k, pi_1);
    auto delta = hn::Abs(hn::Sub(abs_x, k_pi));
    auto is_boundary = hn::Lt(delta, pi_threshold);
    
    auto needs_scalar = hn::Or(is_large, is_boundary);
    needs_scalar = hn::Or(needs_scalar, is_inf);
    
    if (__builtin_expect(!hn::AllFalse(d, needs_scalar), 0)) {
        T NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) input_arr[hn::MaxLanes(d)];
        T NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) result_arr[hn::MaxLanes(d)];
        hn::Store(x, d, input_arr);
        
        size_t lanes_count = hn::Lanes(d);
        for (size_t i = 0; i < lanes_count; i++) {
            T val = input_arr[i];
            result_arr[i] = std::isnan(val) ? NPY_NAN : npy_cos(val);
        }
        
        return hn::Load(d, result_arr);
    }
    
    auto n = hn::Round(hn::MulAdd(x_clean, inv_pi, half));
    auto n_int = hn::ConvertTo(di64, n);
    auto odd = hn::ShiftLeft<63>(hn::BitCast(du64, n_int));
    auto n_shifted = hn::Sub(n, half);
    
    auto r = hn::NegMulAdd(n_shifted, pi_1, x_clean);
    r = hn::NegMulAdd(n_shifted, pi_2, r);
    r = hn::NegMulAdd(n_shifted, pi_3, r);
    
    auto r2 = hn::Mul(r, r);
    auto r3 = hn::Mul(r2, r);
    auto r4 = hn::Mul(r2, r2);
    
    const auto c0 = hn::Set(d, c_poly0);
    const auto c1 = hn::Set(d, c_poly1);
    const auto c2 = hn::Set(d, c_poly2);
    const auto c3 = hn::Set(d, c_poly3);
    const auto c4 = hn::Set(d, c_poly4);
    const auto c5 = hn::Set(d, c_poly5);
    const auto c6 = hn::Set(d, c_poly6);
    
    auto p01 = hn::MulAdd(r2, c1, c0);
    auto p23 = hn::MulAdd(r2, c3, c2);
    auto p45 = hn::MulAdd(r2, c5, c4);
    auto p46 = hn::MulAdd(r4, c6, p45);
    
    auto p26 = hn::MulAdd(r4, p46, p23);
    auto p06 = hn::MulAdd(r4, p26, p01);
    auto y = hn::MulAdd(r3, p06, r);
    
    auto simd_result = hn::BitCast(d, hn::Xor(hn::BitCast(du64, y), odd));
    simd_result = hn::IfThenElse(is_tiny, one, simd_result);
    simd_result = hn::IfThenElse(is_nan, nan_val, simd_result);
    
    return simd_result;
}
};

template <typename T, typename Op>
NPY_FINLINE void HWY_ATTR
simd_unary_loop(const T *src, npy_intp ssrc, T *dst, npy_intp sdst, npy_intp len)
{
    Op op_func;
    constexpr hn::ScalableTag<T> tag_t;
    HWY_LANES_CONSTEXPR int lanes = hn::Lanes(tag_t);
    constexpr int unroll_factor = 4;
    constexpr int unroll_lanes = lanes * unroll_factor;
    
    npy_intp remaining = len;
    
    if (ssrc == 1 && sdst == 1) {
        bool use_scalar = false;
        
        for (; remaining >= unroll_lanes; remaining -= unroll_lanes, src += unroll_lanes, dst += unroll_lanes) {
            __builtin_prefetch(src + unroll_lanes * 2, 0, 3);
            
            if (use_scalar) {
                for (int j = 0; j < unroll_lanes; j++) {
                    dst[j] = Op::scalar_call(src[j]);
                }
            }
            else {
                hn::Vec<decltype(tag_t)> x0 = hn::Load(tag_t, src);
                hn::Vec<decltype(tag_t)> x1 = hn::Load(tag_t, src + lanes);
                hn::Vec<decltype(tag_t)> x2 = hn::Load(tag_t, src + lanes * 2);
                hn::Vec<decltype(tag_t)> x3 = hn::Load(tag_t, src + lanes * 3);
                
                auto has_special = hn::Or(hn::IsNaN(x0), hn::IsInf(x0));
                has_special = hn::Or(has_special, hn::Or(hn::IsNaN(x1), hn::IsInf(x1)));
                has_special = hn::Or(has_special, hn::Or(hn::IsNaN(x2), hn::IsInf(x2)));
                has_special = hn::Or(has_special, hn::Or(hn::IsNaN(x3), hn::IsInf(x3)));
                
                if (!hn::AllFalse(tag_t, has_special)) {
                    use_scalar = true;
                    for (int j = 0; j < unroll_lanes; j++) {
                        dst[j] = Op::scalar_call(src[j]);
                    }
                }
                else {
                    hn::Vec<decltype(tag_t)> y0 = op_func(x0);
                    hn::Vec<decltype(tag_t)> y1 = op_func(x1);
                    hn::Vec<decltype(tag_t)> y2 = op_func(x2);
                    hn::Vec<decltype(tag_t)> y3 = op_func(x3);
                    
                    hn::Store(y0, tag_t, dst);
                    hn::Store(y1, tag_t, dst + lanes);
                    hn::Store(y2, tag_t, dst + lanes * 2);
                    hn::Store(y3, tag_t, dst + lanes * 3);
                }
            }
        }
        
        for (; remaining >= lanes; remaining -= lanes, src += lanes, dst += lanes) {
            if (use_scalar) {
                for (int j = 0; j < lanes; j++) {
                    dst[j] = Op::scalar_call(src[j]);
                }
            }
            else {
                hn::Vec<decltype(tag_t)> x_in = hn::Load(tag_t, src);
                auto has_special = hn::Or(hn::IsNaN(x_in), hn::IsInf(x_in));
                if (!hn::AllFalse(tag_t, has_special)) {
                    use_scalar = true;
                    for (int j = 0; j < lanes; j++) {
                        dst[j] = Op::scalar_call(src[j]);
                    }
                }
                else {
                    auto x_out = op_func(x_in);
                    hn::Store(x_out, tag_t, dst);
                }
            }
        }
        
        if (remaining > 0) {
            for (int j = 0; j < remaining; j++) {
                dst[j] = Op::scalar_call(src[j]);
            }
        }
    }
    else {
        constexpr hn::ScalableTag<typename TypeToInt<T>::Type> ind_tag_t;
        const hn::Vec<decltype(ind_tag_t)> src_index0 = hn::Mul(hn::Iota(ind_tag_t, 0), hn::Set(ind_tag_t, ssrc));
        const hn::Vec<decltype(ind_tag_t)> dst_index0 = hn::Mul(hn::Iota(ind_tag_t, 0), hn::Set(ind_tag_t, sdst));
        
        bool use_scalar = false;
        
        for (; remaining >= lanes; remaining -= lanes, src += ssrc * lanes, dst += sdst * lanes) {
            if (use_scalar) {
                for (int j = 0; j < lanes; j++) {
                    dst[j * sdst] = Op::scalar_call(src[j * ssrc]);
                }
            }
            else {
                hn::Vec<decltype(tag_t)> x_in = hn::GatherIndex(tag_t, src, src_index0);
                auto x_out = op_func(x_in);
                hn::ScatterIndex(x_out, tag_t, dst, dst_index0);
                
                if (!hn::AllFalse(tag_t, hn::Or(hn::IsNaN(x_in), hn::IsInf(x_in)))) {
                    use_scalar = true;
                }
            }
        }
        
        for (npy_intp i = 0; i < remaining; i++) {
            dst[i * sdst] = Op::scalar_call(src[i * ssrc]);
        }
    }
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(DOUBLE_sin)(char **args, npy_intp const *dimensions,
                                   npy_intp const *steps,
                                   void *NPY_UNUSED(data))
{
    npy_intp len = dimensions[0];
    const npy_intp ssrc = steps[0] / sizeof(npy_double);
    const npy_intp sdst = steps[1] / sizeof(npy_double);
    
    if (is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        npy_double *src = (npy_double *)args[0];
        npy_double *dst = (npy_double *)args[1];
        
        NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) npy_double tmp_buf[NPY_SIMD_WIDTH / sizeof(npy_double)];
        constexpr int chunk_size = NPY_SIMD_WIDTH / sizeof(npy_double);
        
        for (npy_intp i = 0; i < len; i += chunk_size) {
            npy_intp current_len = (len - i) < chunk_size ? (len - i) : chunk_size;
            
            for (npy_intp j = 0; j < current_len; j++) {
                tmp_buf[j] = src[ssrc * (i + j)];
            }
            
            const npy_double *tmp_src = tmp_buf;
            npy_double *tmp_dst = tmp_buf;
            simd_unary_loop<npy_double, OpSin<npy_double>>(tmp_src, 1, tmp_dst, 1, current_len);
            
            for (npy_intp j = 0; j < current_len; j++) {
                dst[sdst * (i + j)] = tmp_buf[j];
            }
        }
    }
    else {
        const npy_double *src = (npy_double *)args[0];
        npy_double *dst = (npy_double *)args[1];
        simd_unary_loop<npy_double, OpSin<npy_double>>(src, ssrc, dst, sdst, len);
    }
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(DOUBLE_cos)(char **args, npy_intp const *dimensions,
                                   npy_intp const *steps,
                                   void *NPY_UNUSED(data))
{
    npy_intp len = dimensions[0];
    const npy_intp ssrc = steps[0] / sizeof(npy_double);
    const npy_intp sdst = steps[1] / sizeof(npy_double);
    
    if (is_mem_overlap(args[0], steps[0], args[1], steps[1], len)) {
        npy_double *src = (npy_double *)args[0];
        npy_double *dst = (npy_double *)args[1];
        
        NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) npy_double tmp_buf[NPY_SIMD_WIDTH / sizeof(npy_double)];
        constexpr int chunk_size = NPY_SIMD_WIDTH / sizeof(npy_double);
        
        for (npy_intp i = 0; i < len; i += chunk_size) {
            npy_intp current_len = (len - i) < chunk_size ? (len - i) : chunk_size;
            
            for (npy_intp j = 0; j < current_len; j++) {
                tmp_buf[j] = src[ssrc * (i + j)];
            }
            
            const npy_double *tmp_src = tmp_buf;
            npy_double *tmp_dst = tmp_buf;
            simd_unary_loop<npy_double, OpCos<npy_double>>(tmp_src, 1, tmp_dst, 1, current_len);
            
            for (npy_intp j = 0; j < current_len; j++) {
                dst[sdst * (i + j)] = tmp_buf[j];
            }
        }
    }
    else {
        const npy_double *src = (npy_double *)args[0];
        npy_double *dst = (npy_double *)args[1];
        simd_unary_loop<npy_double, OpCos<npy_double>>(src, ssrc, dst, sdst, len);
    }
}

#else  // X86 and other platforms - use scalar libm

#define DISPATCH_DOUBLE_FUNC(func)                                          \
    NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_##func)(               \
            char **args, npy_intp const *dimensions, npy_intp const *steps, \
            void *NPY_UNUSED(data))                                         \
    {                                                                       \
        UNARY_LOOP                                                          \
        {                                                                   \
            const npy_double in1 = *(npy_double *)ip1;                      \
            *(npy_double *)op1 = npy_##func(in1);                           \
        }                                                                   \
    }

DISPATCH_DOUBLE_FUNC(sin)
DISPATCH_DOUBLE_FUNC(cos)

#endif  // ARM NEON with float64 support

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(FLOAT_sin)(char **args, npy_intp const *dimensions,
                                  npy_intp const *steps,
                                  void *NPY_UNUSED(data))
{
#if NPY_SIMD_FMA3
    npy_intp len = dimensions[0];

    if (is_mem_overlap(args[0], steps[0], args[1], steps[1], len) ||
        !npyv_loadable_stride_f32(steps[0]) ||
        !npyv_storable_stride_f32(steps[1])) {
        UNARY_LOOP
        {
            simd_sincos_f32((npy_float *)ip1, 1, (npy_float *)op1, 1, 1,
                            SIMD_COMPUTE_SIN);
        }
    }
    else {
        const npy_float *src = (npy_float *)args[0];
        npy_float *dst = (npy_float *)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);

        simd_sincos_f32(src, ssrc, dst, sdst, len, SIMD_COMPUTE_SIN);
    }
#else
    UNARY_LOOP
    {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = npy_sinf(in1);
    }
#endif
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(FLOAT_cos)(char **args, npy_intp const *dimensions,
                                  npy_intp const *steps,
                                  void *NPY_UNUSED(data))
{
#if NPY_SIMD_FMA3
    npy_intp len = dimensions[0];

    if (is_mem_overlap(args[0], steps[0], args[1], steps[1], len) ||
        !npyv_loadable_stride_f32(steps[0]) ||
        !npyv_storable_stride_f32(steps[1])) {
        UNARY_LOOP
        {
            simd_sincos_f32((npy_float *)ip1, 1, (npy_float *)op1, 1, 1,
                            SIMD_COMPUTE_COS);
        }
    }
    else {
        const npy_float *src = (npy_float *)args[0];
        npy_float *dst = (npy_float *)args[1];
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);

        simd_sincos_f32(src, ssrc, dst, sdst, len, SIMD_COMPUTE_COS);
    }
#else
    UNARY_LOOP
    {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = npy_cosf(in1);
    }
#endif
}

#if NPY_SIMD_FMA3 && defined(__aarch64__)
static void HWY_ATTR
simd_sincos_f16_impl(const npy_half *src, npy_intp ssrc, npy_half *dst, npy_intp sdst,
                     npy_intp len, SIMD_TRIG_OP trig_op)
{
    const ::hwy::float16_t* src16 = (const ::hwy::float16_t*)src;
    ::hwy::float16_t* dst16 = (::hwy::float16_t*)dst;

    const hn::ScalableTag<npy_float> f32_tag_t;
    const hn::ScalableTag<::hwy::float16_t> f16_tag_t;
    const int lanes = hn::Lanes(f16_tag_t);

    const vec_f32 zerosf = hn::Zero(f32_tag_t);
    const vec_f32 two_over_pi = hn::Set(f32_tag_t, 0x1.45f306p-1f);
    const vec_f32 codyw_pio2_highf = hn::Set(f32_tag_t, -0x1.921fb0p+00f);
    const vec_f32 codyw_pio2_medf = hn::Set(f32_tag_t, -0x1.5110b4p-22f);
    const vec_f32 codyw_pio2_lowf = hn::Set(f32_tag_t, -0x1.846988p-48f);
    const vec_f32 rint_cvt_magic = hn::Set(f32_tag_t, 0x1.800000p+23f);

    float max_codi = 117435.992f;
    if (trig_op == SIMD_COMPUTE_COS) {
        max_codi = 71476.0625f;
    }
    const vec_f32 max_cody = hn::Set(f32_tag_t, max_codi);

    npy_half NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) fallback_buf[hn::MaxLanes(f16_tag_t) * 2];

    for (; len > 0; len -= lanes, src16 += ssrc * lanes, dst16 += sdst * lanes) {
        hn::Vec<decltype(f16_tag_t)> x16_in;
        if (ssrc == 1) {
            x16_in = hn::LoadN(f16_tag_t, src16, len);
        } else {
            ::hwy::float16_t tmp[hn::MaxLanes(f16_tag_t)];
            for (int j = 0; j < lanes && j < len; ++j) {
                tmp[j] = src16[j * ssrc];
            }
            x16_in = hn::LoadN(f16_tag_t, tmp, lanes);
        }

        hn::Store(x16_in, f16_tag_t, (::hwy::float16_t*)fallback_buf);

        auto x32_in0 = hn::PromoteLowerTo(f32_tag_t, x16_in);
        auto x32_in1 = hn::PromoteUpperTo(f32_tag_t, x16_in);

        auto nnan_mask0 = hn::Not(hn::IsNaN(x32_in0));
        auto nnan_mask1 = hn::Not(hn::IsNaN(x32_in1));
        auto x_clean0 = hn::IfThenElse(nnan_mask0, x32_in0, zerosf);
        auto x_clean1 = hn::IfThenElse(nnan_mask1, x32_in1, zerosf);

        auto abs_x0 = hn::Abs(x_clean0);
        auto abs_x1 = hn::Abs(x_clean1);
        auto simd_mask0 = hn::Le(abs_x0, max_cody);
        auto simd_mask1 = hn::Le(abs_x1, max_cody);

        auto x0 = hn::IfThenElse(hn::And(nnan_mask0, simd_mask0), x_clean0, zerosf);
        auto x1 = hn::IfThenElse(hn::And(nnan_mask1, simd_mask1), x_clean1, zerosf);

        auto quadrant0 = hn::Mul(x0, two_over_pi);
        quadrant0 = hn::Add(quadrant0, rint_cvt_magic);
        quadrant0 = hn::Sub(quadrant0, rint_cvt_magic);
        auto quadrant1 = hn::Mul(x1, two_over_pi);
        quadrant1 = hn::Add(quadrant1, rint_cvt_magic);
        quadrant1 = hn::Sub(quadrant1, rint_cvt_magic);

        auto reduced_x0 = hn::MulAdd(quadrant0, codyw_pio2_highf, x0);
        reduced_x0 = hn::MulAdd(quadrant0, codyw_pio2_medf, reduced_x0);
        reduced_x0 = hn::MulAdd(quadrant0, codyw_pio2_lowf, reduced_x0);
        auto reduced_x1 = hn::MulAdd(quadrant1, codyw_pio2_highf, x1);
        reduced_x1 = hn::MulAdd(quadrant1, codyw_pio2_medf, reduced_x1);
        reduced_x1 = hn::MulAdd(quadrant1, codyw_pio2_lowf, reduced_x1);

        auto reduced_x2_0 = hn::Mul(reduced_x0, reduced_x0);
        auto reduced_x2_1 = hn::Mul(reduced_x1, reduced_x1);

        auto cos0 = simd_cosine_poly_f32(reduced_x2_0);
        auto cos1 = simd_cosine_poly_f32(reduced_x2_1);
        auto sin0 = simd_sine_poly_f32(reduced_x0, reduced_x2_0);
        auto sin1 = simd_sine_poly_f32(reduced_x1, reduced_x2_1);

        auto iquadrant0 = hn::NearestInt(quadrant0);
        auto iquadrant1 = hn::NearestInt(quadrant1);

        const hn::ScalableTag<int32_t> s32_tag;
        if (trig_op == SIMD_COMPUTE_COS) {
            iquadrant0 = hn::Add(iquadrant0, hn::Set(s32_tag, 1));
            iquadrant1 = hn::Add(iquadrant1, hn::Set(s32_tag, 1));
        }

        auto sine_mask0 = hn::RebindMask(f32_tag_t, hn::Eq(hn::And(iquadrant0, hn::Set(s32_tag, 1)), hn::Zero(s32_tag)));
        auto sine_mask1 = hn::RebindMask(f32_tag_t, hn::Eq(hn::And(iquadrant1, hn::Set(s32_tag, 1)), hn::Zero(s32_tag)));

        cos0 = hn::IfThenElse(sine_mask0, sin0, cos0);
        cos1 = hn::IfThenElse(sine_mask1, sin1, cos1);

        auto negate_mask0 = hn::RebindMask(f32_tag_t, hn::Eq(hn::And(iquadrant0, hn::Set(s32_tag, 2)), hn::Set(s32_tag, 2)));
        auto negate_mask1 = hn::RebindMask(f32_tag_t, hn::Eq(hn::And(iquadrant1, hn::Set(s32_tag, 2)), hn::Set(s32_tag, 2)));

        cos0 = hn::MaskedSubOr(cos0, negate_mask0, zerosf, cos0);
        cos1 = hn::MaskedSubOr(cos1, negate_mask1, zerosf, cos1);

        auto nan_val = hn::Set(f32_tag_t, NPY_NANF);
        cos0 = hn::IfThenElse(nnan_mask0, cos0, nan_val);
        cos1 = hn::IfThenElse(nnan_mask1, cos1, nan_val);

        auto y16_out0 = hn::DemoteTo(f16_tag_t, cos0);
        auto y16_out1 = hn::DemoteTo(f16_tag_t, cos1);
        auto y16_out = hn::Combine(f16_tag_t, y16_out1, y16_out0);

        hn::Store(y16_out, f16_tag_t, (::hwy::float16_t*)fallback_buf + lanes);

        npy_uint64 mask_bits0, mask_bits1;
        hn::StoreMaskBits(f32_tag_t, simd_mask0, (uint8_t*)&mask_bits0);
        hn::StoreMaskBits(f32_tag_t, simd_mask1, (uint8_t*)&mask_bits1);

        for (int j = 0; j < lanes / 2 && j < len; ++j) {
            npy_half out;
            if ((mask_bits0 >> j) & 1) {
                out = *((npy_half*)(fallback_buf + lanes) + j);
            } else {
                npy_float in1 = npy_half_to_float(fallback_buf[j]);
                if (trig_op == SIMD_COMPUTE_SIN) {
                    out = npy_float_to_half(npy_sinf(in1));
                } else {
                    out = npy_float_to_half(npy_cosf(in1));
                }
            }
            if (sdst == 1) {
                dst16[j] = ::hwy::float16_t::FromBits(out);
            } else {
                dst16[j * sdst] = ::hwy::float16_t::FromBits(out);
            }
        }
        for (int j = 0; j < lanes / 2 && (lanes / 2 + j) < len; ++j) {
            npy_half out;
            if ((mask_bits1 >> j) & 1) {
                out = *((npy_half*)(fallback_buf + lanes) + lanes / 2 + j);
            } else {
                npy_float in1 = npy_half_to_float(fallback_buf[lanes / 2 + j]);
                if (trig_op == SIMD_COMPUTE_SIN) {
                    out = npy_float_to_half(npy_sinf(in1));
                } else {
                    out = npy_float_to_half(npy_cosf(in1));
                }
            }
            if (sdst == 1) {
                dst16[lanes / 2 + j] = ::hwy::float16_t::FromBits(out);
            } else {
                dst16[(lanes / 2 + j) * sdst] = ::hwy::float16_t::FromBits(out);
            }
        }
    }
    npyv_cleanup();
}
#endif

#if NPY_SIMD_FMA3 && defined(__aarch64__)
namespace {
using namespace np::simd;

template <class D>
HWY_INLINE hn::Vec<D> TanPolyDouble(D d, hn::Vec<D> x)
{
    using TI = hn::RebindToSigned<D>;
    TI di;

    const double c0_d = 0x1.5555555555556p-2;
    const double c1_d = 0x1.1111111110a63p-3;
    const double c2_d = 0x1.ba1ba1bb46414p-5;
    const double c3_d = 0x1.664f47e5b5445p-6;
    const double c4_d = 0x1.226e5e5ecdfa3p-7;
    const double c5_d = 0x1.d6c7ddbf87047p-9;
    const double c6_d = 0x1.7ea75d05b583ep-10;
    const double c7_d = 0x1.289f22964a03cp-11;
    const double c8_d = 0x1.4e4fd14147622p-12;

    const double half_pi_hi_d = 0x1.921fb54442d18p0;
    const double half_pi_lo_d = 0x1.1a62633145c07p-54;
    const double two_over_pi_d = 0x1.45f306dc9c883p-1;
    const double shift_d = 0x1.8p52;

    const auto c0 = hn::Set(d, c0_d);
    const auto c1 = hn::Set(d, c1_d);
    const auto c2 = hn::Set(d, c2_d);
    const auto c3 = hn::Set(d, c3_d);
    const auto c4 = hn::Set(d, c4_d);
    const auto c5 = hn::Set(d, c5_d);
    const auto c6 = hn::Set(d, c6_d);
    const auto c7 = hn::Set(d, c7_d);
    const auto c8 = hn::Set(d, c8_d);

    const auto half_pi_hi = hn::Set(d, half_pi_hi_d);
    const auto half_pi_lo = hn::Set(d, half_pi_lo_d);
    const auto two_over_pi = hn::Set(d, two_over_pi_d);
    const auto shift = hn::Set(d, shift_d);
    const auto neg_one = hn::Set(d, -1.0);
    const auto one = hn::Set(d, 1.0);

    auto q = hn::MulAdd(x, two_over_pi, shift);
    q = hn::Sub(q, shift);
    auto qi = hn::NearestInt(q);

    auto r = hn::Neg(hn::MulSub(q, half_pi_hi, x));
    r = hn::Neg(hn::MulSub(q, half_pi_lo, r));
    r = hn::Mul(r, hn::Set(d, 0.5));

    auto r2 = hn::Mul(r, r);
    auto r4 = hn::Mul(r2, r2);
    auto r8 = hn::Mul(r4, r4);

    auto p01 = hn::MulAdd(r2, c2, c1);
    auto p23 = hn::MulAdd(r2, c4, c3);
    auto p03 = hn::MulAdd(r4, p23, p01);
    
    auto p45 = hn::MulAdd(r2, c6, c5);
    auto p67 = hn::MulAdd(r2, c8, c7);
    auto p47 = hn::MulAdd(r4, p67, p45);
    
    auto p_estrin = hn::MulAdd(r8, p47, p03);
    auto p = hn::MulAdd(r2, p_estrin, c0);
    p = hn::MulAdd(r2, hn::Mul(p, r), r);

    auto p2 = hn::Mul(p, p);
    auto n = hn::MulAdd(p2, one, neg_one);
    auto d_vec = hn::Add(p, p);

    auto odd = hn::RebindMask(d, hn::Eq(hn::And(qi, hn::Set(di, 1)), hn::Set(di, 1)));

    auto swap = hn::IfThenElse(odd, n, hn::Neg(d_vec));
    d_vec = hn::IfThenElse(odd, d_vec, n);
    n = swap;

    auto y = hn::Div(n, d_vec);

    return y;
}

template <class D>
HWY_INLINE hn::Vec<D> TanPolyFloat(D d, hn::Vec<D> x)
{
    using TI = hn::RebindToSigned<D>;
    TI di;

    const auto zero = hn::Zero(d);
    auto is_zero = hn::Eq(x, zero);
    if (hn::AllTrue(d, is_zero)) {
        return x;
    }

    const float c0_f = 0x1.55555p-2f;
    const float c1_f = 0x1.11166p-3f;
    const float c2_f = 0x1.b88a78p-5f;
    const float c3_f = 0x1.7b5756p-6f;
    const float c4_f = 0x1.4ef4cep-8f;
    const float c5_f = 0x1.0e1e74p-7f;
    
    const float neg_half_pi_hi_f = -0x1.921fb6p+0f;
    const float half_pi_mid_f = 0x1.777a5cp-25f;
    const float half_pi_lo_f = 0x1.ee59dap-50f;
    const float two_over_pi_f = 0x1.45f306p-1f;
    const float shift_f = 0x1.8p+23f;

    const auto c0 = hn::Set(d, c0_f);
    const auto c1 = hn::Set(d, c1_f);
    const auto c2 = hn::Set(d, c2_f);
    const auto c3 = hn::Set(d, c3_f);
    const auto c4 = hn::Set(d, c4_f);
    const auto c5 = hn::Set(d, c5_f);
    
    const auto neg_half_pi_hi = hn::Set(d, neg_half_pi_hi_f);
    const auto half_pi_mid = hn::Set(d, half_pi_mid_f);
    const auto half_pi_lo = hn::Set(d, half_pi_lo_f);
    const auto two_over_pi = hn::Set(d, two_over_pi_f);
    const auto shift = hn::Set(d, shift_f);
    const auto one = hn::Set(d, 1.0f);

    auto q = hn::MulAdd(x, two_over_pi, shift);
    auto n = hn::Sub(q, shift);
    auto n_int = hn::NearestInt(n);

    auto r = hn::MulAdd(n, neg_half_pi_hi, x);
    r = hn::MulAdd(n, half_pi_mid, r);
    r = hn::MulAdd(n, half_pi_lo, r);

    auto odd = hn::RebindMask(d, hn::Eq(hn::And(n_int, hn::Set(di, 1)), hn::Set(di, 1)));
    auto z = hn::IfThenElse(odd, hn::Neg(r), r);

    auto z2 = hn::Mul(z, z);
    auto z4 = hn::Mul(z2, z2);
    auto z8 = hn::Mul(z4, z4);

    auto p01 = hn::MulAdd(z2, c1, c0);
    auto p23 = hn::MulAdd(z2, c3, c2);
    auto p45 = hn::MulAdd(z2, c5, c4);
    auto p0123 = hn::MulAdd(z4, p23, p01);
    auto p = hn::MulAdd(z8, p45, p0123);

    auto y = hn::MulAdd(p, hn::Mul(z, z2), z);

    auto inv_y = hn::Div(one, y);
    auto result = hn::IfThenElse(odd, inv_y, y);
    
    result = hn::IfThenElse(is_zero, zero, result);

    return result;
}

}
#endif

#if NPY_SIMD_FMA3 && defined(__aarch64__)
static void HWY_ATTR
simd_tan_f32_impl_contiguous(const npy_float *src, npy_float *dst, npy_intp len)
{
    const hn::ScalableTag<float> d;
    const float range_limit = 0x1p15f;
    const auto range_limit_vec = hn::Set(d, range_limit);

    HWY_LANES_CONSTEXPR int lanes = Lanes<float>();
    constexpr int UNROLL = 4;
    constexpr int STEP = lanes * UNROLL;

    float NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) fallback_buf[STEP * 2];

    for (; len >= STEP; len -= STEP, src += STEP, dst += STEP) {
        __builtin_prefetch(src + STEP, 0, 3);

        bool all_simd = true;
        for (int j = 0; j < UNROLL; ++j) {
            auto x_in = hn::LoadU(d, src + j * lanes);
            auto abs_x = hn::Abs(x_in);
            auto needs_fallback = hn::Ge(abs_x, range_limit_vec);
            auto is_nan = hn::IsNaN(x_in);
            auto is_inf = hn::Eq(abs_x, hn::Inf(d));
            auto simd_mask = hn::Not(hn::Or(hn::Or(is_nan, is_inf), needs_fallback));

            hn::Store(x_in, d, fallback_buf + j * lanes);
            hn::StoreMaskBits(d, simd_mask, (uint8_t*)(fallback_buf + j * lanes + lanes));

            if (!hn::AllTrue(d, simd_mask)) {
                all_simd = false;
            }
        }

        if (all_simd) {
            feclearexcept(FE_ALL_EXCEPT);
            for (int j = 0; j < UNROLL; ++j) {
                hn::Store(TanPolyFloat(d, hn::LoadU(d, src + j * lanes)), d, dst + j * lanes);
            }
            feclearexcept(FE_ALL_EXCEPT);
        } else {
            feclearexcept(FE_ALL_EXCEPT);
            for (int j = 0; j < UNROLL; ++j) {
                for (int i = 0; i < lanes; ++i) {
                    dst[j * lanes + i] = npy_tanf(fallback_buf[j * lanes + i]);
                }
            }
        }
    }

    for (; len > 0; len -= lanes, src += lanes, dst += lanes) {
        auto x_in = hn::LoadN(d, src, len);
        auto abs_x = hn::Abs(x_in);
        auto needs_fallback = hn::Ge(abs_x, range_limit_vec);
        auto is_nan = hn::IsNaN(x_in);
        auto is_inf = hn::Eq(abs_x, hn::Inf(d));
        auto simd_mask = hn::Not(hn::Or(hn::Or(is_nan, is_inf), needs_fallback));

        if (hn::AllTrue(d, simd_mask)) {
            feclearexcept(FE_ALL_EXCEPT);
            hn::StoreN(TanPolyFloat(d, x_in), d, dst, len);
            feclearexcept(FE_ALL_EXCEPT);
        } else if (hn::AllFalse(d, simd_mask)) {
            feclearexcept(FE_ALL_EXCEPT);
            hn::Store(x_in, d, fallback_buf);
            for (int i = 0; i < lanes && i < len; ++i) {
                dst[i] = npy_tanf(fallback_buf[i]);
            }
        } else {
            auto y = TanPolyFloat(d, x_in);
            hn::Store(x_in, d, fallback_buf);
            npy_uint64 mask_bits;
            hn::StoreMaskBits(d, simd_mask, (uint8_t*)&mask_bits);
            feclearexcept(FE_ALL_EXCEPT);
            for (int i = 0; i < lanes && i < len; ++i) {
                if ((mask_bits >> i) & 1) {
                    dst[i] = y.raw[i];
                } else {
                    dst[i] = npy_tanf(fallback_buf[i]);
                }
            }
        }
    }
    npyv_cleanup();
}

static void HWY_ATTR
simd_tan_f32_impl_strided(const npy_float *src, npy_intp ssrc, npy_float *dst, npy_intp sdst, npy_intp len)
{
    const hn::ScalableTag<float> d;
    const hn::ScalableTag<int32_t> ind_tag_t;
    
    const float range_limit = 0x1p15f;
    const auto range_limit_vec = hn::Set(d, range_limit);
    
    HWY_LANES_CONSTEXPR int lanes = Lanes<float>();
    const auto src_index = hn::Mul(hn::Iota(ind_tag_t, 0), hn::Set(ind_tag_t, ssrc));
    
    float NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) ip_fback[hn::MaxLanes(d)];
    
    for (; len > 0; len -= lanes, src += ssrc * lanes, dst += sdst * lanes) {
        auto x_in = hn::GatherIndexN(d, src, src_index, len);
        
        auto abs_x = hn::Abs(x_in);
        auto needs_fallback = hn::Ge(abs_x, range_limit_vec);
        auto is_nan = hn::IsNaN(x_in);
        auto is_inf = hn::Eq(abs_x, hn::Inf(d));
        auto simd_mask = hn::Not(hn::Or(hn::Or(is_nan, is_inf), needs_fallback));
        
        if (hn::AllTrue(d, simd_mask)) {
            auto y = TanPolyFloat(d, x_in);
            for (int i = 0; i < lanes && i < len; ++i) {
                dst[i * sdst] = y.raw[i];
            }
        } else if (hn::AllFalse(d, simd_mask)) {
            feclearexcept(FE_ALL_EXCEPT);
            hn::Store(x_in, d, ip_fback);
            for (int i = 0; i < lanes && i < len; ++i) {
                dst[i * sdst] = npy_tanf(ip_fback[i]);
            }
        } else {
            auto y = TanPolyFloat(d, x_in);
            hn::Store(x_in, d, ip_fback);
            npy_uint64 mask_bits;
            hn::StoreMaskBits(d, simd_mask, (uint8_t*)&mask_bits);
            feclearexcept(FE_ALL_EXCEPT);
            for (int i = 0; i < lanes && i < len; ++i) {
                if ((mask_bits >> i) & 1) {
                    dst[i * sdst] = y.raw[i];
                } else {
                    dst[i * sdst] = npy_tanf(ip_fback[i]);
                }
            }
        }
    }
    npyv_cleanup();
}

static void HWY_ATTR
simd_tan_f32_impl(const npy_float *src, npy_intp ssrc, npy_float *dst, npy_intp sdst, npy_intp len)
{
    if (ssrc == 1 && sdst == 1) {
        simd_tan_f32_impl_contiguous(src, dst, len);
    } else {
        simd_tan_f32_impl_strided(src, ssrc, dst, sdst, len);
    }
}
#endif

#if NPY_SIMD_FMA3 && defined(__aarch64__)
static void HWY_ATTR
simd_tan_f16_impl(const npy_half *src, npy_intp ssrc, npy_half *dst, npy_intp sdst, npy_intp len)
{
    const ::hwy::float16_t* src16 = (const ::hwy::float16_t*)src;
    ::hwy::float16_t* dst16 = (::hwy::float16_t*)dst;

    constexpr hn::ScalableTag<npy_float> f32_tag_t;
    constexpr hn::ScalableTag<::hwy::float16_t> f16_tag_t;
    HWY_LANES_CONSTEXPR int lanes = Lanes<::hwy::float16_t>();

    const float range_limit_f16 = 0x1p11f;
    const auto range_limit_vec = hn::Set(f32_tag_t, range_limit_f16);

    npy_half NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) fallback_buf[lanes * 2];

    for (; len > 0; len -= lanes, src16 += ssrc * lanes, dst16 += sdst * lanes) {
        hn::Vec<decltype(f16_tag_t)> x16_in;
        if (ssrc == 1) {
            x16_in = hn::LoadN(f16_tag_t, src16, len);
        } else {
            ::hwy::float16_t tmp[lanes];
            for (int j = 0; j < lanes && j < len; ++j) {
                tmp[j] = src16[j * ssrc];
            }
            x16_in = hn::LoadN(f16_tag_t, tmp, lanes);
        }

        hn::Store(x16_in, f16_tag_t, (::hwy::float16_t*)fallback_buf);

        auto x32_in0 = hn::PromoteLowerTo(f32_tag_t, x16_in);
        auto x32_in1 = hn::PromoteUpperTo(f32_tag_t, x16_in);

        auto abs_x0 = hn::Abs(x32_in0);
        auto abs_x1 = hn::Abs(x32_in1);

        auto is_large0 = hn::Ge(abs_x0, range_limit_vec);
        auto is_large1 = hn::Ge(abs_x1, range_limit_vec);
        auto is_nan0 = hn::IsNaN(x32_in0);
        auto is_nan1 = hn::IsNaN(x32_in1);
        auto is_inf0 = hn::Eq(abs_x0, hn::Inf(f32_tag_t));
        auto is_inf1 = hn::Eq(abs_x1, hn::Inf(f32_tag_t));

        auto needs_fallback0 = hn::Or(is_large0, hn::Or(is_nan0, is_inf0));
        auto needs_fallback1 = hn::Or(is_large1, hn::Or(is_nan1, is_inf1));

        bool all_simd0 = hn::AllTrue(f32_tag_t, hn::Not(needs_fallback0));
        bool all_simd1 = hn::AllTrue(f32_tag_t, hn::Not(needs_fallback1));
        bool none_simd0 = hn::AllFalse(f32_tag_t, hn::Not(needs_fallback0));
        bool none_simd1 = hn::AllFalse(f32_tag_t, hn::Not(needs_fallback1));

        feclearexcept(FE_ALL_EXCEPT);
        auto y32_out0 = TanPolyFloat(f32_tag_t, x32_in0);
        auto y32_out1 = TanPolyFloat(f32_tag_t, x32_in1);
        feclearexcept(FE_ALL_EXCEPT);

        auto y16_out0 = hn::DemoteTo(f16_tag_t, y32_out0);
        auto y16_out1 = hn::DemoteTo(f16_tag_t, y32_out1);
        auto y16_out = hn::Combine(f16_tag_t, y16_out1, y16_out0);

        if (all_simd0 && all_simd1) {
            if (sdst == 1) {
                hn::StoreN(y16_out, f16_tag_t, dst16, len);
            } else {
                ::hwy::float16_t tmp[lanes];
                hn::StoreN(y16_out, f16_tag_t, tmp, lanes);
                for (int j = 0; j < lanes && j < len; ++j) {
                    dst16[j * sdst] = tmp[j];
                }
            }
        } else if (none_simd0 && none_simd1) {
            for (int j = 0; j < lanes && j < len; ++j) {
                npy_float in1 = npy_half_to_float(fallback_buf[j]);
                npy_half out = npy_float_to_half(npy_tanf(in1));
                if (sdst == 1) {
                    dst16[j] = ::hwy::float16_t::FromBits(out);
                } else {
                    dst16[j * sdst] = ::hwy::float16_t::FromBits(out);
                }
            }
        } else {
            hn::Store(y16_out, f16_tag_t, (::hwy::float16_t*)fallback_buf + lanes);
            npy_uint64 mask_bits0, mask_bits1;
            hn::StoreMaskBits(f32_tag_t, hn::Not(needs_fallback0), (uint8_t*)&mask_bits0);
            hn::StoreMaskBits(f32_tag_t, hn::Not(needs_fallback1), (uint8_t*)&mask_bits1);
            for (int j = 0; j < lanes / 2 && j < len; ++j) {
                npy_half out;
                if ((mask_bits0 >> j) & 1) {
                    out = *((npy_half*)(fallback_buf + lanes) + j);
                } else {
                    npy_float in1 = npy_half_to_float(fallback_buf[j]);
                    out = npy_float_to_half(npy_tanf(in1));
                }
                if (sdst == 1) {
                    dst16[j] = ::hwy::float16_t::FromBits(out);
                } else {
                    dst16[j * sdst] = ::hwy::float16_t::FromBits(out);
                }
            }
            for (int j = 0; j < lanes / 2 && (lanes / 2 + j) < len; ++j) {
                npy_half out;
                if ((mask_bits1 >> j) & 1) {
                    out = *((npy_half*)(fallback_buf + lanes) + lanes / 2 + j);
                } else {
                    npy_float in1 = npy_half_to_float(fallback_buf[lanes / 2 + j]);
                    out = npy_float_to_half(npy_tanf(in1));
                }
                if (sdst == 1) {
                    dst16[lanes / 2 + j] = ::hwy::float16_t::FromBits(out);
                } else {
                    dst16[(lanes / 2 + j) * sdst] = ::hwy::float16_t::FromBits(out);
                }
            }
        }
    }
    npyv_cleanup();
}
#endif

#if NPY_SIMD_FMA3 && defined(__aarch64__)
static void HWY_ATTR
simd_tan_f64_impl_contiguous(const npy_double *src, npy_double *dst, npy_intp len)
{
    const hn::ScalableTag<double> d;
    const double range_limit = 0x1p23;
    const auto range_limit_vec = hn::Set(d, range_limit);

    HWY_LANES_CONSTEXPR int lanes = Lanes<double>();
    constexpr int UNROLL = 4;
    constexpr int STEP = lanes * UNROLL;

    double NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) fallback_buf[STEP * 2];

    for (; len >= STEP; len -= STEP, src += STEP, dst += STEP) {
        __builtin_prefetch(src + STEP, 0, 3);

        bool all_simd = true;
        for (int j = 0; j < UNROLL; ++j) {
            auto x_in = hn::LoadU(d, src + j * lanes);
            auto abs_x = hn::Abs(x_in);
            auto needs_fallback = hn::Ge(abs_x, range_limit_vec);
            auto is_nan = hn::IsNaN(x_in);
            auto is_inf = hn::Eq(abs_x, hn::Inf(d));
            auto simd_mask = hn::Not(hn::Or(hn::Or(is_nan, is_inf), needs_fallback));

            hn::Store(x_in, d, fallback_buf + j * lanes);
            hn::StoreMaskBits(d, simd_mask, (uint8_t*)(fallback_buf + j * lanes + lanes));

            if (!hn::AllTrue(d, simd_mask)) {
                all_simd = false;
            }
        }

        if (all_simd) {
            feclearexcept(FE_ALL_EXCEPT);
            for (int j = 0; j < UNROLL; ++j) {
                hn::Store(TanPolyDouble(d, hn::LoadU(d, src + j * lanes)), d, dst + j * lanes);
            }
        } else {
            feclearexcept(FE_ALL_EXCEPT);
            for (int j = 0; j < UNROLL; ++j) {
                for (int i = 0; i < lanes; ++i) {
                    dst[j * lanes + i] = npy_tan(fallback_buf[j * lanes + i]);
                }
            }
        }
    }

    for (; len > 0; len -= lanes, src += lanes, dst += lanes) {
        auto x_in = hn::LoadN(d, src, len);
        auto abs_x = hn::Abs(x_in);
        auto needs_fallback = hn::Ge(abs_x, range_limit_vec);
        auto is_nan = hn::IsNaN(x_in);
        auto is_inf = hn::Eq(abs_x, hn::Inf(d));
        auto simd_mask = hn::Not(hn::Or(hn::Or(is_nan, is_inf), needs_fallback));

        if (hn::AllTrue(d, simd_mask)) {
            feclearexcept(FE_ALL_EXCEPT);
            hn::StoreN(TanPolyDouble(d, x_in), d, dst, len);
            feclearexcept(FE_ALL_EXCEPT);
        } else if (hn::AllFalse(d, simd_mask)) {
            feclearexcept(FE_ALL_EXCEPT);
            hn::Store(x_in, d, fallback_buf);
            for (int i = 0; i < lanes && i < len; ++i) {
                dst[i] = npy_tan(fallback_buf[i]);
            }
        } else {
            auto y = TanPolyDouble(d, x_in);
            hn::Store(x_in, d, fallback_buf);
            npy_uint64 mask_bits;
            hn::StoreMaskBits(d, simd_mask, (uint8_t*)&mask_bits);
            feclearexcept(FE_ALL_EXCEPT);
            for (int i = 0; i < lanes && i < len; ++i) {
                if ((mask_bits >> i) & 1) {
                    dst[i] = y.raw[i];
                } else {
                    dst[i] = npy_tan(fallback_buf[i]);
                }
            }
        }
    }
    npyv_cleanup();
}

static void HWY_ATTR
simd_tan_f64_impl_strided(const npy_double *src, npy_intp ssrc, npy_double *dst, npy_intp sdst, npy_intp len)
{
    const hn::ScalableTag<double> d;
    const hn::ScalableTag<int64_t> ind_tag_t;

    const double range_limit = 0x1p23;
    const auto range_limit_vec = hn::Set(d, range_limit);

    HWY_LANES_CONSTEXPR int lanes = Lanes<double>();
    const auto src_index = hn::Mul(hn::Iota(ind_tag_t, 0), hn::Set(ind_tag_t, ssrc));

    double NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) ip_fback[hn::MaxLanes(d)];

    for (; len > 0; len -= lanes, src += ssrc * lanes, dst += sdst * lanes) {
        auto x_in = hn::GatherIndexN(d, src, src_index, len);

        auto abs_x = hn::Abs(x_in);
        auto needs_fallback = hn::Ge(abs_x, range_limit_vec);
        auto is_nan = hn::IsNaN(x_in);
        auto is_inf = hn::Eq(abs_x, hn::Inf(d));
        auto simd_mask = hn::Not(hn::Or(hn::Or(is_nan, is_inf), needs_fallback));

        if (hn::AllTrue(d, simd_mask)) {
            hn::Vec<decltype(d)> y = TanPolyDouble(d, x_in);
            for (int i = 0; i < lanes && i < len; ++i) {
                dst[i * sdst] = y.raw[i];
            }
        } else if (hn::AllFalse(d, simd_mask)) {
            feclearexcept(FE_ALL_EXCEPT);
            hn::Store(x_in, d, ip_fback);
            for (int i = 0; i < lanes && i < len; ++i) {
                dst[i * sdst] = npy_tan(ip_fback[i]);
            }
        } else {
            auto y = TanPolyDouble(d, x_in);
            hn::Store(x_in, d, ip_fback);
            npy_uint64 mask_bits;
            hn::StoreMaskBits(d, simd_mask, (uint8_t*)&mask_bits);
            feclearexcept(FE_ALL_EXCEPT);
            for (int i = 0; i < lanes && i < len; ++i) {
                if ((mask_bits >> i) & 1) {
                    dst[i * sdst] = y.raw[i];
                } else {
                    dst[i * sdst] = npy_tan(ip_fback[i]);
                }
            }
        }
    }
    npyv_cleanup();
}

static void HWY_ATTR
simd_tan_f64_impl(const npy_double *src, npy_intp ssrc, npy_double *dst, npy_intp sdst, npy_intp len)
{
    if (ssrc == 1 && sdst == 1) {
        simd_tan_f64_impl_contiguous(src, dst, len);
    } else {
        simd_tan_f64_impl_strided(src, ssrc, dst, sdst, len);
    }
}
#endif

#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
static void
simd_svml_tan_f32(const npyv_lanetype_f32 *src, npy_intp ssrc,
                  npyv_lanetype_f32 *dst, npy_intp sdst, npy_intp len)
{
    const int vstep = npyv_nlanes_f32;
    for (; len > 0; len -= vstep, src += ssrc*vstep, dst += sdst*vstep) {
        npyv_f32 x;
        if (ssrc == 1) {
            x = npyv_load_tillz_f32(src, len);
        } else {
            x = npyv_loadn_tillz_f32(src, ssrc, len);
        }
        npyv_f32 out = __svml_tanf16(x);
        if (sdst == 1) {
            npyv_store_till_f32(dst, len, out);
        } else {
            npyv_storen_till_f32(dst, sdst, len, out);
        }
    }
    npyv_cleanup();
}

static void
simd_svml_tan_f64(const npyv_lanetype_f64 *src, npy_intp ssrc,
                  npyv_lanetype_f64 *dst, npy_intp sdst, npy_intp len)
{
    const int vstep = npyv_nlanes_f64;
    for (; len > 0; len -= vstep, src += ssrc*vstep, dst += sdst*vstep) {
        npyv_f64 x;
        if (ssrc == 1) {
            x = npyv_load_tillz_f64(src, len);
        } else {
            x = npyv_loadn_tillz_f64(src, ssrc, len);
        }
        npyv_f64 out = __svml_tan8_ha(x);
        if (sdst == 1) {
            npyv_store_till_f64(dst, len, out);
        } else {
            npyv_storen_till_f64(dst, sdst, len, out);
        }
    }
    npyv_cleanup();
}
#endif

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_tan)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
    const npy_double *src = (npy_double*)args[0];
          npy_double *dst = (npy_double*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        npyv_loadable_stride_f64(steps[0]) &&
        npyv_storable_stride_f64(steps[1]))
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_double);
        const npy_intp sdst = steps[1] / sizeof(npy_double);
        simd_svml_tan_f64(src, ssrc, dst, sdst, len);
        return;
    }
#elif NPY_SIMD_FMA3 && defined(__aarch64__)
    const npy_double *src = (npy_double*)args[0];
          npy_double *dst = (npy_double*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        steps[0] % sizeof(npy_double) == 0 &&
        steps[1] % sizeof(npy_double) == 0)
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_double);
        const npy_intp sdst = steps[1] / sizeof(npy_double);
        simd_tan_f64_impl(src, ssrc, dst, sdst, len);
        return;
    }
#else
    UNARY_LOOP {
        const npy_double in1 = *(npy_double *)ip1;
        *(npy_double *)op1 = npy_tan(in1);
    }
#endif
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_tan)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
    const npy_float *src = (npy_float*)args[0];
          npy_float *dst = (npy_float*)args[1];
    const npy_intp len = dimensions[0];    
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        npyv_loadable_stride_f32(steps[0]) &&
        npyv_storable_stride_f32(steps[1]))
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);
        simd_svml_tan_f32(src, ssrc, dst, sdst, len);
        return;
    }
#elif NPY_SIMD_FMA3 && defined(__aarch64__)
    const npy_float *src = (npy_float*)args[0];
          npy_float *dst = (npy_float*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        steps[0] % sizeof(npy_float) == 0 &&
        steps[1] % sizeof(npy_float) == 0)
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_float);
        const npy_intp sdst = steps[1] / sizeof(npy_float);
        simd_tan_f32_impl(src, ssrc, dst, sdst, len);
        return;
    }
#else
    UNARY_LOOP {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = npy_tanf(in1);
    }
#endif
}

#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
  #define NPY__SVML_IS_ENABLED 1
#else
  #define NPY__SVML_IS_ENABLED 0
#endif

#if NPY__SVML_IS_ENABLED && !defined(NPY_HAVE_AVX512_SPR)

typedef __m256i npyvh_f16;
#define npyv_cvt_f16_f32 _mm512_cvtph_ps
#define npyv_cvt_f32_f16 _mm512_cvtps_ph
NPY_FINLINE npyvh_f16 npyvh_load_f16(const void *ptr) {
    return _mm256_loadu_si256((const __m256i*)(ptr));
}
NPY_FINLINE void npyvh_store_f16(void *ptr, npyvh_f16 data) {
    _mm256_storeu_si256((__m256i*)ptr, data);
}
NPY_FINLINE npyvh_f16 npyvh_load_till_f16(const npy_half *ptr, npy_uintp nlane, npy_half fill)
{
    assert(nlane > 0);
    const __m256i vfill = _mm256_set1_epi16(fill);
    const __mmask16 mask = (0x0001 << nlane) - 0x0001;
    return _mm256_mask_loadu_epi16(vfill, mask, ptr);
}
NPY_FINLINE void npyvh_store_till_f16(npy_half *ptr, npy_uintp nlane, npyvh_f16 data)
{
    assert(nlane > 0);
    const __mmask16 mask = (0x0001 << nlane) - 0x0001;
    _mm256_mask_storeu_epi16(ptr, mask, data);
}

template<typename Func>
static void avx512_trigonometric_f16(const npy_half *src, npy_half *dst, npy_intp len, Func math_func)
{
    const int num_lanes = npyv_nlanes_f32;
    npyvh_f16 x, out;
    npyv_f32 x_ps, out_ps;
    for (; len > 0; len -= num_lanes, src += num_lanes, dst += num_lanes) {
        if (len >= num_lanes) {
            x       = npyvh_load_f16(src);
            x_ps    = npyv_cvt_f16_f32(x);
            out_ps  = math_func(x_ps); 
            out     = npyv_cvt_f32_f16(out_ps, 0);
            npyvh_store_f16(dst, out);
        }
        else {
            x       = npyvh_load_till_f16(src, len, 0);
            x_ps    = npyv_cvt_f16_f32(x);
            out_ps  = math_func(x_ps);
            out     = npyv_cvt_f32_f16(out_ps, 0);
            npyvh_store_till_f16(dst, len, out);
        }
    }
    npyv_cleanup();
}

static void avx512_sin_f16(const npy_half *src, npy_half *dst, npy_intp len)
{
    avx512_trigonometric_f16(src, dst, len, __svml_sinf16);
}

static void avx512_cos_f16(const npy_half *src, npy_half *dst, npy_intp len)
{
    avx512_trigonometric_f16(src, dst, len, __svml_cosf16);
}

static void avx512_tan_f16(const npy_half *src, npy_half *dst, npy_intp len)
{
    avx512_trigonometric_f16(src, dst, len, __svml_tanf16);
}
#endif // NPY__SVML_IS_ENABLED

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(HALF_sin)(char **args, npy_intp const *dimensions,
                                  npy_intp const *steps,
                                  void *NPY_UNUSED(data))
{
#if NPY__SVML_IS_ENABLED
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        (steps[0] == sizeof(npy_half)) &&
        (steps[1] == sizeof(npy_half))) {
    #ifdef NPY_HAVE_AVX512_SPR
        __svml_sins32(src, dst, len);
    #else
        avx512_sin_f16(src, dst, len);
    #endif
        return;
    }
#elif NPY_SIMD_FMA3 && defined(__aarch64__)
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        steps[0] % sizeof(npy_half) == 0 &&
        steps[1] % sizeof(npy_half) == 0)
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_half);
        const npy_intp sdst = steps[1] / sizeof(npy_half);
        simd_sincos_f16_impl(src, ssrc, dst, sdst, len, SIMD_COMPUTE_SIN);
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_sinf(in1));
    }
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(HALF_cos)(char **args, npy_intp const *dimensions,
                                   npy_intp const *steps,
                                   void *NPY_UNUSED(data))
{
#if NPY__SVML_IS_ENABLED
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        (steps[0] == sizeof(npy_half)) &&
        (steps[1] == sizeof(npy_half))) {
    #ifdef NPY_HAVE_AVX512_SPR
        __svml_coss32(src, dst, len);
    #else
        avx512_cos_f16(src, dst, len);
    #endif
        return;
    }
#elif NPY_SIMD_FMA3 && defined(__aarch64__)
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        steps[0] % sizeof(npy_half) == 0 &&
        steps[1] % sizeof(npy_half) == 0)
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_half);
        const npy_intp sdst = steps[1] / sizeof(npy_half);
        simd_sincos_f16_impl(src, ssrc, dst, sdst, len, SIMD_COMPUTE_COS);
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_cosf(in1));
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(HALF_tan)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
#if NPY__SVML_IS_ENABLED
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        (steps[0] == sizeof(npy_half)) &&
        (steps[1] == sizeof(npy_half))) {
    #ifdef NPY_HAVE_AVX512_SPR
        __svml_tans32(src, dst, len);
    #else
        avx512_tan_f16(src, dst, len);
    #endif
        return;
    }
#elif NPY_SIMD_FMA3 && defined(__aarch64__)
    const npy_half *src = (npy_half*)args[0];
          npy_half *dst = (npy_half*)args[1];
    const npy_intp len = dimensions[0];
    if (!is_mem_overlap(src, steps[0], dst, steps[1], len) &&
        steps[0] % sizeof(npy_half) == 0 &&
        steps[1] % sizeof(npy_half) == 0)
    {
        const npy_intp ssrc = steps[0] / sizeof(npy_half);
        const npy_intp sdst = steps[1] / sizeof(npy_half);
        simd_tan_f16_impl(src, ssrc, dst, sdst, len);
        return;
    }
#endif
    UNARY_LOOP {
        const npy_float in1 = npy_half_to_float(*(npy_half *)ip1);
        *((npy_half *)op1) = npy_float_to_half(npy_tanf(in1));
    }
}
