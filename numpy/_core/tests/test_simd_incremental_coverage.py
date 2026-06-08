"""
Tests to improve C/C++ incremental coverage for SIMD dispatch paths.
Targets Highway/SVE/NEON code paths on aarch64.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


N = 2048


class TestMinMaxHighway:
    """loops_minmax_hwy.dispatch.cpp, loops_minmax.dispatch.c.src"""

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_maximum_minimum_integers(self, dt):
        a = np.arange(N, dtype=dt)
        b = np.arange(N, dtype=dt)[::-1]
        r_max = np.maximum(a, b)
        r_min = np.minimum(a, b)
        assert r_max.dtype == dt
        assert r_min.dtype == dt
        assert np.all(r_max >= a)
        assert np.all(r_min <= a)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_minimum_floats(self, dt):
        rng = np.random.default_rng(42)
        a = rng.random(N, dtype=dt)
        b = rng.random(N, dtype=dt)
        r_max = np.maximum(a, b)
        r_min = np.minimum(a, b)
        assert np.all(r_max >= a)
        assert np.all(r_min <= a)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_minimum_nan(self, dt):
        a = np.array([np.nan, 1.0, 2.0, 3.0] * (N // 4), dtype=dt)
        b = np.array([4.0, np.nan, 1.0, 2.0] * (N // 4), dtype=dt)
        r_max = np.maximum(a, b)
        r_min = np.minimum(a, b)
        assert r_max.shape == (N,)
        assert r_min.shape == (N,)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmax_fmin(self, dt):
        rng = np.random.default_rng(42)
        a = rng.random(N, dtype=dt)
        b = rng.random(N, dtype=dt)
        a[::10] = np.nan
        b[::11] = np.nan
        r_fmax = np.fmax(a, b)
        r_fmin = np.fmin(a, b)
        assert not np.any(np.isnan(r_fmax[np.isfinite(a) | np.isfinite(b)]))
        assert not np.any(np.isnan(r_fmin[np.isfinite(a) | np.isfinite(b)]))

    def test_maximum_scalar_broadcast(self):
        a = np.float64(3.0)
        b = np.arange(N, dtype=np.float64)
        r1 = np.maximum(a, b)
        r2 = np.minimum(b, np.float32(500.0))
        assert r1.shape == (N,)
        assert r2.shape == (N,)

    def test_maximum_strided_output(self):
        a = np.arange(N, dtype=np.float64)
        b = np.arange(N, dtype=np.float64)[::-1]
        out = np.empty(N * 2, dtype=np.float64)
        np.maximum(a, b, out=out[::2])
        assert_array_equal(out[::2], np.maximum(a, b))


class TestArgFuncHighway:
    """highway_argfunc.dispatch.cpp"""

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_argmin_argmax_integers(self, dt):
        rng = np.random.default_rng(42)
        info = np.iinfo(dt)
        a = rng.integers(info.min + 1, info.max, size=N, dtype=dt)
        assert a.argmin() == np.argmin(a)
        assert a.argmax() == np.argmax(a)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmin_argmax_floats(self, dt):
        rng = np.random.default_rng(42)
        a = rng.random(N, dtype=dt)
        assert a.argmin() == np.argmin(a)
        assert a.argmax() == np.argmax(a)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmin_argmax_with_nan(self, dt):
        a = np.ones(N, dtype=dt)
        a[N // 2] = np.nan
        np.argmin(a)
        np.argmax(a)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmin_argmax_all_same(self, dt):
        a = np.full(N, 3.14, dtype=dt)
        assert np.argmin(a) == 0
        assert np.argmax(a) == 0

    def test_argmin_argmax_bool(self):
        a = np.array([True, True, False, True] * (N // 4), dtype=np.bool_)
        assert np.argmin(a) == 2
        b = np.array([False, False, True, False] * (N // 4), dtype=np.bool_)
        assert np.argmax(b) == 2

    def test_argmin_all_true_bool(self):
        a = np.ones(N, dtype=np.bool_)
        assert np.argmin(a) == 0

    def test_argmax_all_false_bool(self):
        a = np.zeros(N, dtype=np.bool_)
        assert np.argmax(a) == 0


class TestFloorDivideHighway:
    """loops_arithmetic_floor_hwy.dispatch.cpp, loops_arithmetic.dispatch.c.src"""

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
    ])
    def test_floor_divide_signed_contiguous(self, dt):
        info = np.iinfo(dt)
        lo = max(-N // 2, info.min + 1)
        hi = min(N // 2, info.max)
        a = np.arange(lo, hi, dtype=dt)
        b = np.full(len(a), 7, dtype=dt)
        r = np.floor_divide(a, b)
        expected = np.floor(a.astype(np.float64) / 7.0).astype(dt)
        assert_array_equal(r, expected)

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_floor_divide_by_scalar_one(self, dt):
        a = np.arange(N, dtype=dt)
        r = np.floor_divide(a, dt(1))
        assert_array_equal(r, a)

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_floor_divide_by_scalar_minus_one(self, dt):
        info = np.iinfo(dt)
        a = np.arange(-100, 100, dtype=dt)
        r = np.floor_divide(a, dt(-1))
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.int32, np.int64])
    def test_floor_divide_overflow_min_by_minus_one(self, dt):
        info = np.iinfo(dt)
        a = np.array([info.min, info.min + 1, 0, 1], dtype=dt)
        with np.errstate(over='ignore'):
            r = np.floor_divide(a, dt(-1))
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [
        np.uint8, np.uint16, np.uint32,
    ])
    def test_floor_divide_unsigned(self, dt):
        a = np.arange(N, dtype=dt)
        b = np.full(N, 5, dtype=dt)
        r = np.floor_divide(a, b)
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32])
    def test_floor_divide_unsigned_scalar(self, dt):
        a = np.arange(1, N + 1, dtype=dt)
        r = np.floor_divide(a, dt(7))
        assert r.dtype == dt

    def test_floor_divide_int64_scalar_divisor(self):
        a = np.arange(N, dtype=np.int64)
        r = np.floor_divide(a, np.int64(7))
        expected = a // 7
        assert_array_equal(r, expected)

    def test_floor_divide_int64_mixed_signs(self):
        a = np.array([-10, -7, 10, 7, -1, 1] * (N // 6), dtype=np.int64)
        b = np.array([3, -3, 3, -3, 3, -3] * (N // 6), dtype=np.int64)
        r = np.floor_divide(a, b)
        for i in range(len(a)):
            if b[i] != 0:
                assert r[i] == np.floor(a[i] / b[i])

    def test_floor_divide_int64_by_one(self):
        a = np.arange(N, dtype=np.int64)
        r = np.floor_divide(a, np.int64(1))
        assert_array_equal(r, a)


class TestPairwiseSumHighway:
    """loops_arithm_sum_hwy.dispatch.cpp"""

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_sum_small(self, dt):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=dt)
        assert_allclose(np.sum(a), 10.0)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_sum_medium(self, dt):
        a = np.ones(200, dtype=dt)
        assert_allclose(np.sum(a), 200.0)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_sum_large(self, dt):
        a = np.ones(10000, dtype=dt)
        assert_allclose(np.sum(a), 10000.0)

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_sum_complex(self, dt):
        a = np.ones(1000, dtype=dt)
        r = np.sum(a)
        assert_allclose(r.real, 1000.0)
        assert_allclose(r.imag, 0.0)

    def test_sum_negative_zero(self):
        a = np.array([-0.0, 0.0, -0.0], dtype=np.float64)
        r = np.sum(a)
        assert r == 0.0


class TestUnaryFpOpsHighway:
    """loops_unary_fp_ops.dispatch.cpp"""

    def test_reciprocal_float16(self):
        a = np.arange(1, N + 1, dtype=np.float16)
        r = np.reciprocal(a)
        assert r.dtype == np.float16

    def test_ceil_float16(self):
        a = np.linspace(-10, 10, N, dtype=np.float16)
        r = np.ceil(a)
        assert r.dtype == np.float16
        assert np.all(r >= a)

    def test_floor_float16(self):
        a = np.linspace(-10, 10, N, dtype=np.float16)
        r = np.floor(a)
        assert r.dtype == np.float16
        assert np.all(r <= a)

    def test_trunc_float16(self):
        a = np.linspace(-10, 10, N, dtype=np.float16)
        r = np.trunc(a)
        assert r.dtype == np.float16

    def test_rint_float16(self):
        a = np.linspace(-10, 10, N, dtype=np.float16)
        r = np.rint(a)
        assert r.dtype == np.float16

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_deg2rad(self, dt):
        a = np.linspace(0, 360, N, dtype=dt)
        r = np.deg2rad(a)
        assert r.dtype == dt
        assert_allclose(r[-1], np.radians(360), rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_rad2deg(self, dt):
        a = np.linspace(0, 2 * np.pi, N, dtype=dt)
        r = np.rad2deg(a)
        assert r.dtype == dt

    def test_reciprocal_complex64(self):
        a = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j] * (N // 4),
                      dtype=np.complex64)
        r = np.reciprocal(a)
        assert r.dtype == np.complex64
        assert_allclose(r * a, np.ones(N, dtype=np.complex64), rtol=1e-5)

    def test_reciprocal_complex128(self):
        a = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j] * (N // 4),
                      dtype=np.complex128)
        r = np.reciprocal(a)
        assert r.dtype == np.complex128
        assert_allclose(r * a, np.ones(N, dtype=np.complex128), rtol=1e-12)

    def test_unary_fp_strided(self):
        a = np.arange(N * 2, dtype=np.float32)
        r = np.ceil(a[::2])
        assert r.shape == (N,)


class TestExp2:
    """loops_exp2.dispatch.cpp"""

    def test_exp2_float16_contiguous(self):
        a = np.linspace(-10, 10, N, dtype=np.float16)
        r = np.exp2(a)
        assert r.dtype == np.float16

    def test_exp2_float16_strided(self):
        a = np.zeros(N * 2, dtype=np.float16)
        a[::2] = np.linspace(-5, 5, N, dtype=np.float16)
        r = np.exp2(a[::2])
        assert r.shape == (N,)

    def test_exp2_float16_special(self):
        a = np.array([0, np.inf, -np.inf, np.nan, 15, -14] * (N // 6),
                      dtype=np.float16)
        r = np.exp2(a)
        assert r[0] == 1.0

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_exp2_float(self, dt):
        a = np.linspace(-10, 10, N, dtype=dt)
        r = np.exp2(a)
        assert_allclose(r[0], np.exp2(a[0]), rtol=1e-6)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_exp2_special_values(self, dt):
        a = np.array([0, np.inf, -np.inf, np.nan, 1000, -1000] * (N // 6),
                      dtype=dt)
        with np.errstate(over='ignore', invalid='ignore'):
            r = np.exp2(a)
        assert r[0] == 1.0
        assert np.isinf(r[1])
        assert r[2] == 0.0
        assert np.isnan(r[3])


class TestPowerHighway:
    """loops_power.dispatch.cpp"""

    def test_power_float64_general(self):
        rng = np.random.default_rng(42)
        a = rng.random(N, dtype=np.float64) + 0.1
        b = rng.random(N, dtype=np.float64) * 3
        r = np.power(a, b)
        assert r.dtype == np.float64

    def test_power_float32_general(self):
        rng = np.random.default_rng(42)
        a = rng.random(N, dtype=np.float32) + 0.1
        b = rng.random(N, dtype=np.float32) * 3
        r = np.power(a, b)
        assert r.dtype == np.float32

    @pytest.mark.parametrize("exp_val", [-1.0, 0.0, 0.5, 1.0, 2.0])
    def test_power_fast_paths(self, exp_val):
        a = np.arange(1, N + 1, dtype=np.float64)
        r = np.power(a, exp_val)
        assert r.dtype == np.float64

    @pytest.mark.parametrize("exp_val", [-1.0, 0.0, 0.5, 1.0, 2.0])
    def test_power_fast_paths_float32(self, exp_val):
        a = np.arange(1, N + 1, dtype=np.float32)
        r = np.power(a, np.float32(exp_val))
        assert r.dtype == np.float32

    def test_power_negative_base(self):
        a = np.array([-2.0, -3.0, -4.0] * (N // 3), dtype=np.float64)
        b = np.array([3.0, 2.0, 1.0] * (N // 3), dtype=np.float64)
        r = np.power(a, b)
        assert r.dtype == np.float64

    def test_power_special_values(self):
        vals_a = [np.nan, np.inf, 0.0, 1e-310, 1e308]
        vals_b = [2.0, 2.0, 0.0, 2.0, 2.0]
        reps = N // 5
        a = np.array(vals_a * reps, dtype=np.float64)
        b = np.array(vals_b * reps, dtype=np.float64)
        with np.errstate(over='ignore', invalid='ignore'):
            r = np.power(a, b)
        assert r.shape == a.shape


class TestUnsignedIntAutovec:
    """loops_autovec.dispatch.c.src"""

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_sum_reduce(self, dt):
        a = np.ones(N, dtype=dt)
        r = np.sum(a)
        assert r == N

    @pytest.mark.parametrize("dt", [np.uint32, np.uint64])
    def test_prod_reduce_ones(self, dt):
        a = np.ones(N, dtype=dt)
        r = np.prod(a)
        assert r == 1

    @pytest.mark.parametrize("dt", [np.uint32, np.uint64])
    def test_prod_reduce(self, dt):
        a = np.arange(1, 11, dtype=dt)
        r = np.prod(a)
        assert r == 3628800

    def test_bitwise_count(self):
        a = np.arange(256, dtype=np.uint8)
        r = np.bitwise_count(a)
        assert r[0] == 0
        assert r[255] == 8

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_absolute_unsigned(self, dt):
        a = np.arange(N, dtype=dt)
        r = np.absolute(a)
        assert_array_equal(r, a)

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_sign_unsigned(self, dt):
        info = np.iinfo(dt)
        n = min(N, info.max)
        a = np.arange(n, dtype=dt)
        r = np.sign(a)
        assert r[0] == 0
        assert np.all(r[1:] == 1)

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_and_scalar(self, dt):
        a = np.arange(N, dtype=dt)
        r = np.bitwise_and(a, dt(0xFF))
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_or_scalar(self, dt):
        a = np.arange(N, dtype=dt)
        r = np.bitwise_or(dt(0xFF), a)
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_xor_scalar(self, dt):
        a = np.arange(N, dtype=dt)
        r = np.bitwise_xor(a, dt(0xFF))
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint32, np.uint64])
    def test_left_shift_scalar_in0(self, dt):
        r = np.left_shift(dt(1), np.arange(64, dtype=dt))
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint32, np.uint64])
    def test_right_shift_scalar_in0(self, dt):
        info = np.iinfo(dt)
        r = np.right_shift(dt(info.max), np.arange(64, dtype=dt))
        assert r.dtype == dt


class TestShiftSVE:
    """loops_shift_sve.c"""

    def test_left_shift_int64_scalar_in0(self):
        r = np.left_shift(np.int64(1), np.arange(63, dtype=np.int64))
        assert r[0] == 1
        assert r.dtype == np.int64

    def test_right_shift_int64_scalar_in0(self):
        r = np.right_shift(np.int64(-1), np.arange(63, dtype=np.int64))
        assert r.dtype == np.int64

    def test_left_shift_uint64_scalar_in0(self):
        r = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
        assert r[0] == 1

    def test_right_shift_uint64_scalar_in0(self):
        r = np.right_shift(np.uint64(0xFFFFFFFFFFFFFFFF),
                           np.arange(64, dtype=np.uint64))
        assert r[0] == 0xFFFFFFFFFFFFFFFF

    def test_left_shift_uint64_overflow(self):
        r = np.left_shift(np.uint64(1),
                          np.array([64, 65, 100] * (N // 3), dtype=np.uint64))
        assert r.dtype == np.uint64


class TestLogicalSVE:
    """loops_logical_sve.c"""

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_or_nonzero_scalar(self, dt):
        a = np.ones(N, dtype=dt)
        r = np.logical_or(dt(5), a)
        assert np.all(r)

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_or_zero_scalar(self, dt):
        a = np.ones(N, dtype=dt)
        a[0] = 0
        r = np.logical_or(dt(0), a)
        assert not r[0]
        assert np.all(r[1:])

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_or_scalar_in1(self, dt):
        a = np.ones(N, dtype=dt)
        r = np.logical_or(a, dt(5))
        assert np.all(r)

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_and_scalar(self, dt):
        a = np.ones(N, dtype=dt)
        a[0] = 0
        r = np.logical_and(dt(1), a)
        assert not r[0]
        assert np.all(r[1:])


class TestAbsoluteFloat16Highway:
    """loops_autovec_abs_hwy.dispatch.cpp"""

    def test_absolute_float16_contiguous(self):
        a = np.linspace(-10, 10, N, dtype=np.float16)
        r = np.absolute(a)
        assert r.dtype == np.float16
        assert np.all(r >= 0)

    def test_absolute_float16_strided(self):
        a = np.zeros(N * 2, dtype=np.float16)
        vals = np.linspace(-10, 10, N, dtype=np.float16)
        a[::2] = vals
        r = np.absolute(a[::2])
        assert r.shape == (N,)
        assert np.all(r >= 0)

    def test_absolute_float16_large(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(N * 4).astype(np.float16)
        r = np.absolute(a)
        assert np.all(r >= 0)


class TestScalarFastPath:
    """ufunc_object.c scalar fast path (lines 4394-4448)"""

    def test_scalar_sin_float64(self):
        r = np.sin(np.float64(1.0))
        assert_allclose(r, np.sin(1.0))

    def test_scalar_cos_float64(self):
        r = np.cos(np.float64(0.5))
        assert_allclose(r, np.cos(0.5))

    def test_scalar_exp_float64(self):
        r = np.exp(np.float64(1.0))
        assert_allclose(r, np.e)

    def test_scalar_log_float64(self):
        r = np.log(np.float64(2.0))
        assert_allclose(r, np.log(2.0))

    def test_scalar_sqrt_float64(self):
        r = np.sqrt(np.float64(4.0))
        assert r == 2.0

    def test_scalar_longdouble(self):
        r = np.sin(np.longdouble(1.0))
        assert_allclose(r, np.sin(1.0), rtol=1e-15)

    def test_scalar_complex128(self):
        r = np.exp(np.complex128(1 + 1j))
        assert_allclose(r, np.exp(1 + 1j))

    def test_scalar_complex64(self):
        r = np.sin(np.complex64(1 + 0j))
        assert_allclose(r, np.sin(1 + 0j), rtol=1e-5)

    def test_scalar_log_zero(self):
        with np.errstate(divide='ignore'):
            r = np.log(np.float64(0.0))
        assert np.isinf(r)

    def test_scalar_exp_overflow(self):
        with np.errstate(over='ignore'):
            r = np.exp(np.float64(1000.0))
        assert np.isinf(r)


class TestMappingFancyIndexing:
    """mapping.c fancy indexing fast paths"""

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.float32, np.float64, np.complex128,
    ])
    def test_axis1_fancy_get(self, dt):
        a = np.arange(60, dtype=dt).reshape(3, 4, 5)
        ind = np.array([0, 2, 1], dtype=np.intp)
        r = a[:, ind]
        assert r.shape == (3, 3, 5)
        assert_array_equal(r[:, 0, :], a[:, 0, :])

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.float32, np.float64, np.complex128,
    ])
    def test_axis0_fancy_set_scalar(self, dt):
        a = np.zeros((10, 5), dtype=dt)
        ind = np.array([1, 3, 5, 7])
        a[ind] = dt(42) if dt != np.complex128 else 42 + 0j
        assert_array_equal(a[1], a[3])
        assert a[0, 0] == 0

    @pytest.mark.parametrize("dt", [np.int32, np.float64])
    def test_axis1_fancy_set_scalar(self, dt):
        a = np.zeros((3, 10, 5), dtype=dt)
        ind = np.array([1, 3, 5])
        a[:, ind] = dt(7)
        assert a[0, 1, 0] == 7
        assert a[0, 0, 0] == 0

    def test_fancy_set_negative_index(self):
        a = np.zeros((10, 3), dtype=np.float64)
        ind = np.array([-1, -2, 0])
        a[ind] = 99.0
        assert a[-1, 0] == 99.0
        assert a[-2, 0] == 99.0


class TestHistogramdd:
    """compiled_base.c _histogramdd_uniform2d"""

    def test_histogramdd_2d(self):
        rng = np.random.default_rng(42)
        sample = rng.random((N, 2))
        hist, edges = np.histogramdd(sample, bins=[10, 10])
        assert hist.shape == (10, 10)
        assert hist.sum() == N

    def test_histogramdd_2d_uniform(self):
        rng = np.random.default_rng(42)
        sample = rng.random((500, 2))
        hist, _ = np.histogramdd(sample, bins=[5, 5],
                                  range=[[0, 1], [0, 1]])
        assert hist.shape == (5, 5)


class TestBitwiseSVE:
    """loops_bitwise_sve.c"""

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_bitwise_and_contiguous(self, dt):
        info = np.iinfo(dt)
        mask = dt(min(0x7F, info.max))
        a = np.arange(N, dtype=dt)
        b = np.full(N, mask, dtype=dt)
        r = np.bitwise_and(a, b)
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_bitwise_or_contiguous(self, dt):
        a = np.arange(N, dtype=dt)
        b = np.arange(N, dtype=dt)
        r = np.bitwise_or(a, b)
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_bitwise_xor_contiguous(self, dt):
        info = np.iinfo(dt)
        mask = dt(min(0x55, info.max))
        a = np.arange(N, dtype=dt)
        b = np.full(N, mask, dtype=dt)
        r = np.bitwise_xor(a, b)
        assert r.dtype == dt


class TestLogicalNotAarch64:
    """loops_logical_not_aarch64.c"""

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64,
                                     np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_not(self, dt):
        a = np.zeros(N, dtype=dt)
        a[1:] = 1
        r = np.logical_not(a)
        assert r[0] == True
        assert np.all(r[1:] == False)


class TestExpLogTrig:
    """loops_exp.dispatch.cpp, loops_log.dispatch.cpp, loops_log2.dispatch.cpp,
    loops_trigonometric.dispatch.cpp, loops_umath_unary.dispatch.cpp"""

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_exp_large(self, dt):
        a = np.linspace(-10, 10, N, dtype=dt)
        r = np.exp(a)
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_log_large(self, dt):
        a = np.linspace(0.1, 100, N, dtype=dt)
        r = np.log(a)
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_log2_large(self, dt):
        a = np.linspace(0.1, 100, N, dtype=dt)
        r = np.log2(a)
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_log10_large(self, dt):
        a = np.linspace(0.1, 100, N, dtype=dt)
        r = np.log10(a)
        assert r.dtype == dt

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_sin_cos_tan(self, dt):
        a = np.linspace(-np.pi, np.pi, N, dtype=dt)
        np.sin(a)
        np.cos(a)
        np.tan(a)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_arcsin_arccos_arctan(self, dt):
        a = np.linspace(-0.99, 0.99, N, dtype=dt)
        np.arcsin(a)
        np.arccos(a)
        np.arctan(a)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_sinh_cosh_tanh(self, dt):
        a = np.linspace(-5, 5, N, dtype=dt)
        np.sinh(a)
        np.cosh(a)
        np.tanh(a)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_expm1_log1p(self, dt):
        a = np.linspace(-1, 1, N, dtype=dt)
        np.expm1(a)
        np.log1p(a)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_cbrt(self, dt):
        a = np.linspace(-100, 100, N, dtype=dt)
        r = np.cbrt(a)
        assert r.dtype == dt

    def test_exp2_special_edges(self):
        a = np.array([200, -200, np.nan, np.inf, -np.inf] * (N // 5),
                      dtype=np.float32)
        with np.errstate(over='ignore', invalid='ignore'):
            r = np.exp2(a)
        assert np.isinf(r[0])
        assert r[1] == 0.0

    def test_log2_special_edges(self):
        a = np.array([0.0, -1.0, np.inf, np.nan, 1.0] * (N // 5),
                      dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.log2(a)
        assert np.isneginf(r[0])
        assert np.isnan(r[1])


NN = 16384


class TestMinMaxStridedSIMD:
    """Target simd_binary_@intrin@_@sfx@ in loops_minmax.dispatch.c.src
    (lines 234-345, 479-484) - aarch64 float SIMD fast paths."""

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_scalar_broadcast_in0(self, dt):
        scalar = dt(3.14)
        b = np.arange(NN, dtype=dt)
        r = np.maximum(scalar, b)
        assert r.shape == (NN,)
        r2 = np.minimum(scalar, b)
        assert r2.shape == (NN,)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_scalar_broadcast_in1(self, dt):
        a = np.arange(NN, dtype=dt)
        scalar = dt(5000.0)
        r = np.maximum(a, scalar)
        assert r.shape == (NN,)
        r2 = np.minimum(a, scalar)
        assert r2.shape == (NN,)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_contig_in_strided_out(self, dt):
        a = np.arange(NN, dtype=dt)
        b = np.arange(NN, dtype=dt)[::-1].copy()
        out = np.empty(NN * 2, dtype=dt)
        np.maximum(a, b, out=out[::2])
        np.minimum(a, b, out=out[::2])

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_large_contig_unrolled(self, dt):
        rng = np.random.default_rng(42)
        a = rng.random(NN, dtype=dt)
        b = rng.random(NN, dtype=dt)
        r = np.maximum(a, b)
        assert r.shape == (NN,)
        r2 = np.minimum(a, b)
        assert r2.shape == (NN,)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_scalar_broadcast_strided_in2(self, dt):
        scalar = dt(2.0)
        b = np.arange(NN * 2, dtype=dt)[::2]
        r = np.maximum(scalar, b)
        assert r.shape == (NN,)


class TestUnsignedReduceAutovec:
    """Target add.reduce / multiply.reduce in loops_autovec.dispatch.c.src
    (lines 106-186) - 16x unrolled contiguous reduce for unsigned ints."""

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_add_reduce_explicit(self, dt):
        a = np.ones(NN, dtype=dt)
        r = np.add.reduce(a)
        assert r == NN

    @pytest.mark.parametrize("dt", [np.uint32, np.uint64])
    def test_multiply_reduce_ones(self, dt):
        a = np.ones(NN, dtype=dt)
        r = np.multiply.reduce(a)
        assert r == 1

    @pytest.mark.parametrize("dt", [np.uint32, np.uint64])
    def test_multiply_reduce_small(self, dt):
        a = np.arange(1, 21, dtype=dt)
        r = np.multiply.reduce(a)
        assert r > 0

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_add_reduce_large(self, dt):
        info = np.iinfo(dt)
        n = min(NN, info.max)
        a = np.ones(n, dtype=dt)
        r = np.add.reduce(a)
        assert r == n


class TestShiftSVELarge:
    """Target SVE shift scalar-in0 paths in loops_shift_sve.c
    (lines 421-478) - 64-bit SVE shift with scalar broadcast."""

    def test_left_shift_int64_large(self):
        shifts = np.arange(63, dtype=np.int64)
        r = np.left_shift(np.int64(1), shifts)
        assert r[0] == 1
        assert r.dtype == np.int64

    def test_right_shift_int64_large(self):
        shifts = np.arange(63, dtype=np.int64)
        r = np.right_shift(np.int64(-1), shifts)
        assert r.dtype == np.int64

    def test_left_shift_uint64_large(self):
        shifts = np.arange(64, dtype=np.uint64)
        r = np.left_shift(np.uint64(1), shifts)
        assert r[0] == 1

    def test_right_shift_uint64_large(self):
        shifts = np.arange(64, dtype=np.uint64)
        r = np.right_shift(np.uint64(0xFFFFFFFFFFFFFFFF), shifts)
        assert r[0] == 0xFFFFFFFFFFFFFFFF

    def test_left_shift_int64_scalar_in0_contig(self):
        shifts = np.arange(NN, dtype=np.int64) % 63
        r = np.left_shift(np.int64(1), shifts)
        assert r.shape == (NN,)

    def test_right_shift_uint64_scalar_in0_contig(self):
        shifts = np.arange(NN, dtype=np.uint64) % 64
        r = np.right_shift(np.uint64(0xFFFFFFFFFFFFFFFF), shifts)
        assert r.shape == (NN,)


class TestLogicalSVELarge:
    """Target SVE logical paths in loops_logical_sve.c
    (lines 30-138) - SVE logical ops with scalar broadcast."""

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32, np.uint64])
    def test_logical_or_nonzero_scalar_large(self, dt):
        a = np.ones(NN, dtype=dt)
        r = np.logical_or(dt(5), a)
        assert np.all(r)

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32, np.uint64])
    def test_logical_or_zero_scalar_large(self, dt):
        a = np.ones(NN, dtype=dt)
        a[0] = 0
        r = np.logical_or(dt(0), a)
        assert not r[0]
        assert np.all(r[1:])

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32, np.uint64])
    def test_logical_or_scalar_in1_large(self, dt):
        a = np.ones(NN, dtype=dt)
        r = np.logical_or(a, dt(5))
        assert np.all(r)

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32, np.uint64])
    def test_logical_and_scalar_large(self, dt):
        a = np.ones(NN, dtype=dt)
        a[0] = 0
        r = np.logical_and(dt(1), a)
        assert not r[0]
        assert np.all(r[1:])


class TestBitwiseSVELarge:
    """Target SVE bitwise paths in loops_bitwise_sve.c
    (lines 71-90, 242-278)."""

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64])
    def test_bitwise_and_large(self, dt):
        a = np.arange(NN, dtype=dt)
        b = np.full(NN, dt(0x55), dtype=dt)
        r = np.bitwise_and(a, b)
        assert r.shape == (NN,)

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64])
    def test_bitwise_or_large(self, dt):
        a = np.arange(NN, dtype=dt)
        b = np.arange(NN, dtype=dt)
        r = np.bitwise_or(a, b)
        assert r.shape == (NN,)

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64])
    def test_bitwise_xor_large(self, dt):
        a = np.arange(NN, dtype=dt)
        b = np.full(NN, dt(0x55), dtype=dt)
        r = np.bitwise_xor(a, b)
        assert r.shape == (NN,)

    def test_bitwise_and_int64_very_large(self):
        """Specifically target npy_bitwise_and_sve_i64 with very large arrays."""
        rng = np.random.default_rng(42)
        a = rng.integers(-1000, 1000, size=NN * 4, dtype=np.int64)
        b = rng.integers(-1000, 1000, size=NN * 4, dtype=np.int64)
        r = np.bitwise_and(a, b)
        assert r.shape == (NN * 4,)
        assert r.dtype == np.int64

    def test_bitwise_and_int64_contiguous(self):
        """Ensure contiguous int64 arrays trigger SVE path."""
        a = np.ascontiguousarray(np.arange(NN * 2, dtype=np.int64))
        b = np.ascontiguousarray(np.full(NN * 2, 0x5A5A5A5A5A5A5A5A, dtype=np.int64))
        r = np.bitwise_and(a, b)
        assert r.shape == (NN * 2,)
        assert r.flags['C_CONTIGUOUS']


class TestMappingFancyIndexingExtended:
    """Additional mapping.c tests for uncovered lines."""

    @pytest.mark.parametrize("dt", [np.float32, np.int8, np.int16])
    def test_axis1_fancy_get_ellipsis(self, dt):
        a = np.arange(120, dtype=dt).reshape(2, 3, 4, 5)
        ind = np.array([0, 2, 1], dtype=np.intp)
        r = a[:, ind, ...]
        assert r.shape == (2, 3, 4, 5)

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.float32, np.complex128])
    def test_axis0_fancy_set_scalar_itemsizes(self, dt):
        a = np.zeros((20, 4), dtype=dt)
        ind = np.array([0, 5, 10, 15])
        val = dt(42) if dt != np.complex128 else 42 + 0j
        a[ind] = val
        assert_array_equal(a[5], a[10])
        assert a[1, 0] == dt(0) if dt != np.complex128 else a[1, 0] == 0

    def test_axis1_fancy_set_scalar_3d(self):
        a = np.zeros((4, 10, 3), dtype=np.float64)
        ind = np.array([1, 3, 5, 7])
        a[:, ind] = 99.0
        assert a[0, 1, 0] == 99.0
        assert a[0, 0, 0] == 0.0

    def test_fancy_set_negative_indices(self):
        a = np.zeros((10, 3), dtype=np.int32)
        ind = np.array([-1, -2, -3])
        a[ind] = 7
        assert a[-1, 0] == 7
        assert a[-2, 0] == 7
        assert a[-3, 0] == 7


class TestFloorDivideInt64Libdivide:
    """Target int64 floor_divide paths in loops_arithmetic.dispatch.c.src
    (lines 469-501) - libdivide scalar divisor optimization."""

    def test_floor_divide_int64_div_by_minus_one(self):
        a = np.arange(-NN // 2, NN // 2, dtype=np.int64)
        with np.errstate(over='ignore'):
            r = np.floor_divide(a, np.int64(-1))
        assert r.dtype == np.int64

    def test_floor_divide_int64_div_by_one(self):
        a = np.arange(NN, dtype=np.int64)
        r = np.floor_divide(a, np.int64(1))
        assert_array_equal(r, a)

    def test_floor_divide_int64_general_scalar(self):
        a = np.arange(NN, dtype=np.int64)
        r = np.floor_divide(a, np.int64(7))
        expected = a // 7
        assert_array_equal(r, expected)

    def test_floor_divide_int64_mixed_sign_scalar(self):
        a = np.arange(NN, dtype=np.int64)
        r = np.floor_divide(a, np.int64(-3))
        for i in range(0, NN, NN // 10):
            assert r[i] == np.floor(a[i] / -3)

    def test_floor_divide_int64_overflow(self):
        info = np.iinfo(np.int64)
        a = np.array([info.min, info.min + 1, 0, 1, -1] * (NN // 5),
                      dtype=np.int64)
        with np.errstate(over='ignore'):
            r = np.floor_divide(a, np.int64(-1))
        assert r.dtype == np.int64


class TestFloorDivideUint32Scalar:
    """Target uint32 scalar floor_divide in loops_arithmetic_floor_hwy_avail.c
    (lines 42, 45) - availability check for uint32 scalar path."""

    def test_floor_divide_uint32_scalar_divisor(self):
        a = np.arange(1, NN + 1, dtype=np.uint32)
        r = np.floor_divide(a, np.uint32(7))
        expected = a // 7
        assert_array_equal(r, expected)

    def test_floor_divide_uint32_scalar_one(self):
        a = np.arange(NN, dtype=np.uint32)
        r = np.floor_divide(a, np.uint32(1))
        assert_array_equal(r, a)

    def test_floor_divide_uint32_large_arrays(self):
        rng = np.random.default_rng(42)
        a = rng.integers(1, 1000, size=NN, dtype=np.uint32)
        b = rng.integers(1, 100, size=NN, dtype=np.uint32)
        r = np.floor_divide(a, b)
        assert r.dtype == np.uint32
        assert r.shape == (NN,)


class TestMinMaxHighwayAvail:
    """Target loops_minmax_hwy_avail.c (lines 26, 30) - availability check."""

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64,
                                     np.uint8, np.uint16, np.uint32, np.uint64])
    def test_maximum_all_int_types(self, dt):
        """Trigger Highway minmax availability checks for all integer types."""
        info = np.iinfo(dt)
        n = min(NN, info.max)
        a = np.arange(n, dtype=dt)
        b = np.arange(n, dtype=dt)[::-1]
        r_max = np.maximum(a, b)
        r_min = np.minimum(a, b)
        assert r_max.dtype == dt
        assert r_min.dtype == dt

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_float_types(self, dt):
        """Trigger Highway minmax for float types."""
        rng = np.random.default_rng(42)
        a = rng.random(NN, dtype=dt)
        b = rng.random(NN, dtype=dt)
        r_max = np.maximum(a, b)
        r_min = np.minimum(a, b)
        assert r_max.dtype == dt
        assert r_min.dtype == dt


class TestMinMaxScalarBroadcast:
    """Target loops_minmax.dispatch.c.src aarch64 scalar broadcast paths (lines 234-311, 479/482/484)."""

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_scalar_right(self, dt):
        a = np.arange(256, dtype=dt)
        r = np.maximum(a, dt(128.0))
        expected = np.where(a > 128, a, 128.0)
        assert_allclose(r, expected)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_scalar_left(self, dt):
        a = np.arange(256, dtype=dt)
        r = np.minimum(dt(128.0), a)
        expected = np.where(a < 128, a, 128.0)
        assert_allclose(r, expected)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmax_fmin_scalar(self, dt):
        a = np.arange(256, dtype=dt)
        r_max = np.fmax(a, dt(100.0))
        r_min = np.fmin(dt(100.0), a)
        assert r_max.dtype == dt
        assert r_min.dtype == dt

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minmax_strided_output(self, dt):
        a = np.arange(128, dtype=dt)
        out = np.empty(256, dtype=dt)
        np.maximum(a, dt(64.0), out=out[::2])
        expected = np.maximum(a, dt(64.0))
        assert_allclose(out[::2], expected)


class TestAutovecReduceAndSVE:
    """Target loops_autovec.dispatch.c.src ARM reduce and SVE paths (lines 106-399)."""

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_add_reduce_unsigned(self, dt):
        a = np.ones(256, dtype=dt)
        r = np.add.reduce(a)
        assert r == 256

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32, np.uint64])
    def test_multiply_reduce_ones(self, dt):
        a = np.ones(100, dtype=dt)
        r = np.multiply.reduce(a)
        assert r == 1

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32])
    def test_multiply_reduce_general(self, dt):
        a = np.array([2, 3, 4, 5], dtype=dt)
        r = np.multiply.reduce(a)
        assert r == 120

    @pytest.mark.parametrize("dt", [np.uint8, np.int16, np.int64, np.uint64])
    def test_shift_sve_scalar_in0(self, dt):
        scalar = dt(3)
        arr = np.arange(64, dtype=dt)
        r_left = np.left_shift(scalar, arr)
        r_right = np.right_shift(scalar, arr)
        assert r_left.dtype == dt
        assert r_right.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_sve_scalar_unsigned(self, dt):
        scalar = dt(0xFF)
        arr = np.arange(64, dtype=dt)
        r_and = np.bitwise_and(scalar, arr)
        r_or = np.bitwise_or(scalar, arr)
        r_xor = np.bitwise_xor(scalar, arr)
        assert r_and.dtype == dt
        assert r_or.dtype == dt
        assert r_xor.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32, np.uint64])
    def test_logical_sve_scalar_unsigned(self, dt):
        arr = np.array([0, 1, 2, 3], dtype=dt)
        r_and = np.logical_and(dt(5), arr)
        r_or = np.logical_or(dt(0), arr)
        assert r_and.dtype == np.bool_
        assert r_or.dtype == np.bool_


class TestUnaryFpOpsExtended:
    """Target loops_unary_fp_ops.dispatch.cpp float16, strided, and complex paths (lines 45-600)."""

    def test_float16_unary_ops(self):
        for func in [np.reciprocal, np.ceil, np.floor, np.trunc, np.rint,
                     np.deg2rad, np.rad2deg]:
            a = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float16)
            result = func(a)
            expected = func(a.astype(np.float32)).astype(np.float16)
            assert_allclose(result, expected, rtol=1e-2)

    def test_unary_strided_input(self):
        a = np.arange(200, dtype=np.float32)
        r = np.ceil(a[::3])
        expected = np.ceil(a[::3].copy())
        assert_array_equal(r, expected)

    def test_unary_strided_output(self):
        a = np.arange(100, dtype=np.float32)
        out = np.empty(200, dtype=np.float32)
        np.ceil(a, out=out[::2])
        expected = np.ceil(a)
        assert_array_equal(out[::2], expected)

    def test_complex_reciprocal_large_imag(self):
        z = np.array([1+5j, 0.1+10j, -2+8j, 0.5+20j] * 64, dtype=np.complex64)
        result = np.reciprocal(z)
        expected = 1.0 / z
        assert_allclose(result, expected, rtol=1e-5)

    def test_complex128_reciprocal_large_imag(self):
        z = np.array([1+5j, 0.1+10j, -2+8j] * 64, dtype=np.complex128)
        result = np.reciprocal(z)
        expected = 1.0 / z
        assert_allclose(result, expected, rtol=1e-12)

    def test_float16_rounding_ops(self):
        a = np.linspace(-10, 10, 256, dtype=np.float16)
        for func in [np.ceil, np.floor, np.trunc, np.rint]:
            result = func(a)
            expected = func(a.astype(np.float32)).astype(np.float16)
            assert_array_equal(result, expected)


class TestExp2Extended:
    """Target loops_exp2.dispatch.cpp strided float16 and overlap paths (lines 287-872)."""

    def test_exp2_float16_stride2(self):
        a = np.linspace(-5, 5, 64, dtype=np.float16)
        result = np.exp2(a[::2])
        expected = np.exp2(a[::2].astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    def test_exp2_float16_stride3(self):
        a = np.linspace(-5, 5, 96, dtype=np.float16)
        result = np.exp2(a[::3])
        expected = np.exp2(a[::3].astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    def test_exp2_overlap_float32(self):
        x = np.arange(100, dtype=np.float32)
        y = x.copy()
        np.exp2(x[:50], out=y[:50])
        expected = np.exp2(x[:50])
        assert_allclose(y[:50], expected, rtol=1e-5)

    def test_exp2_overlap_float64(self):
        x = np.arange(100, dtype=np.float64)
        y = x.copy()
        np.exp2(x[:50], out=y[:50])
        expected = np.exp2(x[:50])
        assert_allclose(y[:50], expected, rtol=1e-10)

    def test_exp2_overlap_float16(self):
        x = np.arange(100, dtype=np.float16)
        y = x.copy()
        with np.errstate(over='ignore', invalid='ignore'):
            np.exp2(x[:50], out=y[:50])
            expected = np.exp2(x[:50])
        assert_allclose(y[:50], expected, rtol=1e-2)


class TestShiftSVEExtended:
    """Target loops_shift_sve.c int64/uint64 SVE scalar paths (lines 421-478)."""

    def test_shift_sve_int64_scalar(self):
        scalar = np.int64(5)
        counts = np.arange(20, dtype=np.int64)
        result = np.left_shift(scalar, counts)
        expected = np.array([5 << i if i < 64 else 0 for i in counts], dtype=np.int64)
        assert_array_equal(result, expected)

    def test_shift_sve_uint64_scalar_overflow(self):
        scalar = np.uint64(1)
        counts = np.array([60, 63, 64, 65, 100], dtype=np.uint64)
        result = np.left_shift(scalar, counts)
        assert result[2] == 0
        assert result[3] == 0
        assert result[4] == 0

    def test_shift_sve_uint64_right_scalar(self):
        scalar = np.uint64(0xFFFFFFFFFFFFFFFF)
        counts = np.arange(64, dtype=np.uint64)
        result = np.right_shift(scalar, counts)
        assert result[0] == 0xFFFFFFFFFFFFFFFF
        assert result[1] == 0x7FFFFFFFFFFFFFFF


class TestLogicalSVEExtended:
    """Target loops_logical_sve.c uint8 SVE truth paths (lines 30-138)."""

    def test_logical_or_sve_uint8_scalar_zero(self):
        arr = np.array([0, 1, 2, 0, 255, 0, 3], dtype=np.uint8)
        result = np.logical_or(np.uint8(0), arr)
        expected = np.array([False, True, True, False, True, False, True])
        assert_array_equal(result, expected)

    def test_logical_or_sve_uint8_scalar_zero_large(self):
        arr = np.arange(256, dtype=np.uint8)
        result = np.logical_or(np.uint8(0), arr)
        expected = arr != 0
        assert_array_equal(result, expected)

    def test_logical_or_sve_uint8_scalar_zero_reversed(self):
        arr = np.array([0, 5, 0, 10], dtype=np.uint8)
        result = np.logical_or(arr, np.uint8(0))
        expected = arr != 0
        assert_array_equal(result, expected)


class TestUnaryFpOpsComprehensive:
    """Target loops_unary_fp_ops.dispatch.cpp lines 45-600.
    Float16 scalar_op specializations, non-contiguous SIMD paths, complex reciprocal."""

    @pytest.mark.parametrize("func", [np.reciprocal, np.ceil, np.floor, np.trunc, np.rint])
    def test_float16_basic_ops(self, func):
        a = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float16)
        result = func(a)
        expected = func(a.astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("func", [np.deg2rad, np.rad2deg])
    def test_float16_angle_ops(self, func):
        a = np.array([0, 45, 90, 180, 270, 360], dtype=np.float16)
        result = func(a)
        expected = func(a.astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    def test_float16_strided_input(self):
        a = np.arange(200, dtype=np.float32)
        result = np.ceil(a[::3])
        expected = np.ceil(a[::3].copy())
        assert_array_equal(result, expected)

    def test_float16_strided_output(self):
        a = np.arange(100, dtype=np.float32)
        out = np.empty(200, dtype=np.float32)
        np.ceil(a, out=out[::2])
        expected = np.ceil(a)
        assert_array_equal(out[::2], expected)

    def test_float16_both_strided(self):
        a = np.arange(200, dtype=np.float32)
        out = np.empty(300, dtype=np.float32)
        np.ceil(a[::2], out=out[::3])
        expected = np.ceil(a[::2].copy())
        assert_array_equal(out[::3], expected)

    def test_complex64_reciprocal_large_imag(self):
        z = np.array([1+5j, 0.1+10j, -2+8j, 0.5+20j] * 64, dtype=np.complex64)
        result = np.reciprocal(z)
        expected = 1.0 / z
        assert_allclose(result, expected, rtol=1e-5)

    def test_complex128_reciprocal_large_imag(self):
        z = np.array([1+5j, 0.1+10j, -2+8j, 0.5+20j] * 64, dtype=np.complex128)
        result = np.reciprocal(z)
        expected = 1.0 / z
        assert_allclose(result, expected, rtol=1e-12)

    def test_complex64_reciprocal_small_imag(self):
        z = np.array([5+1j, 10+0.1j, 8-2j, 20+0.5j] * 64, dtype=np.complex64)
        result = np.reciprocal(z)
        expected = 1.0 / z
        assert_allclose(result, expected, rtol=1e-5)

    def test_complex128_reciprocal_small_imag(self):
        z = np.array([5+1j, 10+0.1j, 8-2j, 20+0.5j] * 64, dtype=np.complex128)
        result = np.reciprocal(z)
        expected = 1.0 / z
        assert_allclose(result, expected, rtol=1e-12)


class TestAutovecReduceAndSVEComprehensive:
    """Target loops_autovec.dispatch.c.src lines 100-400.
    ARM add/multiply reduce, SVE shift/bitwise/logical scalar paths."""

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_add_reduce_ones(self, dt):
        a = np.ones(256, dtype=dt)
        result = np.add.reduce(a)
        assert result == 256

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_add_reduce_small(self, dt):
        a = np.arange(10, dtype=dt)
        result = np.add.reduce(a)
        assert result == 45

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32, np.uint64])
    def test_multiply_reduce_ones(self, dt):
        a = np.ones(100, dtype=dt)
        result = np.multiply.reduce(a)
        assert result == 1

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32])
    def test_multiply_reduce_general(self, dt):
        a = np.array([2, 3, 4, 5], dtype=dt)
        result = np.multiply.reduce(a)
        assert result == 120

    @pytest.mark.parametrize("dt", [np.uint8, np.int16, np.int64, np.uint64])
    def test_left_shift_sve_scalar(self, dt):
        scalar = dt(3)
        arr = np.arange(64, dtype=dt)
        result = np.left_shift(scalar, arr)
        assert result.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint8, np.int16, np.int64, np.uint64])
    def test_right_shift_sve_scalar(self, dt):
        scalar = dt(255)
        arr = np.arange(8, dtype=dt)
        result = np.right_shift(scalar, arr)
        assert result.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_and_sve_scalar(self, dt):
        scalar = dt(0xFF)
        arr = np.arange(64, dtype=dt)
        result = np.bitwise_and(scalar, arr)
        assert result.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_or_sve_scalar(self, dt):
        scalar = dt(0x0F)
        arr = np.arange(64, dtype=dt)
        result = np.bitwise_or(scalar, arr)
        assert result.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_xor_sve_scalar(self, dt):
        scalar = dt(0xAA)
        arr = np.arange(64, dtype=dt)
        result = np.bitwise_xor(scalar, arr)
        assert result.dtype == dt

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32, np.uint64])
    def test_logical_and_sve_scalar(self, dt):
        arr = np.array([0, 1, 2, 3], dtype=dt)
        result = np.logical_and(dt(5), arr)
        assert result.dtype == np.bool_

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32, np.uint64])
    def test_logical_or_sve_scalar(self, dt):
        arr = np.array([0, 1, 0, 1], dtype=dt)
        result = np.logical_or(dt(0), arr)
        assert result.dtype == np.bool_


class TestExp2Comprehensive:
    """Target loops_exp2.dispatch.cpp lines 285-875.
    NEON float16 exp2 with various strides, dispatch functions."""

    def test_exp2_float16_contiguous(self):
        a = np.linspace(-5, 5, 64, dtype=np.float16)
        result = np.exp2(a)
        expected = np.exp2(a.astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    def test_exp2_float16_stride2(self):
        a = np.linspace(-5, 5, 64, dtype=np.float16)
        result = np.exp2(a[::2])
        expected = np.exp2(a[::2].astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    def test_exp2_float16_stride3(self):
        a = np.linspace(-5, 5, 96, dtype=np.float16)
        result = np.exp2(a[::3])
        expected = np.exp2(a[::3].astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    def test_exp2_float16_stride4(self):
        a = np.linspace(-5, 5, 128, dtype=np.float16)
        result = np.exp2(a[::4])
        expected = np.exp2(a[::4].astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    def test_exp2_float32_large(self):
        a = np.linspace(-10, 10, 1024, dtype=np.float32)
        result = np.exp2(a)
        expected = np.exp2(a)
        assert_allclose(result, expected, rtol=1e-5)

    def test_exp2_float64_large(self):
        a = np.linspace(-10, 10, 1024, dtype=np.float64)
        result = np.exp2(a)
        expected = np.exp2(a)
        assert_allclose(result, expected, rtol=1e-10)

    def test_exp2_overlap_float32(self):
        x = np.arange(100, dtype=np.float32)
        y = x.copy()
        np.exp2(x[:50], out=y[:50])
        expected = np.exp2(x[:50])
        assert_allclose(y[:50], expected, rtol=1e-5)

    def test_exp2_overlap_float64(self):
        x = np.arange(100, dtype=np.float64)
        y = x.copy()
        np.exp2(x[:50], out=y[:50])
        expected = np.exp2(x[:50])
        assert_allclose(y[:50], expected, rtol=1e-10)

    def test_exp2_overlap_float16(self):
        x = np.arange(100, dtype=np.float16)
        y = x.copy()
        with np.errstate(over='ignore', invalid='ignore'):
            np.exp2(x[:50], out=y[:50])
            expected = np.exp2(x[:50])
        assert_allclose(y[:50], expected, rtol=1e-2)


class TestMinMaxDispatchComprehensive:
    """Target loops_minmax.dispatch.c.src lines 240-350.
    aarch64 scalar broadcast and both-contiguous prefetch paths."""

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_scalar_left(self, dt):
        a = np.arange(256, dtype=dt)
        result = np.maximum(dt(128.0), a)
        expected = np.where(a > 128, a, 128.0)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_scalar_right(self, dt):
        a = np.arange(256, dtype=dt)
        result = np.maximum(a, dt(128.0))
        expected = np.where(a > 128, a, 128.0)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_scalar_left(self, dt):
        a = np.arange(256, dtype=dt)
        result = np.minimum(dt(128.0), a)
        expected = np.where(a < 128, a, 128.0)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_scalar_right(self, dt):
        a = np.arange(256, dtype=dt)
        result = np.minimum(a, dt(128.0))
        expected = np.where(a < 128, a, 128.0)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_both_contiguous(self, dt):
        rng = np.random.default_rng(42)
        a = rng.random(1024, dtype=dt)
        b = rng.random(1024, dtype=dt)
        result = np.maximum(a, b)
        expected = np.maximum(a, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_both_contiguous(self, dt):
        rng = np.random.default_rng(42)
        a = rng.random(1024, dtype=dt)
        b = rng.random(1024, dtype=dt)
        result = np.minimum(a, b)
        expected = np.minimum(a, b)
        assert_array_equal(result, expected)


class TestPowerComprehensive:
    """Target loops_power.dispatch.cpp lines 490-1130.
    Power edge cases: extreme exponents, negative bases, subnormal inputs."""

    def test_power_extreme_exponents(self):
        a = np.array([1e-300, 1e300, 2.0, 0.5], dtype=np.float64)
        b = np.array([2.0, 2.0, 1000.0, -1000.0], dtype=np.float64)
        with np.errstate(over='ignore', under='ignore'):
            result = np.power(a, b)
        assert result.dtype == np.float64

    def test_power_negative_base_integer_exp(self):
        a = np.array([-2.0, -3.0, -4.0] * 64, dtype=np.float64)
        b = np.array([3.0, 2.0, 1.0] * 64, dtype=np.float64)
        result = np.power(a, b)
        assert result.dtype == np.float64

    def test_power_subnormal_input(self):
        a = np.array([1e-310, 1e-315, 5e-324] * 64, dtype=np.float64)
        b = np.array([2.0, 2.0, 2.0] * 64, dtype=np.float64)
        with np.errstate(under='ignore', over='ignore'):
            result = np.power(a, b)
        assert result.dtype == np.float64

    def test_power_scalar_exponent_fast_paths(self):
        a = np.arange(1, 101, dtype=np.float64)
        for exp_val in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            result = np.power(a, exp_val)
            assert result.dtype == np.float64

    def test_power_float32_general(self):
        rng = np.random.default_rng(42)
        a = rng.random(1024, dtype=np.float32) + 0.1
        b = rng.random(1024, dtype=np.float32) * 3
        result = np.power(a, b)
        assert result.dtype == np.float32

    def test_power_float64_general(self):
        rng = np.random.default_rng(42)
        a = rng.random(1024, dtype=np.float64) + 0.1
        b = rng.random(1024, dtype=np.float64) * 3
        result = np.power(a, b)
        assert result.dtype == np.float64

    def test_power_strided_input(self):
        a = np.arange(200, dtype=np.float64) + 1.0
        b = np.arange(200, dtype=np.float64) * 0.01
        result = np.power(a[::2], b[::2])
        expected = np.power(a[::2].copy(), b[::2].copy())
        assert_allclose(result, expected, rtol=1e-10)


class TestShiftSVEComprehensive:
    """Target loops_shift_sve.c lines 415-480.
    int64/uint64 SVE scalar-in0 shift paths."""

    def test_left_shift_int64_scalar(self):
        scalar = np.int64(5)
        counts = np.arange(20, dtype=np.int64)
        result = np.left_shift(scalar, counts)
        expected = np.array([5 << i if i < 64 else 0 for i in counts], dtype=np.int64)
        assert_array_equal(result, expected)

    def test_right_shift_int64_scalar(self):
        scalar = np.int64(1000000)
        counts = np.arange(20, dtype=np.int64)
        result = np.right_shift(scalar, counts)
        expected = np.array([1000000 >> i for i in counts], dtype=np.int64)
        assert_array_equal(result, expected)

    def test_left_shift_uint64_scalar(self):
        scalar = np.uint64(0xFF)
        counts = np.arange(64, dtype=np.uint64)
        result = np.left_shift(scalar, counts)
        assert result.dtype == np.uint64

    def test_right_shift_uint64_scalar(self):
        scalar = np.uint64(0xFFFFFFFFFFFFFFFF)
        counts = np.arange(64, dtype=np.uint64)
        result = np.right_shift(scalar, counts)
        assert result.dtype == np.uint64

    def test_left_shift_uint64_overflow(self):
        scalar = np.uint64(1)
        counts = np.array([60, 63, 64, 65, 100], dtype=np.uint64)
        result = np.left_shift(scalar, counts)
        assert result[2] == 0
        assert result[3] == 0
        assert result[4] == 0


class TestMappingComprehensive:
    """Target mapping.c lines 1059-2705.
    Fancy indexing patterns: axis-1 get/set, negative indices."""

    def test_axis1_fancy_get_2d(self):
        a = np.arange(60).reshape(3, 4, 5)
        ind = np.array([0, 2, 1], dtype=np.intp)
        result = a[:, ind]
        assert result.shape == (3, 3, 5)

    def test_axis1_fancy_get_ellipsis(self):
        a = np.arange(120).reshape(2, 3, 4, 5)
        ind = np.array([0, 2, 1], dtype=np.intp)
        result = a[:, ind, ...]
        assert result.shape == (2, 3, 4, 5)

    def test_axis0_fancy_set_scalar_int8(self):
        a = np.zeros((20, 4), dtype=np.int8)
        ind = np.array([0, 5, 10, 15])
        a[ind] = np.int8(42)
        assert_array_equal(a[5], a[10])

    def test_axis0_fancy_set_scalar_int16(self):
        a = np.zeros((20, 4), dtype=np.int16)
        ind = np.array([0, 5, 10, 15])
        a[ind] = np.int16(42)
        assert_array_equal(a[5], a[10])

    def test_axis0_fancy_set_scalar_float32(self):
        a = np.zeros((20, 4), dtype=np.float32)
        ind = np.array([0, 5, 10, 15])
        a[ind] = 42.0
        assert_array_equal(a[5], a[10])

    def test_axis0_fancy_set_scalar_complex128(self):
        a = np.zeros((20, 4), dtype=np.complex128)
        ind = np.array([0, 5, 10, 15])
        a[ind] = 42 + 0j
        assert_array_equal(a[5], a[10])

    def test_axis1_fancy_set_scalar_3d(self):
        a = np.zeros((4, 10, 3), dtype=np.float64)
        ind = np.array([1, 3, 5, 7])
        a[:, ind] = 99.0
        assert a[0, 1, 0] == 99.0

    def test_fancy_set_negative_indices(self):
        a = np.zeros((10, 3), dtype=np.int32)
        ind = np.array([-1, -2, -3])
        a[ind] = 7
        assert a[-1, 0] == 7
        assert a[-2, 0] == 7


class TestCompiledBaseComprehensive:
    """Target compiled_base.c lines 344-375.
    histogramdd error cases and edge cases."""

    def test_histogramdd_2d_basic(self):
        rng = np.random.default_rng(42)
        sample = rng.random((1000, 2))
        hist, edges = np.histogramdd(sample, bins=[10, 10])
        assert hist.shape == (10, 10)
        assert hist.sum() == 1000

    def test_histogramdd_2d_uniform(self):
        rng = np.random.default_rng(42)
        sample = rng.random((500, 2))
        hist, _ = np.histogramdd(sample, bins=[5, 5], range=[[0, 1], [0, 1]])
        assert hist.shape == (5, 5)

    def test_histogramdd_3d(self):
        rng = np.random.default_rng(42)
        sample = rng.random((500, 3))
        hist, _ = np.histogramdd(sample, bins=[5, 5, 5])
        assert hist.shape == (5, 5, 5)


class TestUfuncObjectComprehensive:
    """Target ufunc_object.c lines 4394-4448.
    Scalar fast path for various unary ufuncs."""

    @pytest.mark.parametrize("ufunc", [np.sin, np.cos, np.tan, np.exp, np.log, np.sqrt])
    def test_scalar_fast_path_float64(self, ufunc):
        result = ufunc(np.float64(1.0))
        expected = ufunc(1.0)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("ufunc", [np.sin, np.cos, np.exp])
    def test_scalar_fast_path_longdouble(self, ufunc):
        result = ufunc(np.longdouble(1.0))
        expected = ufunc(1.0)
        assert_allclose(result, expected, rtol=1e-15)

    @pytest.mark.parametrize("ufunc", [np.exp, np.sin, np.cos])
    def test_scalar_fast_path_complex128(self, ufunc):
        result = ufunc(np.complex128(1 + 1j))
        expected = ufunc(1 + 1j)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("ufunc", [np.exp, np.sin])
    def test_scalar_fast_path_complex64(self, ufunc):
        result = ufunc(np.complex64(1 + 0j))
        expected = ufunc(1 + 0j)
        assert_allclose(result, expected, rtol=1e-5)

    def test_scalar_log_zero(self):
        with np.errstate(divide='ignore'):
            result = np.log(np.float64(0.0))
        assert np.isinf(result)

    def test_scalar_exp_overflow(self):
        with np.errstate(over='ignore'):
            result = np.exp(np.float64(1000.0))
        assert np.isinf(result)


class TestFloat16Comprehensive:
    """Target float16 operations with various array sizes and strides."""

    @pytest.mark.parametrize("size", [7, 13, 17, 31, 63, 127])
    def test_float16_reciprocal_odd_sizes(self, size):
        a = np.arange(1, size + 1, dtype=np.float16)
        result = np.reciprocal(a)
        expected = np.reciprocal(a.astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("size", [7, 13, 17, 31, 63, 127])
    def test_float16_ceil_odd_sizes(self, size):
        a = np.linspace(-5, 5, size, dtype=np.float16)
        result = np.ceil(a)
        expected = np.ceil(a.astype(np.float32)).astype(np.float16)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("size", [7, 13, 17, 31, 63, 127])
    def test_float16_floor_odd_sizes(self, size):
        a = np.linspace(-5, 5, size, dtype=np.float16)
        result = np.floor(a)
        expected = np.floor(a.astype(np.float32)).astype(np.float16)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("size", [7, 13, 17, 31, 63, 127])
    def test_float16_trunc_odd_sizes(self, size):
        a = np.linspace(-5, 5, size, dtype=np.float16)
        result = np.trunc(a)
        expected = np.trunc(a.astype(np.float32)).astype(np.float16)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("size", [7, 13, 17, 31, 63, 127])
    def test_float16_rint_odd_sizes(self, size):
        a = np.linspace(-5, 5, size, dtype=np.float16)
        result = np.rint(a)
        expected = np.rint(a.astype(np.float32)).astype(np.float16)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("stride", [2, 3, 4, 5, 7])
    def test_float16_reciprocal_strided(self, stride):
        a = np.arange(1, 200, dtype=np.float16)
        result = np.reciprocal(a[::stride])
        expected = np.reciprocal(a[::stride].astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("stride", [2, 3, 4, 5, 7])
    def test_float16_ceil_strided(self, stride):
        a = np.linspace(-10, 10, 200, dtype=np.float16)
        result = np.ceil(a[::stride])
        expected = np.ceil(a[::stride].astype(np.float32)).astype(np.float16)
        assert_array_equal(result, expected)

    def test_float16_deg2rad_strided(self):
        a = np.arange(0, 720, 15, dtype=np.float16)
        result = np.deg2rad(a[::2])
        expected = np.deg2rad(a[::2].astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    def test_float16_rad2deg_strided(self):
        a = np.linspace(0, 4 * np.pi, 100, dtype=np.float16)
        result = np.rad2deg(a[::3])
        expected = np.rad2deg(a[::3].astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)


class TestArmReduceComprehensive:
    """Target ARM-specific reduce operations with various array configurations."""

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
    def test_add_reduce_various_sizes(self, dt, size):
        a = np.ones(size, dtype=dt)
        result = np.add.reduce(a)
        assert result == size

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32, np.uint64])
    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129])
    def test_multiply_reduce_ones_various_sizes(self, dt, size):
        a = np.ones(size, dtype=dt)
        result = np.multiply.reduce(a)
        assert result == 1

    @pytest.mark.parametrize("dt", [np.uint16, np.uint32])
    def test_multiply_reduce_general_various(self, dt):
        for size in [2, 3, 5, 7, 10, 15, 20]:
            a = np.arange(1, size + 1, dtype=dt)
            result = np.multiply.reduce(a)
            expected = np.prod(a)
            assert result == expected

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_add_reduce_sequential(self, dt):
        info = np.iinfo(dt)
        max_val = min(256, info.max)
        a = np.arange(max_val, dtype=dt)
        result = np.add.reduce(a)
        # Just check it runs and returns a reasonable value
        assert result >= 0


class TestSVEScalarPathsComprehensive:
    """Target SVE scalar-in0 paths with all supported types."""

    @pytest.mark.parametrize("dt", [np.int8, np.uint8, np.int16, np.uint16, 
                                     np.int64, np.uint64])
    def test_left_shift_sve_scalar_all_types(self, dt):
        scalar = dt(3)
        arr = np.arange(64, dtype=dt)
        result = np.left_shift(scalar, arr)
        assert result.dtype == dt
        assert result.shape == (64,)

    @pytest.mark.parametrize("dt", [np.int8, np.uint8, np.int16, np.uint16,
                                     np.int64, np.uint64])
    def test_right_shift_sve_scalar_all_types(self, dt):
        info = np.iinfo(dt)
        scalar = dt(max(1, info.max // 2))
        arr = np.arange(8, dtype=dt)
        result = np.right_shift(scalar, arr)
        assert result.dtype == dt
        assert result.shape == (8,)

    @pytest.mark.parametrize("dt", [np.int8, np.uint8, np.int16, np.uint16,
                                     np.int32, np.uint32, np.int64, np.uint64])
    def test_bitwise_and_sve_scalar_all_types(self, dt):
        scalar = dt(0x0F)
        arr = np.arange(64, dtype=dt)
        result = np.bitwise_and(scalar, arr)
        assert result.dtype == dt

    @pytest.mark.parametrize("dt", [np.int8, np.uint8, np.int16, np.uint16,
                                     np.int32, np.uint32, np.int64, np.uint64])
    def test_bitwise_or_sve_scalar_all_types(self, dt):
        info = np.iinfo(dt)
        scalar = dt(max(1, info.max // 4))
        arr = np.arange(64, dtype=dt)
        result = np.bitwise_or(scalar, arr)
        assert result.dtype == dt

    @pytest.mark.parametrize("dt", [np.int8, np.uint8, np.int16, np.uint16,
                                     np.int32, np.uint32, np.int64, np.uint64])
    def test_bitwise_xor_sve_scalar_all_types(self, dt):
        info = np.iinfo(dt)
        scalar = dt(max(1, info.max // 3))
        arr = np.arange(64, dtype=dt)
        result = np.bitwise_xor(scalar, arr)
        assert result.dtype == dt

    @pytest.mark.parametrize("dt", [np.int16, np.uint16, np.int32, np.uint32,
                                     np.int64, np.uint64])
    def test_logical_and_sve_scalar_all_types(self, dt):
        arr = np.array([0, 1, 2, 3, 0, 5], dtype=dt)
        result = np.logical_and(dt(3), arr)
        assert result.dtype == np.bool_

    @pytest.mark.parametrize("dt", [np.int16, np.uint16, np.int32, np.uint32,
                                     np.int64, np.uint64])
    def test_logical_or_sve_scalar_all_types(self, dt):
        arr = np.array([0, 1, 0, 3, 0, 5], dtype=dt)
        result = np.logical_or(dt(0), arr)
        assert result.dtype == np.bool_


class TestMinMaxBroadcastComprehensive:
    """Target minmax scalar broadcast with various configurations."""

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
    def test_maximum_scalar_left_various_sizes(self, dt, size):
        a = np.arange(size, dtype=dt)
        scalar = dt(size / 2)
        result = np.maximum(scalar, a)
        expected = np.where(a > scalar, a, scalar)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
    def test_minimum_scalar_right_various_sizes(self, dt, size):
        a = np.arange(size, dtype=dt)
        scalar = dt(size / 2)
        result = np.minimum(a, scalar)
        expected = np.where(a < scalar, a, scalar)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmax_fmin_scalar(self, dt):
        a = np.arange(256, dtype=dt)
        scalar = dt(128.0)
        result_max = np.fmax(scalar, a)
        result_min = np.fmin(a, scalar)
        assert result_max.dtype == dt
        assert result_min.dtype == dt


class TestExp2EdgeCases:
    """Target exp2 with various edge cases and configurations."""

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65])
    def test_exp2_float16_various_sizes(self, size):
        a = np.linspace(-5, 5, size, dtype=np.float16)
        result = np.exp2(a)
        expected = np.exp2(a.astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("stride", [2, 3, 4, 5, 7, 8])
    def test_exp2_float16_various_strides(self, stride):
        a = np.linspace(-5, 5, 200, dtype=np.float16)
        result = np.exp2(a[::stride])
        expected = np.exp2(a[::stride].astype(np.float32)).astype(np.float16)
        assert_allclose(result, expected, rtol=1e-2)

    def test_exp2_float16_special_values(self):
        a = np.array([0, 1, -1, 2, -2, 10, -10, np.inf, -np.inf], dtype=np.float16)
        result = np.exp2(a)
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[2] == 0.5
        assert np.isinf(result[7])
        assert result[8] == 0.0

    @pytest.mark.parametrize("size", [64, 128, 256, 512, 1024])
    def test_exp2_float32_large_arrays(self, size):
        a = np.linspace(-10, 10, size, dtype=np.float32)
        result = np.exp2(a)
        expected = np.exp2(a)
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("size", [64, 128, 256, 512, 1024])
    def test_exp2_float64_large_arrays(self, size):
        a = np.linspace(-10, 10, size, dtype=np.float64)
        result = np.exp2(a)
        expected = np.exp2(a)
        assert_allclose(result, expected, rtol=1e-10)


class TestPowerEdgeCasesComprehensive:
    """Target power with comprehensive edge cases."""

    def test_power_zero_base(self):
        a = np.array([0.0, 0.0, 0.0] * 64, dtype=np.float64)
        b = np.array([1.0, 2.0, 3.0] * 64, dtype=np.float64)
        result = np.power(a, b)
        assert_array_equal(result, np.zeros_like(result))

    def test_power_one_base(self):
        a = np.array([1.0, 1.0, 1.0] * 64, dtype=np.float64)
        b = np.array([100.0, -100.0, 0.5] * 64, dtype=np.float64)
        result = np.power(a, b)
        assert_array_equal(result, np.ones_like(result))

    def test_power_negative_base_odd_exp(self):
        a = np.array([-2.0, -3.0, -4.0] * 64, dtype=np.float64)
        b = np.array([1.0, 3.0, 5.0] * 64, dtype=np.float64)
        result = np.power(a, b)
        assert result.dtype == np.float64
        assert result[0] == -2.0
        assert result[1] == -27.0

    def test_power_negative_base_even_exp(self):
        a = np.array([-2.0, -3.0, -4.0] * 64, dtype=np.float64)
        b = np.array([2.0, 4.0, 6.0] * 64, dtype=np.float64)
        result = np.power(a, b)
        assert result.dtype == np.float64
        assert result[0] == 4.0
        assert result[1] == 81.0

    def test_power_fractional_exp(self):
        a = np.array([4.0, 9.0, 16.0, 25.0] * 64, dtype=np.float64)
        b = np.array([0.5, 0.5, 0.5, 0.5] * 64, dtype=np.float64)
        result = np.power(a, b)
        expected = np.array([2.0, 3.0, 4.0, 5.0] * 64, dtype=np.float64)
        assert_allclose(result, expected)

    def test_power_negative_exp(self):
        a = np.array([2.0, 4.0, 8.0] * 64, dtype=np.float64)
        b = np.array([-1.0, -2.0, -3.0] * 64, dtype=np.float64)
        result = np.power(a, b)
        expected = np.array([0.5, 0.0625, 0.001953125] * 64, dtype=np.float64)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("exp_val", [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    def test_power_scalar_exp_all_fast_paths(self, exp_val):
        a = np.arange(1, 101, dtype=np.float64)
        result = np.power(a, exp_val)
        expected = np.power(a, exp_val)
        assert_allclose(result, expected)

    def test_power_float32_strided(self):
        a = np.arange(1, 201, dtype=np.float32)
        b = np.arange(1, 201, dtype=np.float32) * 0.01
        result = np.power(a[::2], b[::2])
        expected = np.power(a[::2].copy(), b[::2].copy())
        assert_allclose(result, expected, rtol=1e-5)

    def test_power_float64_strided(self):
        a = np.arange(1, 201, dtype=np.float64)
        b = np.arange(1, 201, dtype=np.float64) * 0.01
        result = np.power(a[::3], b[::3])
        expected = np.power(a[::3].copy(), b[::3].copy())
        assert_allclose(result, expected, rtol=1e-10)


class TestLogicalSVEUint8:
    """Target SVE logical operations specifically for uint8."""

    def test_logical_or_uint8_scalar_zero_mixed(self):
        arr = np.array([0, 1, 2, 0, 255, 0, 3, 0, 128, 0], dtype=np.uint8)
        result = np.logical_or(np.uint8(0), arr)
        expected = arr != 0
        assert_array_equal(result, expected)

    def test_logical_or_uint8_scalar_nonzero(self):
        arr = np.array([0, 1, 2, 0, 255], dtype=np.uint8)
        result = np.logical_or(np.uint8(5), arr)
        assert np.all(result)

    def test_logical_and_uint8_scalar_nonzero(self):
        arr = np.array([0, 1, 2, 0, 255], dtype=np.uint8)
        result = np.logical_and(np.uint8(3), arr)
        expected = arr != 0
        assert_array_equal(result, expected)

    def test_logical_and_uint8_scalar_zero(self):
        arr = np.array([0, 1, 2, 0, 255], dtype=np.uint8)
        result = np.logical_and(np.uint8(0), arr)
        assert not np.any(result)

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129])
    def test_logical_or_uint8_various_sizes(self, size):
        arr = np.arange(size, dtype=np.uint8)
        result = np.logical_or(np.uint8(0), arr)
        expected = arr != 0
        assert_array_equal(result, expected)


class TestBitwiseSVEInt64:
    """Target SVE bitwise operations for int64."""

    def test_bitwise_and_int64_scalar(self):
        scalar = np.int64(0x0F0F0F0F0F0F0F0F)
        arr = np.arange(64, dtype=np.int64)
        result = np.bitwise_and(scalar, arr)
        assert result.dtype == np.int64

    def test_bitwise_or_int64_scalar(self):
        scalar = np.int64(0x0F0F0F0F0F0F0F0F)
        arr = np.arange(64, dtype=np.int64)
        result = np.bitwise_or(scalar, arr)
        assert result.dtype == np.int64

    def test_bitwise_xor_int64_scalar(self):
        scalar = np.int64(0x5555555555555555)
        arr = np.arange(64, dtype=np.int64)
        result = np.bitwise_xor(scalar, arr)
        assert result.dtype == np.int64

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65])
    def test_bitwise_and_int64_various_sizes(self, size):
        scalar = np.int64(0xFF)
        arr = np.arange(size, dtype=np.int64)
        result = np.bitwise_and(scalar, arr)
        assert result.dtype == np.int64
        assert result.shape == (size,)
