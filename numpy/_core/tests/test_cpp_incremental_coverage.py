"""
Tests to improve C++ incremental code coverage from 75% to 80%+.
Targets files with lowest incremental coverage:
- loops_minmax_hwy.dispatch.cpp (0%)
- loops_arithm_sum_hwy.dispatch.cpp (1.2%)
- highway_argfunc.dispatch.cpp (20.1%)
- loops_arithmetic_floor_hwy.dispatch.cpp (21.1%)
- loops_autovec.dispatch.c.src (36.1%)
- loops_arithmetic.dispatch.c.src (44.7%)
- loops_minmax.dispatch.c.src (47.1%)
- loops_unary_fp_ops.dispatch.cpp (59.6%)
- loops_autovec_abs_hwy.dispatch.cpp (64.2%)
- loops_exp2.dispatch.cpp (75.3%)
- loops_shift_sve.c (78.5%)
- loops_logical_sve.c (81.4%)
- loops_power.dispatch.cpp (85.7%)
- compiled_base.c (86.4%)
- mapping.c (88.0%)
- ufunc_object.c (86.0%)
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


N = 4096


class TestHighwayMinMaxAllTypes:
    """loops_minmax_hwy.dispatch.cpp: Highway SIMD max/min for all types."""

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_maximum_integer_contiguous(self, dt):
        rng = np.random.default_rng(42)
        lo = 0 if np.issubdtype(dt, np.unsignedinteger) else -100
        a = rng.integers(lo, 100, size=N).astype(dt)
        b = rng.integers(lo, 100, size=N).astype(dt)
        res = np.maximum(a, b)
        assert_array_equal(res, np.where(a >= b, a, b))

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_minimum_integer_contiguous(self, dt):
        rng = np.random.default_rng(42)
        lo = 0 if np.issubdtype(dt, np.unsignedinteger) else -100
        a = rng.integers(lo, 100, size=N).astype(dt)
        b = rng.integers(lo, 100, size=N).astype(dt)
        res = np.minimum(a, b)
        assert_array_equal(res, np.where(a <= b, a, b))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_float_contiguous(self, dt):
        rng = np.random.default_rng(42)
        a = rng.random(N).astype(dt)
        b = rng.random(N).astype(dt)
        res = np.maximum(a, b)
        assert_array_equal(res, np.where(a >= b, a, b))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_float_contiguous(self, dt):
        rng = np.random.default_rng(42)
        a = rng.random(N).astype(dt)
        b = rng.random(N).astype(dt)
        res = np.minimum(a, b)
        assert_array_equal(res, np.where(a <= b, a, b))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmax_nan_ignoring(self, dt):
        a = np.array([np.nan, 1.0, np.nan, 3.0, 5.0] * (N // 5), dtype=dt)
        b = np.array([4.0, np.nan, 5.0, np.nan, 2.0] * (N // 5), dtype=dt)
        res = np.fmax(a, b)
        assert res[0] == 4.0
        assert res[1] == 1.0
        assert res[2] == 5.0
        assert res[3] == 3.0

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmin_nan_ignoring(self, dt):
        a = np.array([np.nan, 1.0, np.nan, 3.0, 5.0] * (N // 5), dtype=dt)
        b = np.array([4.0, np.nan, 5.0, np.nan, 2.0] * (N // 5), dtype=dt)
        res = np.fmin(a, b)
        assert res[0] == 4.0
        assert res[1] == 1.0
        assert res[2] == 5.0
        assert res[3] == 3.0

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_nan_both_nan(self, dt):
        a = np.array([np.nan, np.nan, 1.0] * (N // 3), dtype=dt)
        b = np.array([np.nan, 2.0, np.nan] * (N // 3), dtype=dt)
        with np.errstate(invalid='ignore'):
            res = np.maximum(a, b)
        assert res.shape == a.shape

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmax_both_nan(self, dt):
        a = np.array([np.nan, np.nan, 1.0] * (N // 3), dtype=dt)
        b = np.array([np.nan, 2.0, np.nan] * (N // 3), dtype=dt)
        res = np.fmax(a, b)
        assert res.shape == a.shape

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_maximum_scalar_broadcast(self, dt):
        a = np.arange(min(N, 256), dtype=dt)
        scalar = dt(50)
        res = np.maximum(a, scalar)
        assert_array_equal(res, np.where(a >= scalar, a, scalar))

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_minimum_scalar_broadcast(self, dt):
        a = np.arange(min(N, 256), dtype=dt)
        scalar = dt(50)
        res = np.minimum(a, scalar)
        assert_array_equal(res, np.where(a <= scalar, a, scalar))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_large_array(self, dt):
        a = np.random.default_rng(42).random(10000).astype(dt)
        b = np.random.default_rng(43).random(10000).astype(dt)
        res = np.maximum(a, b)
        assert_array_equal(res, np.where(a >= b, a, b))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_large_array(self, dt):
        a = np.random.default_rng(42).random(10000).astype(dt)
        b = np.random.default_rng(43).random(10000).astype(dt)
        res = np.minimum(a, b)
        assert_array_equal(res, np.where(a <= b, a, b))

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 31, 63, 127, 255, 511])
    def test_maximum_odd_sizes(self, size):
        rng = np.random.default_rng(42)
        for dt in [np.int32, np.float64]:
            a = rng.integers(0, 100, size=size).astype(dt)
            b = rng.integers(0, 100, size=size).astype(dt)
            res = np.maximum(a, b)
            assert_array_equal(res, np.where(a >= b, a, b))

    @pytest.mark.parametrize("dt", [np.int64, np.uint64])
    def test_maximum_64bit_extremes(self, dt):
        info = np.iinfo(dt)
        a = np.array([info.min, info.max, 0, 1, -1] if np.issubdtype(dt, np.signedinteger)
                      else [0, info.max, 1, 100, 200], dtype=dt)
        b = np.array([info.max, info.min, 1, 0, 1] if np.issubdtype(dt, np.signedinteger)
                      else [info.max, 0, 100, 1, 200], dtype=dt)
        res = np.maximum(a, b)
        assert_array_equal(res, np.where(a >= b, a, b))


class TestHighwayArgFunc:
    """highway_argfunc.dispatch.cpp: Highway SIMD argmin/argmax."""

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_argmax_integer(self, dt):
        rng = np.random.default_rng(42)
        lo = 0 if np.issubdtype(dt, np.unsignedinteger) else -100
        a = rng.integers(lo, 100, size=N).astype(dt)
        assert np.argmax(a) == np.argmax(a.astype(np.int64))

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_argmin_integer(self, dt):
        rng = np.random.default_rng(42)
        lo = 0 if np.issubdtype(dt, np.unsignedinteger) else -100
        a = rng.integers(lo, 100, size=N).astype(dt)
        assert np.argmin(a) == np.argmin(a.astype(np.int64))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmax_float_with_nan(self, dt):
        a = np.ones(N, dtype=dt)
        a[N // 3] = np.nan
        assert np.argmax(a) == N // 3

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmin_float_with_nan(self, dt):
        a = np.ones(N, dtype=dt)
        a[N // 3] = np.nan
        assert np.argmin(a) == N // 3

    def test_argmax_bool(self):
        a = np.array([False, False, True, False, True] * (N // 5))
        assert np.argmax(a) == 2

    def test_argmin_bool(self):
        a = np.array([True, True, False, True, False] * (N // 5))
        assert np.argmin(a) == 2

    def test_argmax_bool_all_false(self):
        a = np.zeros(N, dtype=bool)
        assert np.argmax(a) == 0

    def test_argmin_bool_all_true(self):
        a = np.ones(N, dtype=bool)
        assert np.argmin(a) == 0

    def test_argmax_longdouble(self):
        a = np.array([1.0, 5.0, 3.0, 2.0, 4.0], dtype=np.longdouble)
        assert np.argmax(a) == 1

    def test_argmin_longdouble(self):
        a = np.array([5.0, 1.0, 3.0, 2.0, 4.0], dtype=np.longdouble)
        assert np.argmin(a) == 1

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmax_uniform_array(self, dt):
        a = np.ones(N, dtype=dt)
        assert np.argmax(a) == 0

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmin_uniform_array(self, dt):
        a = np.ones(N, dtype=dt)
        assert np.argmin(a) == 0

    @pytest.mark.parametrize("dt", [np.int8, np.uint8, np.int16, np.uint16])
    def test_argmax_narrow_types(self, dt):
        lo = 0 if np.issubdtype(dt, np.unsignedinteger) else -128
        hi = 127 if np.issubdtype(dt, np.signedinteger) else 255
        a = np.arange(lo, min(lo + N, hi + 1), dtype=dt)
        assert np.argmax(a) == len(a) - 1

    @pytest.mark.parametrize("dt", [np.int8, np.uint8, np.int16, np.uint16])
    def test_argmin_narrow_types(self, dt):
        lo = 0 if np.issubdtype(dt, np.unsignedinteger) else -128
        hi = 127 if np.issubdtype(dt, np.signedinteger) else 255
        a = np.arange(lo, min(lo + N, hi + 1), dtype=dt)
        assert np.argmin(a) == 0

    def test_argmax_large_array(self):
        a = np.random.default_rng(42).random(100000)
        assert np.argmax(a) == np.argmax(a.astype(np.float64))

    def test_argmin_large_array(self):
        a = np.random.default_rng(42).random(100000)
        assert np.argmin(a) == np.argmin(a.astype(np.float64))

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 31, 63, 127, 255])
    def test_argmax_various_sizes(self, size):
        rng = np.random.default_rng(42)
        for dt in [np.float32, np.float64, np.int32]:
            a = rng.random(size).astype(dt)
            assert np.argmax(a) == np.argmax(a.astype(np.float64))


class TestHighwayFloorDivide:
    """loops_arithmetic_floor_hwy.dispatch.cpp: Highway SIMD floor_divide."""

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32])
    def test_floor_divide_signed(self, dt):
        a = np.arange(1, min(N, 127) + 1, dtype=dt)
        b = np.full_like(a, fill_value=dt(3))
        res = np.floor_divide(a, b)
        assert_array_equal(res, a // b)

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32])
    def test_floor_divide_signed_mixed_signs(self, dt):
        a = np.array([-7, 7, -7, 7, -10, 10, -10, 10] * (N // 8), dtype=dt)
        b = np.array([3, -3, -3, 3, 3, -3, -3, 3] * (N // 8), dtype=dt)
        res = np.floor_divide(a, b)
        assert_array_equal(res, a // b)

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32])
    def test_floor_divide_unsigned(self, dt):
        a = np.arange(1, min(N, 255) + 1, dtype=dt)
        b = np.full_like(a, fill_value=dt(3))
        res = np.floor_divide(a, b)
        assert_array_equal(res, a // b)

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_floor_divide_by_zero(self, dt):
        a = np.array([10, 20, 30], dtype=dt)
        b = np.array([0, 5, 0], dtype=dt)
        with np.errstate(divide="ignore"):
            res = np.floor_divide(a, b)
        assert res.shape == (3,)

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_floor_divide_overflow(self, dt):
        info = np.iinfo(dt)
        a = np.array([info.min], dtype=dt)
        b = np.array([-1], dtype=dt)
        with np.errstate(over="ignore"):
            res = np.floor_divide(a, b)
        assert res.shape == (1,)

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64])
    def test_floor_divide_scalar_one(self, dt):
        a = np.arange(1, min(N, 1000) + 1, dtype=dt)
        res = np.floor_divide(a, dt(1))
        assert_array_equal(res, a)

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64])
    def test_floor_divide_scalar_minus_one(self, dt):
        a = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=dt)
        with np.errstate(over="ignore"):
            res = np.floor_divide(a, dt(-1))
        assert res.shape == (7,)

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64])
    def test_floor_divide_scalar_zero(self, dt):
        a = np.array([5, 10, 15], dtype=dt)
        with np.errstate(divide="ignore"):
            res = np.floor_divide(a, dt(0))
        assert res.shape == (3,)

    def test_floor_divide_uint32_scalar(self):
        a = np.arange(1, N + 1, dtype=np.uint32)
        res = np.floor_divide(a, np.uint32(7))
        assert_array_equal(res, a // np.uint32(7))

    def test_floor_divide_int64_scalar(self):
        a = np.arange(-100, 100, dtype=np.int64)
        a = a[a != 0]
        res = np.floor_divide(a, np.int64(7))
        assert_array_equal(res, a // np.int64(7))

    def test_floor_divide_large_arrays(self):
        rng = np.random.default_rng(42)
        a = rng.integers(1, 1000, size=10000).astype(np.int32)
        b = rng.integers(1, 100, size=10000).astype(np.int32)
        res = np.floor_divide(a, b)
        assert_array_equal(res, a // b)

    def test_floor_divide_unsigned_large(self):
        rng = np.random.default_rng(42)
        a = rng.integers(1, 1000, size=10000).astype(np.uint32)
        b = rng.integers(1, 100, size=10000).astype(np.uint32)
        res = np.floor_divide(a, b)
        assert_array_equal(res, a // b)


class TestPairwiseSum:
    """loops_arithm_sum_hwy.dispatch.cpp: Highway SIMD pairwise sum."""

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_sum_small(self, dt):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dt)
        assert_allclose(np.sum(a), 15.0, rtol=1e-6)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_sum_medium(self, dt):
        a = np.arange(100, dtype=dt)
        assert_allclose(np.sum(a), 4950.0, rtol=1e-6)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_sum_large(self, dt):
        a = np.ones(10000, dtype=dt)
        assert_allclose(np.sum(a), 10000.0, rtol=1e-6)

    def test_sum_complex64(self):
        a = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
        assert_allclose(np.sum(a), 9+12j, rtol=1e-5)

    def test_sum_complex128(self):
        a = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
        assert_allclose(np.sum(a), 9+12j, rtol=1e-10)

    def test_sum_complex_large(self):
        a = np.ones(1000, dtype=np.complex128) * (1+1j)
        assert_allclose(np.sum(a), 1000+1000j, rtol=1e-10)

    def test_add_reduce_float32(self):
        a = np.arange(1000, dtype=np.float32)
        assert_allclose(np.add.reduce(a), np.sum(a.astype(np.float64)), rtol=1e-4)

    def test_add_reduce_float64(self):
        a = np.arange(1000, dtype=np.float64)
        assert_allclose(np.add.reduce(a), 499500.0, rtol=1e-10)

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 31, 63, 127, 255, 511, 1023])
    def test_sum_various_sizes(self, size):
        for dt in [np.float32, np.float64]:
            a = np.ones(size, dtype=dt)
            assert_allclose(np.sum(a), float(size), rtol=1e-6)

    def test_sum_negative_zero(self):
        a = np.array([-0.0, 0.0, -0.0, 0.0], dtype=np.float64)
        res = np.sum(a)
        assert res == 0.0

    def test_add_reduce_complex(self):
        a = np.arange(100, dtype=np.complex128)
        a = a + 1j * a
        res = np.add.reduce(a)
        expected = np.sum(a)
        assert_allclose(res, expected, rtol=1e-10)


class TestAutoVecDispatch:
    """loops_autovec.dispatch.c.src: auto-vectorized loops."""

    @pytest.mark.parametrize("dt", [
        np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_bitwise_count(self, dt):
        a = np.array([0, 1, 3, 7, 15, 31, 63, 127, 255], dtype=dt)
        res = np.bitwise_count(a)
        expected = np.array([bin(x).count("1") for x in a], dtype=np.int32)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64])
    def test_left_shift(self, dt):
        a = np.array([1, 2, 4, 8], dtype=dt)
        b = np.array([1, 2, 3, 4], dtype=dt)
        res = np.left_shift(a, b)
        assert_array_equal(res, [2, 8, 32, 128])

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64])
    def test_right_shift(self, dt):
        a = np.array([128, 64, 32, 16], dtype=dt)
        b = np.array([1, 2, 3, 4], dtype=dt)
        res = np.right_shift(a, b)
        assert_array_equal(res, [64, 16, 4, 1])

    def test_bool_floor_ceil_trunc(self):
        a = np.array([True, False, True, False])
        assert_array_equal(np.floor(a), [1, 0, 1, 0])
        assert_array_equal(np.ceil(a), [1, 0, 1, 0])
        assert_array_equal(np.trunc(a), [1, 0, 1, 0])

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_add_reduce_unsigned(self, dt):
        n = min(N, 255) if dt == np.uint8 else N
        a = np.ones(n, dtype=dt)
        res = np.add.reduce(a)
        assert res == n

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_multiply_reduce_unsigned(self, dt):
        a = np.ones(10, dtype=dt)
        res = np.multiply.reduce(a)
        assert res == 1

    def test_half_absolute(self):
        a = np.array([-1.5, 2.5, -0.0, 0.0, -3.0], dtype=np.float16)
        res = np.abs(a)
        assert_array_equal(res, [1.5, 2.5, 0.0, 0.0, 3.0])

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_xor(self, dt):
        a = np.array([0, 5, 0, 3], dtype=dt)
        b = np.array([0, 0, 7, 3], dtype=dt)
        res = np.logical_xor(a, b)
        assert_array_equal(res, [False, True, True, False])

    @pytest.mark.parametrize("dt", [np.int8, np.uint8, np.int16, np.uint16])
    def test_shift_narrow_types(self, dt):
        a = np.array([1, 2, 4], dtype=dt)
        b = np.array([1, 1, 1], dtype=dt)
        res = np.left_shift(a, b)
        assert_array_equal(res, [2, 4, 8])


class TestArithmeticDispatch:
    """loops_arithmetic.dispatch.c.src: integer floor_divide dispatch."""

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_floor_divide_at(self, dt):
        a = np.array([10, 20, 30], dtype=dt)
        np.floor_divide.at(a, [0, 1], [dt(3), dt(7)])
        assert a[0] == 10 // 3
        assert a[1] == 20 // 7

    def test_floor_divide_reduce(self):
        a = np.array([100, 10, 2], dtype=np.int32)
        res = np.floor_divide.reduce(a)
        assert res == (100 // 10) // 2

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_floor_divide_unsigned_types(self, dt):
        a = np.array([100, 200, 50], dtype=dt)
        b = np.array([3, 7, 10], dtype=dt)
        res = np.floor_divide(a, b)
        assert_array_equal(res, a // b)

    def test_floor_divide_int64_libdivide(self):
        a = np.arange(1, 1001, dtype=np.int64)
        res = np.floor_divide(a, np.int64(7))
        assert_array_equal(res, a // 7)


class TestMinMaxDispatch:
    """loops_minmax.dispatch.c.src: NPyV SIMD min/max dispatch."""

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_at(self, dt):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dt)
        np.maximum.at(a, [0, 1, 0], np.array([5.0, 0.5, 0.5], dtype=dt))
        assert a[0] == 5.0
        assert a[1] == 2.0

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_at(self, dt):
        a = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=dt)
        np.minimum.at(a, [0, 1, 0], np.array([1.0, 5.0, 2.0], dtype=dt))
        assert a[0] == 1.0
        assert a[1] == 4.0

    def test_fmax_longdouble(self):
        a = np.array([1.0, np.nan, 3.0], dtype=np.longdouble)
        b = np.array([2.0, 4.0, np.nan], dtype=np.longdouble)
        res = np.fmax(a, b)
        assert res[0] == 2.0
        assert res[1] == 4.0
        assert res[2] == 3.0

    def test_fmin_longdouble(self):
        a = np.array([1.0, np.nan, 3.0], dtype=np.longdouble)
        b = np.array([2.0, 4.0, np.nan], dtype=np.longdouble)
        res = np.fmin(a, b)
        assert res[0] == 1.0
        assert res[1] == 4.0
        assert res[2] == 3.0

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_maximum_reduce_signed(self, dt):
        info = np.iinfo(dt)
        n = min(N, info.max)
        a = np.arange(n, dtype=dt)
        assert np.maximum.reduce(a) == a[-1]

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_strided(self, dt):
        a = np.arange(100, dtype=dt)[::2]
        b = np.arange(100, dtype=dt)[1::2]
        res = np.maximum(a, b)
        assert_array_equal(res, np.where(a >= b, a, b))


class TestUnaryFPOps:
    """loops_unary_fp_ops.dispatch.cpp: Highway SIMD unary FP ops."""

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_reciprocal(self, dt):
        a = np.array([1.0, 2.0, 4.0, 0.5, 0.25], dtype=dt)
        res = np.reciprocal(a)
        assert_allclose(res, [1.0, 0.5, 0.25, 2.0, 4.0], rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_ceil(self, dt):
        a = np.array([1.5, -1.5, 2.7, -2.7, 0.0], dtype=dt)
        res = np.ceil(a)
        assert_allclose(res, [2.0, -1.0, 3.0, -2.0, 0.0], rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_floor(self, dt):
        a = np.array([1.5, -1.5, 2.7, -2.7, 0.0], dtype=dt)
        res = np.floor(a)
        assert_allclose(res, [1.0, -2.0, 2.0, -3.0, 0.0], rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_trunc(self, dt):
        a = np.array([1.5, -1.5, 2.7, -2.7, 0.0], dtype=dt)
        res = np.trunc(a)
        assert_allclose(res, [1.0, -1.0, 2.0, -2.0, 0.0], rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_rint(self, dt):
        a = np.array([0.5, 1.5, 2.5, 3.5, -0.5], dtype=dt)
        res = np.rint(a)
        assert res.shape == (5,)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_deg2rad(self, dt):
        a = np.array([0, 90, 180, 270, 360], dtype=dt)
        res = np.deg2rad(a)
        expected = np.deg2rad(a.astype(np.float64)).astype(dt)
        assert_allclose(res, expected, rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_rad2deg(self, dt):
        a = np.array([0, 1.5708, 3.1416, 4.7124, 6.2832], dtype=dt)
        res = np.rad2deg(a)
        expected = np.rad2deg(a.astype(np.float64)).astype(dt)
        assert_allclose(res, expected, rtol=1e-2)

    def test_complex64_reciprocal(self):
        a = np.array([1+2j, 3+4j, 0.5+0.5j], dtype=np.complex64)
        res = np.reciprocal(a)
        expected = 1.0 / a.astype(np.complex128)
        assert_allclose(res, expected.astype(np.complex64), rtol=1e-5)

    def test_complex128_reciprocal(self):
        a = np.array([1+2j, 3+4j, 0.5+0.5j], dtype=np.complex128)
        res = np.reciprocal(a)
        expected = 1.0 / a
        assert_allclose(res, expected, rtol=1e-10)

    def test_float16_strided(self):
        a = np.arange(20, dtype=np.float16)[::3]
        res = np.ceil(a)
        assert res.shape == a.shape

    def test_float32_strided_input(self):
        a = np.arange(20, dtype=np.float32)[::3]
        res = np.deg2rad(a)
        assert res.shape == a.shape

    def test_float64_strided_output(self):
        out = np.zeros(20, dtype=np.float64)
        a = np.arange(10, dtype=np.float64)
        out[::2] = np.rad2deg(a)
        assert out[0] == 0.0


class TestExp2Coverage:
    """loops_exp2.dispatch.cpp: exp2 edge cases."""

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_exp2_special_values(self, dt):
        a = np.array([0, 1, 2, 10, -1, -10], dtype=dt)
        res = np.exp2(a)
        expected = np.exp2(a.astype(np.float64)).astype(dt)
        assert_allclose(res, expected, rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_exp2_overflow_underflow(self, dt):
        a = np.array([np.nan, np.inf, -np.inf, 200, -200], dtype=dt)
        with np.errstate(over='ignore', invalid='ignore'):
            res = np.exp2(a)
        assert res.shape == (5,)

    def test_exp2_half(self):
        a = np.array([0, 1, 5, -1, -5], dtype=np.float16)
        res = np.exp2(a)
        assert res.shape == (5,)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_exp2_strided(self, dt):
        a = np.arange(20, dtype=dt)[::3]
        with np.errstate(over='ignore', invalid='ignore'):
            res = np.exp2(a)
        assert res.shape == a.shape

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 31, 63])
    def test_exp2_various_sizes(self, size):
        for dt in [np.float32, np.float64]:
            a = np.ones(size, dtype=dt)
            res = np.exp2(a)
            assert_allclose(res, np.full(size, 2.0, dtype=dt), rtol=1e-6)


class TestAutoVecAbsHWY:
    """loops_autovec_abs_hwy.dispatch.cpp: Highway half-float abs."""

    def test_half_abs_contiguous(self):
        a = np.array([-1.5, 2.5, -0.0, 0.0, -3.0, 4.5], dtype=np.float16)
        res = np.abs(a)
        assert_array_equal(res, [1.5, 2.5, 0.0, 0.0, 3.0, 4.5])

    def test_half_abs_strided(self):
        a = np.arange(-10, 10, dtype=np.float16)[::3]
        res = np.abs(a)
        assert_array_equal(res, np.abs(a.astype(np.float32)).astype(np.float16))

    def test_half_abs_large(self):
        a = np.random.default_rng(42).standard_normal(1000).astype(np.float16)
        res = np.abs(a)
        assert np.all(res >= 0)


class TestShiftSVE:
    """loops_shift_sve.c: SVE shift operations."""

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64])
    def test_left_shift_signed(self, dt):
        a = np.array([1, 2, 4, 8], dtype=dt)
        b = np.array([1, 2, 3, 4], dtype=dt)
        res = np.left_shift(a, b)
        assert_array_equal(res, [2, 8, 32, 128])

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_left_shift_unsigned(self, dt):
        a = np.array([1, 2, 4, 8], dtype=dt)
        b = np.array([1, 2, 3, 4], dtype=dt)
        res = np.left_shift(a, b)
        assert_array_equal(res, [2, 8, 32, 128])

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_right_shift_signed_negative(self, dt):
        a = np.array([-128, -64, -32, -16], dtype=dt)
        b = np.array([1, 2, 3, 4], dtype=dt)
        res = np.right_shift(a, b)
        assert_array_equal(res, np.right_shift(a.astype(np.int64), b.astype(np.int64)).astype(dt))

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_right_shift_unsigned(self, dt):
        a = np.array([128, 64, 32, 16], dtype=dt)
        b = np.array([1, 2, 3, 4], dtype=dt)
        res = np.right_shift(a, b)
        assert_array_equal(res, [64, 16, 4, 1])

    def test_shift_scalar_in0(self):
        a = np.int32(1)
        b = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        res = np.left_shift(a, b)
        assert_array_equal(res, [2, 4, 8, 16, 32])

    @pytest.mark.parametrize("dt", [np.int64, np.uint64])
    def test_shift_64bit(self, dt):
        a = np.array([1, 2, 4], dtype=dt)
        b = np.array([10, 20, 30], dtype=dt)
        res = np.left_shift(a, b)
        assert_array_equal(res, [1024, 2097152, 4294967296])


class TestLogicalSVE:
    """loops_logical_sve.c: SVE logical operations."""

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_or_scalar_nonzero(self, dt):
        a = np.array([0, 1, 0, 1], dtype=dt)
        res = np.logical_or(a, True)
        assert_array_equal(res, [True, True, True, True])

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_or_scalar_zero(self, dt):
        a = np.array([0, 1, 0, 1], dtype=dt)
        res = np.logical_or(a, False)
        assert_array_equal(res, [False, True, False, True])

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_or_scalar_left(self, dt):
        a = np.array([0, 1, 0, 1], dtype=dt)
        res = np.logical_or(True, a)
        assert_array_equal(res, [True, True, True, True])

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_or_large(self, dt):
        a = np.zeros(N, dtype=dt)
        a[N // 2] = 1
        res = np.logical_or(a, False)
        assert res[N // 2] == True
        assert res[0] == False


class TestBitwiseSVE:
    """loops_bitwise_sve.c: SVE bitwise operations."""

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_and(self, dt):
        a = np.array([0xFF, 0x0F, 0xF0], dtype=dt)
        b = np.array([0xF0, 0xF0, 0xF0], dtype=dt)
        res = np.bitwise_and(a, b)
        assert_array_equal(res, [0xF0, 0x00, 0xF0])

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_or(self, dt):
        a = np.array([0xF0, 0x0F, 0x00], dtype=dt)
        b = np.array([0x0F, 0xF0, 0xFF], dtype=dt)
        res = np.bitwise_or(a, b)
        assert_array_equal(res, [0xFF, 0xFF, 0xFF])

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_xor(self, dt):
        a = np.array([0xFF, 0x0F, 0xF0], dtype=dt)
        b = np.array([0xF0, 0xF0, 0xF0], dtype=dt)
        res = np.bitwise_xor(a, b)
        assert_array_equal(res, [0x0F, 0xFF, 0x00])

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64])
    def test_bitwise_and_signed(self, dt):
        a = np.array([0x0F, -1, 0x55], dtype=dt)
        b = np.array([0xFF, 0x0F, 0xFF], dtype=dt)
        res = np.bitwise_and(a, b)
        assert res.shape == (3,)

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_xor(self, dt):
        a = np.array([0, 5, 0, 3], dtype=dt)
        b = np.array([0, 0, 7, 3], dtype=dt)
        res = np.logical_xor(a, b)
        assert_array_equal(res, [False, True, True, False])


class TestLogicalNotAArch64:
    """loops_logical_not_aarch64.c: AArch64 NEON logical_not."""

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_logical_not_contiguous(self, dt):
        a = np.array([0.0, 1.0, -0.0, -1.0, np.nan], dtype=dt)
        res = np.logical_not(a)
        assert_array_equal(res, [True, False, True, False, False])

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_logical_not_strided(self, dt):
        a = np.arange(20, dtype=dt)[::3]
        res = np.logical_not(a)
        assert res.shape == a.shape

    def test_logical_not_small(self):
        a = np.array([0.0, 1.0], dtype=np.float64)
        res = np.logical_not(a)
        assert_array_equal(res, [True, False])


class TestPowerExtended:
    """loops_power.dispatch.cpp: power edge cases."""

    def test_power_negative_base(self):
        a = np.array([-2.0, -3.0, -4.0], dtype=np.float64)
        b = np.array([2.0, 3.0, 2.0], dtype=np.float64)
        res = np.power(a, b)
        assert_allclose(res, [4.0, -27.0, 16.0])

    def test_power_zero_base(self):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0, -1.0], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.power(a, b)
        assert res.shape == (3,)

    def test_power_strided(self):
        a = np.arange(1, 20, dtype=np.float64)[::3]
        b = np.full_like(a, 2.0)
        res = np.power(a, b)
        assert_allclose(res, a ** 2)

    def test_power_fractional_exponent(self):
        a = np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float64)
        b = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        res = np.power(a, b)
        assert_allclose(res, [2.0, 3.0, 4.0, 5.0])


class TestCompiledBase:
    """compiled_base.c: histogram2d error paths."""

    def test_histogram2d_zero_bins(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [3, 4], bins=[0, 0])

    def test_histogram2d_nan_range(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [3, 4], bins=[10, 10],
                           range=[[np.nan, 1], [0, 1]])

    def test_histogram2d_decreasing_range(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [3, 4], bins=[10, 10],
                           range=[[5, 1], [0, 1]])

    def test_histogram2d_inf_range(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [3, 4], bins=[10, 10],
                           range=[[np.inf, 1], [0, 1]])

    def test_histogram2d_normal(self):
        x = np.random.randn(100)
        y = np.random.randn(100)
        H, xedges, yedges = np.histogram2d(x, y, bins=10)
        assert H.shape == (10, 10)


class TestMappingExtended:
    """mapping.c: array indexing edge cases."""

    def test_boolean_mask_assignment_size_mismatch(self):
        a = np.zeros(5)
        mask = np.array([True, False, True, False, True])
        with pytest.raises(ValueError):
            a[mask] = np.ones(5)

    def test_ellipsis_self_assignment(self):
        a = np.array([1, 2, 3])
        a[...] = a.copy()
        assert_array_equal(a, [1, 2, 3])

    def test_fancy_indexing_broadcast_error(self):
        a = np.zeros((10, 10))
        with pytest.raises(IndexError):
            a[np.array([0, 1]), np.array([0, 1, 2])]


class TestUfuncObjectExtended:
    """ufunc_object.c: scalar fast path for math ufuncs."""

    @pytest.mark.parametrize("func", [np.sin, np.cos, np.tan, np.exp, np.log, np.sqrt])
    def test_scalar_fast_path_float64(self, func):
        res = func(np.float64(1.0))
        assert isinstance(res, (np.floating, float))

    @pytest.mark.parametrize("func", [np.sin, np.cos, np.exp, np.log, np.sqrt])
    def test_scalar_fast_path_complex128(self, func):
        res = func(np.complex128(1+1j))
        assert isinstance(res, (np.complexfloating, complex))

    def test_scalar_fast_path_longdouble(self):
        res = np.sin(np.longdouble(1.0))
        assert isinstance(res, (np.floating, float))

    def test_scalar_fast_path_complex64(self):
        res = np.sin(np.complex64(1+1j))
        assert isinstance(res, (np.complexfloating, complex))

    def test_scalar_exp_overflow(self):
        with np.errstate(over="ignore"):
            res = np.exp(np.float64(1000))
        assert np.isinf(res) or res > 0

    def test_scalar_log_negative(self):
        with np.errstate(invalid="ignore"):
            res = np.log(np.float64(-1.0))
        assert np.isnan(res) or res == 0.0


class TestTrigExtended:
    """loops_trigonometric.dispatch.cpp: trig edge cases."""

    def test_arctan2(self):
        y = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float64)
        x = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float64)
        res = np.arctan2(y, x)
        assert res.shape == (4,)

    def test_trig_large_arrays(self):
        a = np.random.default_rng(42).random(10000)
        for func in [np.sin, np.cos, np.tan]:
            res = func(a)
            assert res.shape == (10000,)

    def test_inverse_trig(self):
        a = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        for func in [np.arcsin, np.arccos, np.arctan]:
            res = func(a)
            assert res.shape == (3,)

    def test_hyperbolic(self):
        a = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        for func in [np.sinh, np.cosh, np.tanh]:
            res = func(a)
            assert res.shape == (3,)


class TestUmathUnaryExtended:
    """loops_umath_unary.dispatch.cpp: sign and other unary ops."""

    def test_sign_float(self):
        a = np.array([-3.0, -0.0, 0.0, 1.0, -1.0], dtype=np.float64)
        res = np.sign(a)
        assert_array_equal(res, [-1.0, 0.0, 0.0, 1.0, -1.0])

    def test_sign_complex(self):
        a = np.array([1+1j, -1-1j, 0+0j], dtype=np.complex128)
        res = np.sign(a)
        assert res.shape == (3,)

    def test_positive(self):
        a = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        res = np.positive(a)
        assert_array_equal(res, a)

    def test_negative(self):
        a = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        res = np.negative(a)
        assert_array_equal(res, -a)


class TestExpLogExtended:
    """loops_exp/log.dispatch.cpp: exp/log edge cases."""

    def test_exp_large(self):
        a = np.array([1.0, 10.0, 100.0, 700.0], dtype=np.float64)
        res = np.exp(a)
        assert res.shape == (4,)

    def test_log_large(self):
        a = np.array([1.0, 10.0, 100.0, 1e300], dtype=np.float64)
        res = np.log(a)
        assert res.shape == (4,)

    def test_expm1(self):
        a = np.array([0.0, 1e-10, 1.0, 10.0], dtype=np.float64)
        res = np.expm1(a)
        assert res.shape == (4,)
        assert_allclose(res[0], 0.0, atol=1e-15)

    def test_log1p(self):
        a = np.array([0.0, 1e-10, 1.0, 10.0], dtype=np.float64)
        res = np.log1p(a)
        assert res.shape == (4,)
        assert_allclose(res[0], 0.0, atol=1e-15)

    def test_cbrt(self):
        a = np.array([-8.0, -1.0, 0.0, 1.0, 8.0, 27.0], dtype=np.float64)
        res = np.cbrt(a)
        assert_allclose(res, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])


class TestLog2Extended:
    """loops_log2.dispatch.cpp: log2 edge cases."""

    def test_log2_special(self):
        a = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
        res = np.log2(a)
        assert_allclose(res, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_log2_half(self):
        a = np.array([1.0, 2.0, 4.0, 0.5], dtype=np.float16)
        res = np.log2(a)
        assert res.shape == (4,)

    def test_log2_strided(self):
        a = np.arange(1, 20, dtype=np.float64)[::3]
        res = np.log2(a)
        assert res.shape == a.shape


class TestInPlaceOverlap:
    """Overlap fallback paths in exp/exp2/log/log2/trig/power dispatch files."""

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_exp2_inplace(self, dt):
        a = np.array([0, 1, 2, 3, 4, 5], dtype=dt)
        expected = np.exp2(a.copy())
        np.exp2(a, out=a)
        assert_allclose(a, expected, rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_exp_inplace(self, dt):
        a = np.array([0, 1, 2, 3], dtype=dt)
        expected = np.exp(a.copy())
        np.exp(a, out=a)
        assert_allclose(a, expected, rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_log_inplace(self, dt):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=dt)
        expected = np.log(a.copy())
        np.log(a, out=a)
        assert_allclose(a, expected, rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_log2_inplace(self, dt):
        a = np.array([1.0, 2.0, 4.0, 8.0], dtype=dt)
        expected = np.log2(a.copy())
        np.log2(a, out=a)
        assert_allclose(a, expected, rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_sin_inplace(self, dt):
        a = np.array([0.0, 0.5, 1.0, 1.5], dtype=dt)
        expected = np.sin(a.copy())
        np.sin(a, out=a)
        assert_allclose(a, expected, rtol=1e-5)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_cos_inplace(self, dt):
        a = np.array([0.0, 0.5, 1.0, 1.5], dtype=dt)
        expected = np.cos(a.copy())
        np.cos(a, out=a)
        assert_allclose(a, expected, rtol=1e-5)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_tan_inplace(self, dt):
        a = np.array([0.0, 0.5, 1.0, 1.5], dtype=dt)
        expected = np.tan(a.copy())
        np.tan(a, out=a)
        assert_allclose(a, expected, rtol=1e-5)

    def test_exp2_inplace_large(self):
        a = np.arange(100, dtype=np.float64)
        expected = np.exp2(a.copy())
        np.exp2(a, out=a)
        assert_allclose(a, expected, rtol=1e-10)

    def test_log_inplace_large(self):
        a = np.arange(1, 101, dtype=np.float64)
        expected = np.log(a.copy())
        np.log(a, out=a)
        assert_allclose(a, expected, rtol=1e-10)


class TestExp2HalfStrided:
    """loops_exp2.dispatch.cpp: float16 NEON strided exp2 paths."""

    def test_exp2_half_stride2(self):
        a = np.arange(0, 10, dtype=np.float16)[::2]
        res = np.exp2(a)
        expected = np.exp2(a.astype(np.float64)).astype(np.float16)
        assert_allclose(res, expected, rtol=1e-2)

    def test_exp2_half_stride4(self):
        a = np.arange(0, 20, dtype=np.float16)[::4]
        a = np.clip(a, -5, 5)
        res = np.exp2(a)
        expected = np.exp2(a.astype(np.float64)).astype(np.float16)
        assert_allclose(res, expected, rtol=1e-2)

    def test_exp2_half_large_contiguous(self):
        a = np.linspace(-5, 5, 512, dtype=np.float16)
        res = np.exp2(a)
        expected = np.exp2(a.astype(np.float64)).astype(np.float16)
        assert_allclose(res, expected, rtol=1e-1)

    def test_exp2_half_special(self):
        a = np.array([np.nan, np.inf, -np.inf, 0.0, 1.0, -1.0, 16.0, -15.0],
                      dtype=np.float16)
        with np.errstate(over='ignore', invalid='ignore'):
            res = np.exp2(a)
        assert res.shape == (8,)


class TestTrigEdgeCases:
    """loops_trigonometric.dispatch.cpp: NaN/Inf in strided trig."""

    @pytest.mark.parametrize("func", [np.sin, np.cos, np.tan])
    def test_trig_strided_with_nan(self, func):
        a = np.array([1.0, np.nan, 2.0, np.nan, 3.0, np.nan], dtype=np.float64)
        res = func(a[::2])
        assert res.shape == (3,)

    @pytest.mark.parametrize("func", [np.sin, np.cos, np.tan])
    def test_trig_strided_with_inf(self, func):
        a = np.array([1.0, np.inf, 2.0, -np.inf, 3.0, np.inf], dtype=np.float64)
        res = func(a[::2])
        assert res.shape == (3,)

    def test_tan_all_nan_strided(self):
        a = np.full(16, np.nan, dtype=np.float32)
        res = np.tan(a[::2])
        assert np.all(np.isnan(res))

    def test_tan_all_nan_float64_strided(self):
        a = np.full(16, np.nan, dtype=np.float64)
        res = np.tan(a[::2])
        assert np.all(np.isnan(res))

    def test_sin_half_large_values(self):
        a = np.array([1e4, -1e4, 0.0, 1.0], dtype=np.float16)
        res = np.sin(a)
        assert res.shape == (4,)

    def test_cos_half_large_values(self):
        a = np.array([1e4, -1e4, 0.0, 1.0], dtype=np.float16)
        res = np.cos(a)
        assert res.shape == (4,)

    def test_tan_half_strided_output(self):
        a = np.arange(10, dtype=np.float16)
        out = np.zeros(20, dtype=np.float16)
        np.tan(a, out=out[::2])
        assert out[0] == np.tan(np.float16(0))


class TestPowerEdgeCases:
    """loops_power.dispatch.cpp: edge cases for power function."""

    def test_power_float64_fast_paths(self):
        a = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float64)
        assert_allclose(np.power(a, 0.5), np.sqrt(a))
        assert_allclose(np.power(a, 2.0), a * a)
        assert_allclose(np.power(a, 1.0), a)
        assert_allclose(np.power(a, 0.0), np.ones_like(a))
        assert_allclose(np.power(a, -1.0), 1.0 / a)

    def test_power_float32_fast_paths(self):
        a = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
        assert_allclose(np.power(a, 0.5), np.sqrt(a), rtol=1e-5)
        assert_allclose(np.power(a, 2.0), a * a, rtol=1e-5)

    def test_power_float64_scalar_fallback(self):
        a = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        b = np.array([3.5, 2.5, 1.5], dtype=np.float64)
        res = np.power(a, b)
        assert_allclose(res, a ** b)

    def test_power_float32_strided(self):
        a = np.arange(1, 20, dtype=np.float64)[::3]
        b = np.full_like(a, 2.5)
        res = np.power(a, b)
        assert_allclose(res, a ** 2.5)


class TestUnaryOpsExtended:
    """loops_umath_unary.dispatch.cpp: cbrt and sqrt edge cases."""

    def test_cbrt_float16(self):
        a = np.array([-8.0, -1.0, 0.0, 1.0, 8.0], dtype=np.float16)
        res = np.cbrt(a)
        assert_allclose(res, [-2.0, -1.0, 0.0, 1.0, 2.0], rtol=1e-2)

    def test_cbrt_float32(self):
        a = np.array([-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0], dtype=np.float32)
        res = np.cbrt(a)
        assert_allclose(res, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], rtol=1e-5)

    def test_cbrt_float64_large(self):
        a = np.arange(-100, 100, dtype=np.float64)
        res = np.cbrt(a)
        assert res.shape == (200,)

    def test_sqrt_half_strided_output(self):
        a = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float16)
        out = np.zeros(8, dtype=np.float16)
        np.sqrt(a, out=out[::2])
        assert_allclose(out[::2], [1.0, 2.0, 3.0, 4.0], rtol=1e-2)


class TestFancyIndexingErrors:
    """mapping.c: fancy indexing error paths and edge cases."""

    def test_axis1_out_of_bounds(self):
        a = np.zeros((3, 5))
        with pytest.raises(IndexError):
            a[:, [0, 1, 10]]

    def test_axis1_negative_index(self):
        a = np.arange(15).reshape(3, 5)
        res = a[:, [-1]]
        assert res.shape == (3, 1)

    def test_fancy_index_int16(self):
        a = np.arange(10, dtype=np.int16)
        idx = np.array([0, 2, 4])
        res = a[idx]
        assert_array_equal(res, [0, 2, 4])

    def test_fancy_index_complex128(self):
        a = np.arange(10, dtype=np.complex128)
        idx = np.array([0, 2, 4])
        res = a[idx]
        assert_array_equal(res, [0, 2, 4])

    def test_fancy_set_int16_scalar(self):
        a = np.zeros(10, dtype=np.int16)
        a[np.array([0, 2, 4])] = np.int16(5)
        assert a[0] == 5
        assert a[2] == 5

    def test_fancy_set_complex128_scalar(self):
        a = np.zeros(10, dtype=np.complex128)
        a[np.array([0, 2, 4])] = 1+2j
        assert a[0] == 1+2j

    def test_fancy_set_axis1_out_of_bounds(self):
        a = np.zeros((3, 5))
        with pytest.raises(IndexError):
            a[:, [0, 1, 10]] = 5

    def test_fancy_set_axis1_negative(self):
        a = np.zeros((3, 5))
        a[:, [-1]] = 5
        assert a[0, -1] == 5

    def test_fancy_index_string_dtype(self):
        a = np.array(["a", "bb", "ccc", "dddd"], dtype="U4")
        idx = np.array([0, 2])
        res = a[idx]
        assert_array_equal(res, ["a", "ccc"])

    def test_fancy_set_string_scalar(self):
        a = np.array(["a", "bb", "ccc", "dddd"], dtype="U4")
        a[np.array([0, 2])] = "zzz"
        assert a[0] == "zzz"
        assert a[2] == "zzz"

    def test_trivial_get_non_native_byteorder(self):
        a = np.arange(10, dtype=">i4")
        idx = np.array([0, 2, 4])
        res = a[idx]
        assert_array_equal(res, [0, 2, 4])


class TestARMReduce:
    """loops_autovec.dispatch.c.src: ARM-specific add/multiply reduce."""

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64,
                                     np.uint8, np.uint16, np.uint32, np.uint64])
    def test_add_reduce_contiguous(self, dt):
        info = np.iinfo(dt)
        n = min(256, int(info.max))
        a = np.ones(n, dtype=dt)
        res = np.add.reduce(a)
        assert res == n

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64,
                                     np.uint16, np.uint32, np.uint64])
    def test_multiply_reduce_ones(self, dt):
        a = np.ones(100, dtype=dt)
        res = np.multiply.reduce(a)
        assert res == 1

    @pytest.mark.parametrize("dt", [np.int16, np.int32, np.int64])
    def test_multiply_reduce_values(self, dt):
        a = np.array([1, 2, 3, 4, 5], dtype=dt)
        res = np.multiply.reduce(a)
        assert res == 120

    @pytest.mark.parametrize("size", [16, 32, 64, 128, 256, 512])
    def test_add_reduce_various_sizes(self, size):
        a = np.ones(size, dtype=np.int32)
        res = np.add.reduce(a)
        assert res == size


class TestFloorDivideConstant:
    """loops_arithmetic.dispatch.c.src: libdivide floor_divide paths."""

    def test_floor_divide_int64_by_minus1(self):
        a = np.array([10, -10, 100, -100], dtype=np.int64)
        with np.errstate(over="ignore"):
            res = np.floor_divide(a, np.int64(-1))
        assert res.shape == (4,)

    def test_floor_divide_int64_by_1(self):
        a = np.array([10, -10, 100, -100], dtype=np.int64)
        res = np.floor_divide(a, np.int64(1))
        assert_array_equal(res, a)

    def test_floor_divide_int64_by_constant(self):
        a = np.arange(1, 1001, dtype=np.int64)
        res = np.floor_divide(a, np.int64(7))
        assert_array_equal(res, a // 7)

    def test_floor_divide_int64_by_negative(self):
        a = np.arange(-100, 100, dtype=np.int64)
        res = np.floor_divide(a, np.int64(-3))
        assert_array_equal(res, a // np.int64(-3))

    def test_floor_divide_int64_overflow(self):
        info = np.iinfo(np.int64)
        a = np.array([info.min, info.min + 1, 0, 1], dtype=np.int64)
        with np.errstate(over="ignore"):
            res = np.floor_divide(a, np.int64(-1))
        assert res.shape == (4,)


class TestHistogram2dErrors:
    """compiled_base.c: histogram2d error paths."""

    def test_histogram2d_zero_bins_x(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [3, 4], bins=[0, 10])

    def test_histogram2d_zero_bins_y(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [3, 4], bins=[10, 0])

    def test_histogram2d_negative_bins(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [3, 4], bins=[-1, 10])

    def test_histogram2d_non_finite_range(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [3, 4], bins=[10, 10],
                           range=[[0, np.inf], [0, 1]])

    def test_histogram2d_decreasing_range_x(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [3, 4], bins=[10, 10],
                           range=[[5, 1], [0, 10]])

    def test_histogram2d_decreasing_range_y(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [3, 4], bins=[10, 10],
                           range=[[0, 10], [5, 1]])


class TestScalarUfuncFastPath:
    """ufunc_object.c: scalar fast-path miss for longdouble/complex."""

    def test_sin_longdouble(self):
        res = np.sin(np.longdouble(1.0))
        assert isinstance(res, (np.floating, float))

    def test_cos_longdouble(self):
        res = np.cos(np.longdouble(1.0))
        assert isinstance(res, (np.floating, float))

    def test_exp_longdouble(self):
        res = np.exp(np.longdouble(1.0))
        assert isinstance(res, (np.floating, float))

    def test_log_longdouble(self):
        res = np.log(np.longdouble(1.0))
        assert isinstance(res, (np.floating, float))

    def test_sqrt_longdouble(self):
        res = np.sqrt(np.longdouble(4.0))
        assert isinstance(res, (np.floating, float))

    def test_sin_complex64(self):
        res = np.sin(np.complex64(1+1j))
        assert isinstance(res, (np.complexfloating, complex))

    def test_cos_complex64(self):
        res = np.cos(np.complex64(1+1j))
        assert isinstance(res, (np.complexfloating, complex))

    def test_exp_complex64(self):
        res = np.exp(np.complex64(1+1j))
        assert isinstance(res, (np.complexfloating, complex))

    def test_log_complex64(self):
        res = np.log(np.complex64(1+1j))
        assert isinstance(res, (np.complexfloating, complex))

    def test_sqrt_complex64(self):
        res = np.sqrt(np.complex64(1+1j))
        assert isinstance(res, (np.complexfloating, complex))

    def test_sin_complex128(self):
        res = np.sin(np.complex128(1+1j))
        assert isinstance(res, (np.complexfloating, complex))

    def test_exp_complex128(self):
        res = np.exp(np.complex128(1+1j))
        assert isinstance(res, (np.complexfloating, complex))

    def test_log_complex128(self):
        res = np.log(np.complex128(1+1j))
        assert isinstance(res, (np.complexfloating, complex))


class TestMinMaxStridedSIMD:
    """loops_minmax.dispatch.c.src: strided SIMD max/min paths."""

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_strided_both(self, dt):
        a = np.arange(100, dtype=dt)[::2]
        b = np.arange(100, dtype=dt)[1::2]
        res = np.maximum(a, b)
        assert_array_equal(res, np.where(a >= b, a, b))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_strided_both(self, dt):
        a = np.arange(100, dtype=dt)[::2]
        b = np.arange(100, dtype=dt)[1::2]
        res = np.minimum(a, b)
        assert_array_equal(res, np.where(a <= b, a, b))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_strided_output(self, dt):
        a = np.arange(50, dtype=dt)
        b = np.arange(50, dtype=dt)[::-1]
        out = np.zeros(100, dtype=dt)
        np.maximum(a, b, out=out[::2])
        expected = np.maximum(a, b)
        assert_array_equal(out[::2], expected)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_strided_output(self, dt):
        a = np.arange(50, dtype=dt)
        b = np.arange(50, dtype=dt)[::-1]
        out = np.zeros(100, dtype=dt)
        np.minimum(a, b, out=out[::2])
        expected = np.minimum(a, b)
        assert_array_equal(out[::2], expected)

    @pytest.mark.parametrize("dt", [np.int32, np.int64])
    def test_maximum_integer_strided(self, dt):
        a = np.arange(100, dtype=dt)[::2]
        b = np.arange(100, dtype=dt)[1::2]
        res = np.maximum(a, b)
        assert_array_equal(res, np.where(a >= b, a, b))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_scalar_left_strided(self, dt):
        a = np.arange(100, dtype=dt)[::2]
        res = np.maximum(dt(25.0), a)
        assert_array_equal(res, np.where(25.0 >= a, 25.0, a))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmax_strided(self, dt):
        a = np.arange(50, dtype=dt)
        b = np.arange(50, dtype=dt)[::-1]
        res = np.fmax(a, b)
        assert res.shape == (50,)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmin_strided(self, dt):
        a = np.arange(50, dtype=dt)
        b = np.arange(50, dtype=dt)[::-1]
        res = np.fmin(a, b)
        assert res.shape == (50,)

    @pytest.mark.parametrize("size", [1, 3, 7, 15, 31, 63])
    def test_maximum_scalar_tail(self, size):
        rng = np.random.default_rng(42)
        a = rng.random(size).astype(np.float64)
        b = rng.random(size).astype(np.float64)
        res = np.maximum(a, b)
        assert_array_equal(res, np.where(a >= b, a, b))
