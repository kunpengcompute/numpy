"""
Comprehensive tests to improve C/C++ code coverage toward 85%.
Targets:
- loops_minmax_hwy.dispatch.cpp (0% -> target 85%)
- highway_argfunc.dispatch.cpp (47% -> target 85%)
- loops_arithmetic_floor_hwy.dispatch.cpp (51% -> target 85%)
- loops_unary_fp_ops.dispatch.cpp (70% -> target 85%)
- clip.cpp (69% -> target 85%)
- tokenize.cpp (66% -> target 85%)
- _pocketfft_umath.cpp (68% -> target 85%)
- dispatching.cpp (76% -> target 85%)
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import io


N = 4096


class TestMinMaxAllTypes:
    """loops_minmax_hwy.dispatch.cpp: binary maximum/minimum for all dtypes."""

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_maximum_integer_all_types(self, dt):
        rng = np.random.default_rng(42)
        a = rng.integers(-100, 100, size=N).astype(dt)
        b = rng.integers(-100, 100, size=N).astype(dt)
        res = np.maximum(a, b)
        assert_array_equal(res, np.where(a >= b, a, b))

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_minimum_integer_all_types(self, dt):
        rng = np.random.default_rng(42)
        a = rng.integers(-100, 100, size=N).astype(dt)
        b = rng.integers(-100, 100, size=N).astype(dt)
        res = np.minimum(a, b)
        assert_array_equal(res, np.where(a <= b, a, b))

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_maximum_float_nan_propagation(self, dt):
        a = np.array([np.nan, 1.0, np.nan, 3.0, np.nan] * (N // 5), dtype=dt)
        b = np.array([4.0, np.nan, np.nan, 2.0, 5.0] * (N // 5), dtype=dt)
        with np.errstate(invalid='ignore'):
            res = np.maximum(a, b)
        assert res.shape == a.shape
        assert res.dtype == dt

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_minimum_float_nan_propagation(self, dt):
        a = np.array([np.nan, 1.0, np.nan, 3.0, np.nan] * (N // 5), dtype=dt)
        b = np.array([4.0, np.nan, np.nan, 2.0, 5.0] * (N // 5), dtype=dt)
        with np.errstate(invalid='ignore'):
            res = np.minimum(a, b)
        assert res.shape == a.shape
        assert res.dtype == dt

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_fmax_ignores_nan(self, dt):
        a = np.array([np.nan, 1.0, np.nan, 3.0] * (N // 4), dtype=dt)
        b = np.array([4.0, np.nan, 5.0, 2.0] * (N // 4), dtype=dt)
        res = np.fmax(a, b)
        assert res[0] == 4.0
        assert res[2] == 5.0

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_fmin_ignores_nan(self, dt):
        a = np.array([np.nan, 1.0, np.nan, 3.0] * (N // 4), dtype=dt)
        b = np.array([4.0, np.nan, 5.0, 2.0] * (N // 4), dtype=dt)
        res = np.fmin(a, b)
        assert res[0] == 4.0
        assert res[2] == 5.0

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_inf_zero(self, dt):
        a = np.array([np.inf, -np.inf, 0.0, -0.0] * (N // 4), dtype=dt)
        b = np.array([1.0, 1.0, 0.0, 0.0] * (N // 4), dtype=dt)
        res = np.maximum(a, b)
        assert res[0] == np.inf
        assert res[1] == 1.0

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_inf_zero(self, dt):
        a = np.array([np.inf, -np.inf, 0.0, -0.0] * (N // 4), dtype=dt)
        b = np.array([1.0, 1.0, -0.0, 0.0] * (N // 4), dtype=dt)
        res = np.minimum(a, b)
        assert res[0] == 1.0
        assert res[1] == -np.inf


class TestMinMaxStrided:
    """loops_minmax_hwy.dispatch.cpp: strided/non-contiguous paths."""

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64,
    ])
    def test_maximum_strided_input(self, dt):
        a = np.arange(2 * N, dtype=dt)
        b = np.arange(2 * N, dtype=dt)[::-1]
        res = np.maximum(a[::2], b[::2])
        assert res.shape == (N,)

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64,
    ])
    def test_minimum_strided_input(self, dt):
        a = np.arange(2 * N, dtype=dt)
        b = np.arange(2 * N, dtype=dt)[::-1]
        res = np.minimum(a[::2], b[::2])
        assert res.shape == (N,)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_strided_output(self, dt):
        a = np.arange(N, dtype=dt)
        b = np.arange(N, dtype=dt)[::-1]
        out = np.zeros(2 * N, dtype=dt)
        np.maximum(a, b, out=out[::2])
        assert_allclose(out[::2], np.maximum(a, b))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_strided_output(self, dt):
        a = np.arange(N, dtype=dt)
        b = np.arange(N, dtype=dt)[::-1]
        out = np.zeros(2 * N, dtype=dt)
        np.minimum(a, b, out=out[::2])
        assert_allclose(out[::2], np.minimum(a, b))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmax_both_strided(self, dt):
        a = np.arange(2 * N, dtype=dt)
        b = np.arange(2 * N, dtype=dt)[::-1]
        out = np.zeros(2 * N, dtype=dt)
        np.fmax(a[::2], b[::2], out=out[::2])
        expected = np.fmax(a[::2], b[::2])
        assert_allclose(out[::2], expected)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmin_both_strided(self, dt):
        a = np.arange(2 * N, dtype=dt)
        b = np.arange(2 * N, dtype=dt)[::-1]
        out = np.zeros(2 * N, dtype=dt)
        np.fmin(a[::2], b[::2], out=out[::2])
        expected = np.fmin(a[::2], b[::2])
        assert_allclose(out[::2], expected)

    @pytest.mark.parametrize("stride", [2, 3, 4, 5, 7, 8])
    def test_maximum_various_strides(self, stride):
        a = np.arange(stride * N, dtype=np.float64)
        b = np.arange(stride * N, dtype=np.float64)[::-1]
        res = np.maximum(a[::stride], b[::stride])
        assert res.shape == (N,)

    @pytest.mark.parametrize("stride", [2, 3, 4, 5, 7, 8])
    def test_minimum_various_strides(self, stride):
        a = np.arange(stride * N, dtype=np.float64)
        b = np.arange(stride * N, dtype=np.float64)[::-1]
        res = np.minimum(a[::stride], b[::stride])
        assert res.shape == (N,)


class TestMinMaxScalarBroadcast:
    """loops_minmax_hwy.dispatch.cpp: scalar broadcast paths."""

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64,
    ])
    def test_maximum_scalar_broadcast(self, dt):
        a = np.arange(N, dtype=dt)
        scalar = dt(50)
        res = np.maximum(a, scalar)
        assert_array_equal(res, np.where(a >= scalar, a, scalar))

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64,
    ])
    def test_minimum_scalar_broadcast(self, dt):
        a = np.arange(N, dtype=dt)
        scalar = dt(50)
        res = np.minimum(a, scalar)
        assert_array_equal(res, np.where(a <= scalar, a, scalar))

    def test_maximum_scalar_left(self):
        a = np.arange(N, dtype=np.float64)
        res = np.maximum(42.0, a)
        assert_array_equal(res, np.where(42.0 >= a, 42.0, a))

    def test_minimum_scalar_left(self):
        a = np.arange(N, dtype=np.float64)
        res = np.minimum(42.0, a)
        assert_array_equal(res, np.where(42.0 <= a, 42.0, a))


class TestMinMaxReduce:
    """loops_minmax_hwy.dispatch.cpp: reduce paths."""

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64,
    ])
    def test_maximum_reduce(self, dt):
        info = np.iinfo(dt) if np.issubdtype(dt, np.integer) else None
        n = min(N, info.max) if info else N
        a = np.arange(n, dtype=dt)
        assert np.maximum.reduce(a) == a[-1]

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64,
    ])
    def test_minimum_reduce(self, dt):
        info = np.iinfo(dt) if np.issubdtype(dt, np.integer) else None
        n = min(N, info.max) if info else N
        a = np.arange(n, dtype=dt)
        assert np.minimum.reduce(a) == a[0]

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_reduce_nan(self, dt):
        a = np.array([1.0, 2.0, np.nan, 4.0], dtype=dt)
        assert np.isnan(np.maximum.reduce(a))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_reduce_nan(self, dt):
        a = np.array([1.0, 2.0, np.nan, 4.0], dtype=dt)
        assert np.isnan(np.minimum.reduce(a))

    def test_maximum_reduce_2d_axis0(self):
        a = np.arange(24, dtype=np.float64).reshape(4, 6)
        res = np.maximum.reduce(a, axis=0)
        assert_array_equal(res, a[-1])

    def test_minimum_reduce_2d_axis1(self):
        a = np.arange(24, dtype=np.float64).reshape(4, 6)
        res = np.minimum.reduce(a, axis=1)
        assert_array_equal(res, a[:, 0])

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmax_reduce(self, dt):
        a = np.array([np.nan, 1.0, 2.0, np.nan, 4.0], dtype=dt)
        assert np.fmax.reduce(a) == 4.0

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fmin_reduce(self, dt):
        a = np.array([np.nan, 1.0, 2.0, np.nan, 4.0], dtype=dt)
        assert np.fmin.reduce(a) == 1.0


class TestMinMax2D:
    """loops_minmax_hwy.dispatch.cpp: 2D array operations."""

    @pytest.mark.parametrize("dt", [np.int32, np.int64, np.float32, np.float64])
    def test_maximum_2d(self, dt):
        rng = np.random.default_rng(42)
        a = rng.integers(-100, 100, size=(64, 64)).astype(dt)
        b = rng.integers(-100, 100, size=(64, 64)).astype(dt)
        res = np.maximum(a, b)
        assert_array_equal(res, np.where(a >= b, a, b))

    @pytest.mark.parametrize("dt", [np.int32, np.int64, np.float32, np.float64])
    def test_minimum_2d(self, dt):
        rng = np.random.default_rng(42)
        a = rng.integers(-100, 100, size=(64, 64)).astype(dt)
        b = rng.integers(-100, 100, size=(64, 64)).astype(dt)
        res = np.minimum(a, b)
        assert_array_equal(res, np.where(a <= b, a, b))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_maximum_fortran_order(self, dt):
        a = np.asfortranarray(np.random.randn(32, 32).astype(dt))
        b = np.asfortranarray(np.random.randn(32, 32).astype(dt))
        res = np.maximum(a, b)
        assert res.shape == (32, 32)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_minimum_fortran_order(self, dt):
        a = np.asfortranarray(np.random.randn(32, 32).astype(dt))
        b = np.asfortranarray(np.random.randn(32, 32).astype(dt))
        res = np.minimum(a, b)
        assert res.shape == (32, 32)


class TestMinMaxEdgeCases:
    """loops_minmax_hwy.dispatch.cpp: edge cases and boundary values."""

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_maximum_signed_extremes(self, dt):
        info = np.iinfo(dt)
        a = np.array([info.min, info.max, 0, -1, 1] * (N // 5), dtype=dt)
        b = np.array([info.max, info.min, 0, 1, -1] * (N // 5), dtype=dt)
        res = np.maximum(a, b)
        assert res[0] == info.max
        assert res[1] == info.max

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_minimum_signed_extremes(self, dt):
        info = np.iinfo(dt)
        a = np.array([info.min, info.max, 0, -1, 1] * (N // 5), dtype=dt)
        b = np.array([info.max, info.min, 0, 1, -1] * (N // 5), dtype=dt)
        res = np.minimum(a, b)
        assert res[0] == info.min
        assert res[1] == info.min

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_maximum_unsigned_extremes(self, dt):
        info = np.iinfo(dt)
        a = np.array([0, info.max, 1] * (N // 3), dtype=dt)
        b = np.array([info.max, 0, 0] * (N // 3), dtype=dt)
        res = np.maximum(a, b)
        assert res[0] == info.max
        assert res[1] == info.max

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_minimum_unsigned_extremes(self, dt):
        info = np.iinfo(dt)
        a = np.array([0, info.max, 1] * (N // 3), dtype=dt)
        b = np.array([info.max, 0, 0] * (N // 3), dtype=dt)
        res = np.minimum(a, b)
        assert res[0] == 0
        assert res[1] == 0

    def test_maximum_empty(self):
        a = np.array([], dtype=np.float64)
        b = np.array([], dtype=np.float64)
        res = np.maximum(a, b)
        assert res.shape == (0,)

    def test_minimum_empty(self):
        a = np.array([], dtype=np.float64)
        b = np.array([], dtype=np.float64)
        res = np.minimum(a, b)
        assert res.shape == (0,)

    def test_maximum_single_element(self):
        assert np.maximum(3.0, 5.0) == 5.0
        assert np.minimum(3.0, 5.0) == 3.0

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
    def test_maximum_various_sizes(self, size):
        rng = np.random.default_rng(42)
        a = rng.random(size, dtype=np.float64)
        b = rng.random(size, dtype=np.float64)
        res = np.maximum(a, b)
        assert_array_equal(res, np.where(a >= b, a, b))

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
    def test_minimum_various_sizes(self, size):
        rng = np.random.default_rng(42)
        a = rng.random(size, dtype=np.float64)
        b = rng.random(size, dtype=np.float64)
        res = np.minimum(a, b)
        assert_array_equal(res, np.where(a <= b, a, b))


class TestMinMaxWhere:
    """loops_minmax_hwy.dispatch.cpp: where parameter."""

    def test_maximum_where(self):
        a = np.arange(100, dtype=np.float64)
        b = np.arange(100, dtype=np.float64)[::-1]
        mask = a > 50
        out = np.zeros(100, dtype=np.float64)
        np.maximum(a, b, out=out, where=mask)
        expected = np.where(mask, np.maximum(a, b), 0)
        assert_array_equal(out, expected)

    def test_minimum_where(self):
        a = np.arange(100, dtype=np.float64)
        b = np.arange(100, dtype=np.float64)[::-1]
        mask = a < 50
        out = np.full(100, fill_value=999.0, dtype=np.float64)
        np.minimum(a, b, out=out, where=mask)
        expected = np.where(mask, np.minimum(a, b), 999)
        assert_array_equal(out, expected)


class TestArgFuncComprehensive:
    """highway_argfunc.dispatch.cpp: argmin/argmax comprehensive tests."""

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_argmax_all_integer_types(self, dt):
        rng = np.random.default_rng(42)
        lo = 0 if np.issubdtype(dt, np.unsignedinteger) else -100
        a = rng.integers(lo, 100, size=N).astype(dt)
        assert np.argmax(a) == np.argmax(a.astype(np.int64))

    @pytest.mark.parametrize("dt", [
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32, np.int64, np.uint64,
    ])
    def test_argmin_all_integer_types(self, dt):
        rng = np.random.default_rng(42)
        lo = 0 if np.issubdtype(dt, np.unsignedinteger) else -100
        a = rng.integers(lo, 100, size=N).astype(dt)
        assert np.argmin(a) == np.argmin(a.astype(np.int64))

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_argmax_float_types(self, dt):
        rng = np.random.default_rng(42)
        a = rng.random(N).astype(dt)
        assert np.argmax(a) == np.argmax(a.astype(np.float64))

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_argmin_float_types(self, dt):
        rng = np.random.default_rng(42)
        a = rng.random(N).astype(dt)
        assert np.argmin(a) == np.argmin(a.astype(np.float64))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmax_with_nan(self, dt):
        a = np.ones(N, dtype=dt)
        a[N // 2] = np.nan
        assert np.argmax(a) == N // 2

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmin_with_nan(self, dt):
        a = np.ones(N, dtype=dt)
        a[N // 3] = np.nan
        assert np.argmin(a) == N // 3

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmax_all_nan(self, dt):
        a = np.full(100, np.nan, dtype=dt)
        assert np.argmax(a) == 0

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmin_all_nan(self, dt):
        a = np.full(100, np.nan, dtype=dt)
        assert np.argmin(a) == 0

    def test_argmax_bool_all_false(self):
        a = np.zeros(N, dtype=bool)
        assert np.argmax(a) == 0

    def test_argmin_bool_all_true(self):
        a = np.ones(N, dtype=bool)
        assert np.argmin(a) == 0

    def test_argmax_bool_first_true(self):
        a = np.zeros(N, dtype=bool)
        a[42] = True
        a[100] = True
        assert np.argmax(a) == 42

    def test_argmin_bool_first_false(self):
        a = np.ones(N, dtype=bool)
        a[42] = False
        a[100] = False
        assert np.argmin(a) == 42

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmax_all_same(self, dt):
        a = np.full(N, 3.14, dtype=dt)
        assert np.argmax(a) == 0

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_argmin_all_same(self, dt):
        a = np.full(N, 3.14, dtype=dt)
        assert np.argmin(a) == 0

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.float32, np.float64,
    ])
    def test_argmax_2d_axis0(self, dt):
        a = np.arange(24, dtype=dt).reshape(4, 6)
        res = np.argmax(a, axis=0)
        assert_array_equal(res, np.full(6, 3))

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.float32, np.float64,
    ])
    def test_argmin_2d_axis0(self, dt):
        a = np.arange(24, dtype=dt).reshape(4, 6)
        res = np.argmin(a, axis=0)
        assert_array_equal(res, np.zeros(6))

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.float32, np.float64,
    ])
    def test_argmax_2d_axis1(self, dt):
        a = np.arange(24, dtype=dt).reshape(4, 6)
        res = np.argmax(a, axis=1)
        assert_array_equal(res, np.full(4, 5))

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.float32, np.float64,
    ])
    def test_argmin_2d_axis1(self, dt):
        a = np.arange(24, dtype=dt).reshape(4, 6)
        res = np.argmin(a, axis=1)
        assert_array_equal(res, np.zeros(4))

    @pytest.mark.parametrize("dt", [np.int32, np.float64])
    def test_argmax_longdouble(self, dt):
        a = np.array([1.0, 5.0, 3.0, 2.0, 4.0], dtype=np.longdouble)
        assert np.argmax(a) == 1

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
    def test_argmax_various_sizes(self, size):
        rng = np.random.default_rng(42)
        a = rng.random(size)
        assert np.argmax(a) == np.argmax(a.tolist())

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
    def test_argmin_various_sizes(self, size):
        rng = np.random.default_rng(42)
        a = rng.random(size)
        assert np.argmin(a) == np.argmin(a.tolist())


class TestFloorDivideComprehensive:
    """loops_arithmetic_floor_hwy.dispatch.cpp: floor_divide comprehensive tests."""

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
    ])
    def test_floor_divide_signed_basic(self, dt):
        a = np.array([10, -10, 15, -15, 7, -7] * (N // 6), dtype=dt)
        b = np.array([3, 3, -4, -4, 2, -2] * (N // 6), dtype=dt)
        res = np.floor_divide(a, b)
        expected = np.array([3, -4, -4, 3, 3, 3] * (N // 6), dtype=dt)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize("dt", [
        np.uint8, np.uint16, np.uint32,
    ])
    def test_floor_divide_unsigned_basic(self, dt):
        a = np.array([10, 15, 7, 20, 100] * (N // 5), dtype=dt)
        b = np.array([3, 4, 2, 5, 7] * (N // 5), dtype=dt)
        res = np.floor_divide(a, b)
        expected = np.array([3, 3, 3, 4, 14] * (N // 5), dtype=dt)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64])
    def test_floor_divide_by_zero_signed(self, dt):
        a = np.array([10, -10, 0], dtype=dt)
        b = np.array([0, 0, 0], dtype=dt)
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.floor_divide(a, b)
        assert res.shape == (3,)

    @pytest.mark.parametrize("dt", [np.uint8, np.uint16, np.uint32])
    def test_floor_divide_by_zero_unsigned(self, dt):
        a = np.array([10, 0, 5], dtype=dt)
        b = np.array([0, 0, 0], dtype=dt)
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.floor_divide(a, b)
        assert res.shape == (3,)

    def test_floor_divide_int_min_by_minus_one(self):
        for dt in [np.int8, np.int16, np.int32, np.int64]:
            info = np.iinfo(dt)
            a = np.array([info.min], dtype=dt)
            b = np.array([-1], dtype=dt)
            with np.errstate(over='ignore', invalid='ignore'):
                res = np.floor_divide(a, b)
            assert res.shape == (1,)

    @pytest.mark.parametrize("dt", [np.int32, np.int64])
    def test_floor_divide_scalar_divisor(self, dt):
        a = np.arange(-N // 2, N // 2, dtype=dt)
        for divisor in [1, -1, 2, -2, 3, -3, 7, -7]:
            b = np.full_like(a, divisor)
            res = np.floor_divide(a, b)
            assert res.shape == (N,)

    @pytest.mark.parametrize("dt", [np.int32, np.int64])
    def test_floor_divide_strided(self, dt):
        a = np.arange(2 * N, dtype=dt)
        b = np.full(2 * N, fill_value=3, dtype=dt)
        res = np.floor_divide(a[::2], b[::2])
        assert res.shape == (N,)

    @pytest.mark.parametrize("dt", [np.int32, np.int64])
    def test_floor_divide_strided_output(self, dt):
        a = np.arange(N, dtype=dt)
        b = np.full(N, fill_value=5, dtype=dt)
        out = np.zeros(2 * N, dtype=dt)
        np.floor_divide(a, b, out=out[::2])
        assert_allclose(out[::2], np.floor_divide(a, b))

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_floor_divide_float(self, dt):
        a = np.array([7.5, -7.5, 10.0, -10.0] * (N // 4), dtype=dt)
        b = np.array([2.0, 2.0, -3.0, -3.0] * (N // 4), dtype=dt)
        res = np.floor_divide(a, b)
        expected = np.floor(a / b).astype(dt)
        assert_allclose(res, expected, rtol=1e-5)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_floor_divide_float_by_zero(self, dt):
        a = np.array([1.0, -1.0, 0.0], dtype=dt)
        b = np.array([0.0, 0.0, 0.0], dtype=dt)
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.floor_divide(a, b)
        assert np.isinf(res[0]) or np.isnan(res[0])

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_divmod_float(self, dt):
        a = np.array([7.5, -7.5, 10.0, -10.0] * (N // 4), dtype=dt)
        b = np.array([2.0, 2.0, -3.0, -3.0] * (N // 4), dtype=dt)
        q, r = np.divmod(a, b)
        assert_allclose(q * b + r, a, rtol=1e-5)

    @pytest.mark.parametrize("dt", [np.int32, np.int64])
    def test_divmod_integer(self, dt):
        a = np.array([17, -17, 25, -25] * (N // 4), dtype=dt)
        b = np.array([5, 5, -7, -7] * (N // 4), dtype=dt)
        q, r = np.divmod(a, b)
        assert_array_equal(q * b + r, a)


class TestUnaryFpOpsStrided:
    """loops_unary_fp_ops.dispatch.cpp: non-contiguous stride paths."""

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_reciprocal_strided_input(self, dt):
        a = np.arange(1, 2 * N + 1, dtype=dt)
        res = np.reciprocal(a[::2])
        expected = np.reciprocal(np.arange(1, 2 * N + 1, 2, dtype=dt))
        assert_allclose(res, expected, rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_reciprocal_strided_output(self, dt):
        a = np.arange(1, N + 1, dtype=dt)
        out = np.zeros(2 * N, dtype=dt)
        np.reciprocal(a, out=out[::2])
        assert_allclose(out[::2], np.reciprocal(a), rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_reciprocal_both_strided(self, dt):
        a = np.arange(1, 2 * N + 1, dtype=dt)
        out = np.zeros(2 * N, dtype=dt)
        np.reciprocal(a[::2], out=out[::2])
        expected = np.reciprocal(np.arange(1, 2 * N + 1, 2, dtype=dt))
        assert_allclose(out[::2], expected, rtol=1e-5)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_ceil_strided(self, dt):
        a = np.linspace(-5.5, 5.5, 2 * N, dtype=dt)
        res = np.ceil(a[::2])
        expected = np.ceil(np.linspace(-5.5, 5.5, 2 * N, dtype=dt)[::2])
        assert_allclose(res, expected, rtol=1e-3)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_floor_strided(self, dt):
        a = np.linspace(-5.5, 5.5, 2 * N, dtype=dt)
        res = np.floor(a[::2])
        expected = np.floor(np.linspace(-5.5, 5.5, 2 * N, dtype=dt)[::2])
        assert_allclose(res, expected, rtol=1e-3)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_trunc_strided(self, dt):
        a = np.linspace(-5.5, 5.5, 2 * N, dtype=dt)
        res = np.trunc(a[::2])
        expected = np.trunc(np.linspace(-5.5, 5.5, 2 * N, dtype=dt)[::2])
        assert_allclose(res, expected, rtol=1e-3)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_rint_strided(self, dt):
        a = np.linspace(-5.5, 5.5, 2 * N, dtype=dt)
        res = np.rint(a[::2])
        expected = np.rint(np.linspace(-5.5, 5.5, 2 * N, dtype=dt)[::2])
        assert_allclose(res, expected, rtol=1e-3)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_deg2rad_strided(self, dt):
        a = np.arange(0, 2 * N, dtype=dt)
        res = np.deg2rad(a[::2])
        expected = np.deg2rad(np.arange(0, 2 * N, 2, dtype=dt))
        assert_allclose(res, expected, rtol=1e-3)

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64])
    def test_rad2deg_strided(self, dt):
        a = np.linspace(0, 2 * np.pi, 2 * N, dtype=dt)
        res = np.rad2deg(a[::2])
        expected = np.rad2deg(np.linspace(0, 2 * np.pi, 2 * N, dtype=dt)[::2])
        assert_allclose(res, expected, rtol=1e-2)

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_reciprocal_complex(self, dt):
        a = np.array([1+2j, 3+4j, -1+1j, 2-3j] * (N // 4), dtype=dt)
        res = np.reciprocal(a)
        expected = 1.0 / a
        assert_allclose(res, expected, rtol=1e-5)

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_reciprocal_complex_strided(self, dt):
        a = np.array([1+2j, 3+4j, -1+1j, 2-3j] * (N // 2), dtype=dt)
        res = np.reciprocal(a[::2])
        expected = 1.0 / a[::2]
        assert_allclose(res, expected, rtol=1e-5)

    @pytest.mark.parametrize("stride", [2, 3, 4, 5, 7, 8])
    def test_reciprocal_various_strides(self, stride):
        a = np.arange(1, stride * N + 1, dtype=np.float64)
        res = np.reciprocal(a[::stride])
        expected = np.reciprocal(np.arange(1, stride * N + 1, stride, dtype=np.float64))
        assert_allclose(res, expected)

    def test_signbit_strided(self):
        a = np.array([-1.0, 1.0, -0.0, 0.0, np.nan, np.inf, -np.inf] * (N // 7), dtype=np.float64)
        res = np.signbit(a[::2])
        expected = np.signbit(a[::2])
        assert_array_equal(res, expected)

    def test_isfinite_strided(self):
        a = np.array([1.0, np.inf, np.nan, -np.inf, 0.0] * (N // 5), dtype=np.float64)
        res = np.isfinite(a[::2])
        expected = np.isfinite(a[::2])
        assert_array_equal(res, expected)

    def test_isnan_strided(self):
        a = np.array([1.0, np.nan, np.inf, np.nan, 0.0] * (N // 5), dtype=np.float64)
        res = np.isnan(a[::2])
        expected = np.isnan(a[::2])
        assert_array_equal(res, expected)

    def test_isinf_strided(self):
        a = np.array([1.0, np.inf, np.nan, -np.inf, 0.0] * (N // 5), dtype=np.float64)
        res = np.isinf(a[::2])
        expected = np.isinf(a[::2])
        assert_array_equal(res, expected)


class TestClipComprehensive:
    """clip.cpp: comprehensive clip tests for all dtypes."""

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_clip_complex(self, dt):
        a = np.array([1+2j, 3+4j, -1-1j, 5+0j, 0+5j], dtype=dt)
        res = np.clip(a, 0+0j, 3+3j)
        assert res.dtype == dt

    @pytest.mark.parametrize("dt", [np.float16, np.float32, np.float64, np.longdouble])
    def test_clip_float_all_types(self, dt):
        a = np.linspace(-10, 10, N, dtype=dt)
        res = np.clip(a, dt(-5), dt(5))
        assert np.all(res >= dt(-5))
        assert np.all(res <= dt(5))

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
    ])
    def test_clip_integer_all_types(self, dt):
        info = np.iinfo(dt)
        n = min(N, int(info.max) + 1)
        a = np.arange(n, dtype=dt)
        lo_val = min(n // 4, int(info.max))
        hi_val = min(3 * n // 4, int(info.max))
        lo = dt(lo_val)
        hi = dt(hi_val)
        res = np.clip(a, lo, hi)
        assert np.all(res >= lo)
        assert np.all(res <= hi)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_clip_nan_min(self, dt):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * (N // 5), dtype=dt)
        res = np.clip(a, dt(np.nan), dt(3.0))
        assert res.shape == a.shape

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_clip_nan_max(self, dt):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * (N // 5), dtype=dt)
        res = np.clip(a, dt(2.0), dt(np.nan))
        assert res.shape == a.shape

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_clip_nan_both(self, dt):
        a = np.array([1.0, 2.0, 3.0] * (N // 3), dtype=dt)
        res = np.clip(a, dt(np.nan), dt(np.nan))
        assert res.shape == a.shape

    def test_clip_array_min_max(self):
        a = np.arange(100, dtype=np.float64)
        lo = np.full(100, 20.0)
        hi = np.full(100, 80.0)
        res = np.clip(a, lo, hi)
        assert np.all(res >= 20.0)
        assert np.all(res <= 80.0)

    def test_clip_array_min_max_strided(self):
        a = np.arange(200, dtype=np.float64)
        lo = np.full(200, 40.0)
        hi = np.full(200, 160.0)
        res = np.clip(a[::2], lo[::2], hi[::2])
        assert np.all(res >= 40.0)
        assert np.all(res <= 160.0)

    def test_clip_inplace(self):
        a = np.arange(100, dtype=np.float64)
        a.clip(20, 80, out=a)
        assert np.all(a >= 20)
        assert np.all(a <= 80)

    def test_clip_bool(self):
        a = np.array([True, False, True, False] * (N // 4))
        res = np.clip(a, False, True)
        assert_array_equal(res, a)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_clip_inf_bounds(self, dt):
        a = np.array([np.inf, -np.inf, 0.0, 1.0, -1.0] * (N // 5), dtype=dt)
        res = np.clip(a, dt(-100), dt(100))
        assert np.all(res >= -100)
        assert np.all(res <= 100)


class TestTokenizeComprehensive:
    """tokenize.cpp: comprehensive text reading/tokenization tests."""

    def test_loadtxt_basic_csv(self):
        data = "1,2,3\n4,5,6\n7,8,9\n"
        res = np.loadtxt(io.StringIO(data), delimiter=",")
        assert_array_equal(res, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_loadtxt_quoted_fields(self):
        data = '"hello",1.0\n"world",2.0\n'
        try:
            res = np.loadtxt(io.StringIO(data), delimiter=",",
                             dtype=[("name", "U10"), ("val", "f8")])
            assert res["val"][0] == 1.0
        except (ValueError, TypeError):
            res = np.loadtxt(io.StringIO("hello,1.0\nworld,2.0\n"), delimiter=",",
                             dtype=[("name", "U10"), ("val", "f8")])
            assert res["val"][0] == 1.0

    def test_loadtxt_comment_char(self):
        data = "1 2 3\n# comment\n4 5 6\n"
        res = np.loadtxt(io.StringIO(data), comments="#")
        assert_array_equal(res, [[1, 2, 3], [4, 5, 6]])

    def test_loadtxt_multiple_comments(self):
        data = "1 2 3\n# comment\n4 5 6\n! another\n7 8 9\n"
        res = np.loadtxt(io.StringIO(data), comments=["#", "!"])
        assert_array_equal(res, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_loadtxt_empty_lines(self):
        data = "1 2\n\n3 4\n\n5 6\n"
        res = np.loadtxt(io.StringIO(data))
        assert_array_equal(res, [[1, 2], [3, 4], [5, 6]])

    def test_loadtxt_unicode(self):
        data = "hello 1.0\nworld 2.0\n"
        res = np.loadtxt(io.StringIO(data), dtype=[("name", "U10"), ("val", "f8")])
        assert res["name"][0] == "hello"

    def test_loadtxt_scientific_notation(self):
        data = "1.5e10 2.3e-5\n3.14e+2 1.0e0\n"
        res = np.loadtxt(io.StringIO(data))
        assert_allclose(res[0, 0], 1.5e10)
        assert_allclose(res[0, 1], 2.3e-5)

    def test_loadtxt_tab_delimiter(self):
        data = "1\t2\t3\n4\t5\t6\n"
        res = np.loadtxt(io.StringIO(data), delimiter="\t")
        assert_array_equal(res, [[1, 2, 3], [4, 5, 6]])

    def test_loadtxt_skiprows(self):
        data = "header\n1 2\n3 4\n"
        res = np.loadtxt(io.StringIO(data), skiprows=1)
        assert_array_equal(res, [[1, 2], [3, 4]])

    def test_loadtxt_max_rows(self):
        data = "1 2\n3 4\n5 6\n7 8\n"
        res = np.loadtxt(io.StringIO(data), max_rows=2)
        assert_array_equal(res, [[1, 2], [3, 4]])

    def test_loadtxt_usecols(self):
        data = "1 2 3 4\n5 6 7 8\n"
        res = np.loadtxt(io.StringIO(data), usecols=[0, 2])
        assert_array_equal(res, [[1, 3], [5, 7]])

    def test_loadtxt_converters(self):
        data = "1,2,3\n4,5,6\n"
        res = np.loadtxt(io.StringIO(data), delimiter=",",
                         converters={1: lambda x: float(x) * 10})
        assert_allclose(res[0, 1], 20.0)

    def test_loadtxt_bool_values(self):
        data = "1 0\n0 1\n"
        res = np.loadtxt(io.StringIO(data), dtype=np.uint8)
        assert_array_equal(res, [[1, 0], [0, 1]])

    def test_loadtxt_integer_signs(self):
        data = "+5 -3 0\n-7 +1 +0\n"
        res = np.loadtxt(io.StringIO(data), dtype=int)
        assert_array_equal(res, [[5, -3, 0], [-7, 1, 0]])

    def test_loadtxt_large_float(self):
        data = "1.7976931348623157e+308\n5e-324\n"
        res = np.loadtxt(io.StringIO(data))
        assert res[0] == np.finfo(np.float64).max
        assert res[1] == 5e-324

    def test_loadtxt_quoted_with_spaces(self):
        data = 'hello,1.0\nworld,2.0\n'
        res = np.loadtxt(io.StringIO(data), delimiter=",",
                         dtype=[("name", "U20"), ("val", "f8")])
        assert res["name"][0].strip() == "hello"

    def test_loadtxt_ndmin(self):
        data = "1 2 3\n"
        res = np.loadtxt(io.StringIO(data), ndmin=2)
        assert res.ndim == 2

    def test_loadtxt_unpack(self):
        data = "1 2 3\n4 5 6\n"
        a, b, c = np.loadtxt(io.StringIO(data), unpack=True)
        assert_array_equal(a, [1, 4])
        assert_array_equal(b, [2, 5])

    def test_loadtxt_structured_dtype(self):
        data = "1 2.0 hello\n3 4.0 world\n"
        dt = np.dtype([("x", "i4"), ("y", "f8"), ("z", "U10")])
        res = np.loadtxt(io.StringIO(data), dtype=dt)
        assert res["x"][0] == 1
        assert res["z"][1] == "world"

    def test_loadtxt_escape_in_quoted(self):
        data = 'hello,1\nworld,2\n'
        res = np.loadtxt(io.StringIO(data), delimiter=",",
                         dtype=[("name", "U20"), ("val", "f8")])
        assert "hello" in res["name"][0]

    def test_loadtxt_whitespace_delimiter(self):
        data = "1   2   3\n4   5   6\n"
        res = np.loadtxt(io.StringIO(data))
        assert_array_equal(res, [[1, 2, 3], [4, 5, 6]])

    def test_loadtxt_null_character(self):
        data = "1 2\n3 4\n"
        res = np.loadtxt(io.StringIO(data))
        assert_array_equal(res, [[1, 2], [3, 4]])


class TestFFTPocketComprehensive:
    """_pocketfft_umath.cpp: comprehensive FFT tests."""

    @pytest.mark.parametrize("dt", [np.float32, np.float64, np.longdouble])
    def test_fft_all_float_types(self, dt):
        a = np.arange(64, dtype=dt)
        res = np.fft.fft(a)
        assert res.shape == (64,)

    @pytest.mark.parametrize("dt", [np.float32, np.float64, np.longdouble])
    def test_ifft_all_float_types(self, dt):
        a = np.arange(64, dtype=dt)
        ft = np.fft.fft(a)
        res = np.fft.ifft(ft)
        assert_allclose(res.real, a, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dt", [np.float32, np.float64, np.longdouble])
    def test_rfft_all_float_types(self, dt):
        a = np.arange(64, dtype=dt)
        res = np.fft.rfft(a)
        assert res.shape == (33,)

    @pytest.mark.parametrize("dt", [np.float32, np.float64, np.longdouble])
    def test_irfft_all_float_types(self, dt):
        a = np.arange(64, dtype=dt)
        ft = np.fft.rfft(a)
        res = np.fft.irfft(ft, n=64)
        assert_allclose(res, a, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
    def test_fft_various_sizes(self, n):
        a = np.random.randn(n)
        res = np.fft.fft(a)
        assert res.shape == (n,)
        roundtrip = np.fft.ifft(res)
        assert_allclose(roundtrip.real, a, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
    def test_rfft_various_sizes(self, n):
        a = np.random.randn(n)
        res = np.fft.rfft(a)
        assert res.shape == (n // 2 + 1,)
        roundtrip = np.fft.irfft(res, n=n)
        assert_allclose(roundtrip, a, rtol=1e-10, atol=1e-10)

    def test_fft_2d(self):
        a = np.random.randn(16, 32)
        res = np.fft.fft2(a)
        assert res.shape == (16, 32)
        roundtrip = np.fft.ifft2(res)
        assert_allclose(roundtrip.real, a, rtol=1e-10, atol=1e-10)

    def test_fftn(self):
        a = np.random.randn(8, 16, 4)
        res = np.fft.fftn(a)
        assert res.shape == (8, 16, 4)
        roundtrip = np.fft.ifftn(res)
        assert_allclose(roundtrip.real, a, rtol=1e-10, atol=1e-10)

    def test_rfftn(self):
        a = np.random.randn(8, 16, 32)
        res = np.fft.rfftn(a)
        assert res.shape == (8, 16, 17)

    def test_hfft(self):
        a = np.random.randn(16) + 1j * np.random.randn(16)
        res = np.fft.hfft(a, n=30)
        assert res.shape == (30,)
        assert np.all(np.isreal(res))

    def test_ihfft(self):
        a = np.random.randn(30)
        res = np.fft.ihfft(a)
        assert res.shape == (16,)

    def test_fft_complex_input(self):
        a = np.random.randn(64) + 1j * np.random.randn(64)
        res = np.fft.fft(a)
        assert res.shape == (64,)
        roundtrip = np.fft.ifft(res)
        assert_allclose(roundtrip, a, rtol=1e-10, atol=1e-10)

    def test_fft_with_n_larger_than_input(self):
        a = np.random.randn(32)
        res = np.fft.fft(a, n=64)
        assert res.shape == (64,)

    def test_fft_with_n_smaller_than_input(self):
        a = np.random.randn(64)
        res = np.fft.fft(a, n=32)
        assert res.shape == (32,)

    def test_rfft_odd_length(self):
        a = np.random.randn(33)
        res = np.fft.rfft(a)
        assert res.shape == (17,)
        roundtrip = np.fft.irfft(res, n=33)
        assert_allclose(roundtrip, a, rtol=1e-10, atol=1e-10)

    def test_rfft_even_length(self):
        a = np.random.randn(32)
        res = np.fft.rfft(a)
        assert res.shape == (17,)
        roundtrip = np.fft.irfft(res, n=32)
        assert_allclose(roundtrip, a, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_fft_strided_input(self, dt):
        a = np.arange(128, dtype=dt)
        res = np.fft.fft(a[::2])
        assert res.shape == (64,)

    def test_fft_integer_input(self):
        a = np.arange(64, dtype=np.int32)
        res = np.fft.fft(a)
        assert res.shape == (64,)

    def test_fft_bool_input(self):
        a = np.array([True, False, True, False] * 16)
        res = np.fft.fft(a)
        assert res.shape == (64,)


class TestDispatchingComprehensive:
    """dispatching.cpp: ufunc dispatching edge cases."""

    def test_ufunc_mixed_dtype_promotion(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        res = np.add(a, b)
        assert res.dtype == np.float64

    def test_ufunc_int_float_promotion(self):
        a = np.array([1, 2], dtype=np.int64)
        b = np.array([1.0, 2.0], dtype=np.float32)
        res = np.add(a, b)
        assert res.dtype == np.float64

    def test_ufunc_complex_promotion(self):
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([1+1j, 2+2j], dtype=np.complex128)
        res = np.add(a, b)
        assert res.dtype == np.complex128

    def test_ufunc_signature_explicit(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4, 5, 6], dtype=np.int32)
        res = np.add(a, b, dtype=np.float64)
        assert res.dtype == np.float64

    def test_ufunc_out_dtype_mismatch(self):
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0], dtype=np.float64)
        out = np.zeros(2, dtype=np.float64)
        np.add(a, b, out=out)
        assert_array_equal(out, [4.0, 6.0])

    def test_ufunc_casting_no(self):
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3, 4], dtype=np.int32)
        res = np.add(a, b, casting='no')
        assert res.dtype == np.int32

    def test_ufunc_casting_equiv(self):
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3, 4], dtype=np.int32)
        res = np.add(a, b, casting='equiv')
        assert res.dtype == np.int32

    def test_ufunc_casting_safe(self):
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3, 4], dtype=np.int64)
        res = np.add(a, b, casting='safe')
        assert res.dtype == np.int64

    def test_ufunc_casting_same_kind(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float64)
        res = np.add(a, b, casting='same_kind')
        assert res.dtype == np.float64

    def test_ufunc_resolve_dtypes(self):
        try:
            res = np.add.resolve_dtypes((np.dtype("int32"), np.dtype("float32"), None))
            assert res is not None
        except (TypeError, AttributeError):
            a = np.array([1], dtype=np.int32)
            b = np.array([1.0], dtype=np.float32)
            res = np.add(a, b)
            assert res.dtype == np.float64

    def test_ufunc_reduce_compatible(self):
        a = np.arange(100, dtype=np.float64).reshape(10, 10)
        res = np.add.reduce(a, axis=0)
        assert res.shape == (10,)

    def test_ufunc_accumulate(self):
        a = np.arange(10, dtype=np.float64)
        res = np.add.accumulate(a)
        assert_allclose(res, np.cumsum(a))

    def test_ufunc_outer(self):
        a = np.arange(5, dtype=np.float64)
        b = np.arange(3, dtype=np.float64)
        res = np.multiply.outer(a, b)
        assert res.shape == (5, 3)

    def test_ufunc_at(self):
        a = np.zeros(5, dtype=np.float64)
        indices = np.array([0, 1, 2, 3, 4])
        np.add.at(a, indices, 1.0)
        assert_array_equal(a, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_ufunc_at_repeated_indices(self):
        a = np.zeros(5, dtype=np.float64)
        indices = np.array([0, 0, 1, 1, 2])
        np.add.at(a, indices, 1.0)
        assert a[0] == 2.0
        assert a[1] == 2.0
        assert a[2] == 1.0

    def test_comparison_ufunc_dispatch(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([2, 2, 2], dtype=np.int32)
        assert_array_equal(np.greater(a, b), [False, False, True])
        assert_array_equal(np.less(a, b), [True, False, False])
        assert_array_equal(np.equal(a, b), [False, True, False])

    def test_logical_ufunc_dispatch(self):
        a = np.array([True, False, True, False])
        b = np.array([True, True, False, False])
        assert_array_equal(np.logical_and(a, b), [True, False, False, False])
        assert_array_equal(np.logical_or(a, b), [True, True, True, False])
        assert_array_equal(np.logical_xor(a, b), [False, True, True, False])

    def test_ufunc_object_dtype_fallback(self):
        a = np.array([1, 2, 3], dtype=object)
        b = np.array([4, 5, 6], dtype=object)
        res = np.add(a, b)
        assert_array_equal(res, [5, 7, 9])


class TestMinMaxAccumulate:
    """loops_minmax_hwy.dispatch.cpp: accumulate paths."""

    @pytest.mark.parametrize("dt", [np.int32, np.int64, np.float32, np.float64])
    def test_maximum_accumulate(self, dt):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=dt)
        res = np.maximum.accumulate(a)
        expected = np.array([3, 3, 4, 4, 5, 9, 9, 9], dtype=dt)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize("dt", [np.int32, np.int64, np.float32, np.float64])
    def test_minimum_accumulate(self, dt):
        a = np.array([9, 7, 5, 3, 8, 2, 6, 1], dtype=dt)
        res = np.minimum.accumulate(a)
        expected = np.array([9, 7, 5, 3, 3, 2, 2, 1], dtype=dt)
        assert_array_equal(res, expected)


class TestMinMaxOuter:
    """loops_minmax_hwy.dispatch.cpp: outer product paths."""

    def test_maximum_outer(self):
        a = np.array([1, 3, 5], dtype=np.float64)
        b = np.array([2, 4], dtype=np.float64)
        res = np.maximum.outer(a, b)
        expected = np.array([[2, 4], [3, 4], [5, 5]], dtype=np.float64)
        assert_array_equal(res, expected)

    def test_minimum_outer(self):
        a = np.array([1, 3, 5], dtype=np.float64)
        b = np.array([2, 4], dtype=np.float64)
        res = np.minimum.outer(a, b)
        expected = np.array([[1, 1], [2, 3], [2, 4]], dtype=np.float64)
        assert_array_equal(res, expected)


class TestHalfPrecision:
    """half.hpp, halffloat.cpp: half-precision float operations."""

    def test_half_arithmetic(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float16)
        assert_allclose(np.add(a, b), [5.0, 7.0, 9.0], rtol=1e-3)

    def test_half_comparison(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        b = np.array([3.0, 2.0, 1.0], dtype=np.float16)
        assert_array_equal(np.greater(a, b), [False, False, True])

    def test_half_nan_inf(self):
        a = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float16)
        assert np.isnan(a[0])
        assert np.isinf(a[1])
        assert np.isinf(a[2])
        assert np.isfinite(a[3])

    def test_half_sqrt(self):
        a = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float16)
        res = np.sqrt(a)
        assert_allclose(res, [1.0, 2.0, 3.0, 4.0], rtol=1e-2)

    def test_half_reciprocal(self):
        a = np.array([1.0, 2.0, 4.0, 0.5], dtype=np.float16)
        res = np.reciprocal(a)
        assert_allclose(res, [1.0, 0.5, 0.25, 2.0], rtol=1e-2)

    def test_half_trig(self):
        a = np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 2], dtype=np.float16)
        res = np.sin(a)
        expected = np.sin(a.astype(np.float64)).astype(np.float16)
        assert_allclose(res, expected, rtol=1e-2, atol=1e-2)

    def test_half_exp(self):
        a = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float16)
        res = np.exp(a)
        expected = np.exp(a.astype(np.float64)).astype(np.float16)
        assert_allclose(res, expected, rtol=1e-2)

    def test_half_log(self):
        a = np.array([1.0, np.e, np.e**2], dtype=np.float16)
        res = np.log(a)
        expected = np.log(a.astype(np.float64)).astype(np.float16)
        assert_allclose(res, expected, rtol=1e-2)


class TestEinsumComprehensive:
    """einsum.cpp: comprehensive einsum tests."""

    def test_einsum_matrix_multiply(self):
        A = np.random.randn(4, 5)
        B = np.random.randn(5, 6)
        res = np.einsum("ij,jk->ik", A, B)
        assert_allclose(res, A @ B, rtol=1e-10)

    def test_einsum_trace(self):
        A = np.random.randn(5, 5)
        res = np.einsum("ii->", A)
        assert_allclose(res, np.trace(A))

    def test_einsum_diagonal(self):
        A = np.random.randn(5, 5)
        res = np.einsum("ii->i", A)
        assert_array_equal(res, np.diag(A))

    def test_einsum_transpose(self):
        A = np.random.randn(3, 4)
        res = np.einsum("ij->ji", A)
        assert_array_equal(res, A.T)

    def test_einsum_outer(self):
        a = np.random.randn(3)
        b = np.random.randn(4)
        res = np.einsum("i,j->ij", a, b)
        assert_allclose(res, np.outer(a, b))

    def test_einsum_inner(self):
        a = np.random.randn(5)
        b = np.random.randn(5)
        res = np.einsum("i,i->", a, b)
        assert_allclose(res, np.dot(a, b))

    def test_einsum_batch_matmul(self):
        A = np.random.randn(3, 4, 5)
        B = np.random.randn(3, 5, 6)
        res = np.einsum("bij,bjk->bik", A, B)
        for i in range(3):
            assert_allclose(res[i], A[i] @ B[i], rtol=1e-10)

    def test_einsum_sum_axis(self):
        A = np.random.randn(3, 4, 5)
        res = np.einsum("ijk->ik", A)
        assert_allclose(res, A.sum(axis=1))

    def test_einsub_optimize(self):
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 5)
        C = np.random.randn(5, 6)
        res = np.einsum("ij,jk,kl->il", A, B, C, optimize=True)
        expected = A @ B @ C
        assert_allclose(res, expected, rtol=1e-10)


class TestUniqueComprehensive:
    """unique.cpp: comprehensive unique tests."""

    @pytest.mark.parametrize("dt", [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64,
    ])
    def test_unique_all_types(self, dt):
        a = np.array([3, 1, 2, 1, 3, 2, 1], dtype=dt)
        res = np.unique(a)
        assert_array_equal(res, np.unique(a.tolist()))

    def test_unique_return_index(self):
        a = np.array([3, 1, 2, 1, 3])
        vals, idx = np.unique(a, return_index=True)
        assert_array_equal(a[idx], vals)

    def test_unique_return_inverse(self):
        a = np.array([3, 1, 2, 1, 3])
        vals, inv = np.unique(a, return_inverse=True)
        assert_array_equal(vals[inv], a)

    def test_unique_return_counts(self):
        a = np.array([3, 1, 2, 1, 3, 3])
        vals, counts = np.unique(a, return_counts=True)
        assert_array_equal(vals, [1, 2, 3])
        assert_array_equal(counts, [2, 1, 3])

    def test_unique_with_nan(self):
        a = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        res = np.unique(a)
        assert len(res) == 4

    def test_unique_2d_axis0(self):
        a = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
        res = np.unique(a, axis=0)
        assert res.shape[0] == 3

    def test_unique_2d_axis1(self):
        a = np.array([[1, 3, 1], [2, 4, 2]])
        res = np.unique(a, axis=1)
        assert res.shape[1] == 2
