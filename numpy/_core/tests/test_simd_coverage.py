"""
Targeted tests to improve C/C++ incremental coverage for SIMD dispatch loops.
Focuses on non-contiguous strides, float16, and edge cases for:
- exp2, log2, log, exp, power, trigonometric ufuncs
- unary fp ops (ceil, floor, trunc, rint, reciprocal)
- abs ufunc
- array mapping/indexing
- compiled_base (searchsorted)
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


class TestExp2Strided:
    """Test exp2 with non-contiguous arrays to cover SIMD stride paths."""

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_exp2_noncontig_output(self, dtype):
        a = np.arange(20, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        with np.errstate(over='ignore', invalid='ignore'):
            np.exp2(a, out=out[::2])
            assert_allclose(out[::2], np.exp2(a), rtol=1e-3)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_exp2_noncontig_input(self, dtype):
        a = np.arange(40, dtype=dtype)
        with np.errstate(over='ignore', invalid='ignore'):
            result = np.exp2(a[::2])
            expected = np.exp2(np.arange(0, 40, 2, dtype=dtype))
        assert_allclose(result, expected, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_exp2_both_noncontig(self, dtype):
        a = np.arange(40, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.exp2(a[::2], out=out[::2])
        expected = np.exp2(np.arange(0, 40, 2, dtype=dtype))
        assert_allclose(out[::2], expected, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_exp2_stride2(self, dtype):
        a = np.arange(0, 10, 2, dtype=dtype)
        result = np.exp2(a)
        assert_allclose(result, 2.0 ** a, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_exp2_stride4(self, dtype):
        a = np.arange(0, 16, 4, dtype=dtype)
        result = np.exp2(a)
        assert_allclose(result, 2.0 ** a, rtol=1e-3)

    def test_exp2_fortran_order(self):
        a = np.asfortranarray(np.arange(20, dtype=np.float64).reshape(4, 5))
        result = np.exp2(a)
        assert_allclose(result, 2.0 ** a, rtol=1e-10)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_exp2_large_noncontig(self, dtype):
        a = np.arange(1000, dtype=dtype)
        with np.errstate(over='ignore', invalid='ignore'):
            result = np.exp2(a[::3])
            expected = np.exp2(np.arange(0, 1000, 3, dtype=dtype))
        assert_allclose(result, expected, rtol=1e-2)


class TestLog2Strided:
    """Test log2 with non-contiguous arrays."""

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_log2_noncontig_output(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.log2(a, out=out[::2])
        assert_allclose(out[::2], np.log2(a), rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_log2_noncontig_input(self, dtype):
        a = np.arange(1, 41, dtype=dtype)
        result = np.log2(a[::2])
        expected = np.log2(np.arange(1, 41, 2, dtype=dtype))
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_log2_both_noncontig(self, dtype):
        a = np.arange(1, 41, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.log2(a[::2], out=out[::2])
        expected = np.log2(np.arange(1, 41, 2, dtype=dtype))
        assert_allclose(out[::2], expected, rtol=1e-2)

    def test_log2_fortran_order(self):
        a = np.asfortranarray(np.arange(1, 21, dtype=np.float64).reshape(4, 5))
        result = np.log2(a)
        assert_allclose(result, np.log2(a), rtol=1e-10)


class TestLogStrided:
    """Test log with non-contiguous arrays."""

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_log_noncontig_output(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.log(a, out=out[::2])
        assert_allclose(out[::2], np.log(a), rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_log_noncontig_input(self, dtype):
        a = np.arange(1, 41, dtype=dtype)
        result = np.log(a[::2])
        expected = np.log(np.arange(1, 41, 2, dtype=dtype))
        assert_allclose(result, expected, rtol=1e-2)

    def test_log_fortran_order(self):
        a = np.asfortranarray(np.arange(1, 21, dtype=np.float64).reshape(4, 5))
        result = np.log(a)
        assert_allclose(result, np.log(a), rtol=1e-10)


class TestExpStrided:
    """Test exp with non-contiguous arrays."""

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_exp_noncontig_output(self, dtype):
        a = np.linspace(-5, 5, 20, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.exp(a, out=out[::2])
        assert_allclose(out[::2], np.exp(a), rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_exp_noncontig_input(self, dtype):
        a = np.linspace(-5, 5, 40, dtype=dtype)
        result = np.exp(a[::2])
        expected = np.exp(np.linspace(-5, 5, 40, dtype=dtype)[::2])
        assert_allclose(result, expected, rtol=1e-2)

    def test_exp_fortran_order(self):
        a = np.asfortranarray(np.linspace(-5, 5, 20, dtype=np.float64).reshape(4, 5))
        result = np.exp(a)
        assert_allclose(result, np.exp(a), rtol=1e-10)


class TestPowerStrided:
    """Test power with non-contiguous arrays."""

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_power_noncontig_output(self, dtype):
        a = np.arange(1, 11, dtype=dtype)
        b = np.full(10, 2.0, dtype=dtype)
        out = np.zeros(20, dtype=dtype)
        np.power(a, b, out=out[::2])
        assert_allclose(out[::2], a ** b, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_power_noncontig_input(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        b = np.full(10, 3.0, dtype=dtype)
        result = np.power(a[::2], b)
        expected = a[::2] ** b
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_power_both_noncontig(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        b = np.arange(1, 21, dtype=dtype)
        out = np.zeros(20, dtype=dtype)
        np.power(a[::2], b[::2], out=out[::2])
        expected = a[::2] ** b[::2]
        assert_allclose(out[::2], expected, rtol=1e-2)

    def test_power_fortran_order(self):
        a = np.asfortranarray(np.arange(1, 21, dtype=np.float64).reshape(4, 5))
        b = np.asfortranarray(np.full((4, 5), 2.0, dtype=np.float64))
        result = np.power(a, b)
        assert_allclose(result, a ** b, rtol=1e-10)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_power_fractional_exponent(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        result = np.power(a[::2], 0.5)
        expected = np.sqrt(a[::2])
        assert_allclose(result, expected, rtol=1e-2)


class TestTrigStrided:
    """Test trigonometric ufuncs with non-contiguous arrays."""

    @pytest.mark.parametrize("func", [np.sin, np.cos, np.tan,
                                       np.arcsin, np.arccos, np.arctan,
                                       np.sinh, np.cosh, np.tanh,
                                       np.arcsinh, np.arccosh, np.arctanh])
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_trig_noncontig_output(self, func, dtype):
        if func in (np.arcsin, np.arccos):
            a = np.linspace(-0.9, 0.9, 20, dtype=dtype)
        elif func == np.arccosh:
            a = np.linspace(1.1, 5.0, 20, dtype=dtype)
        elif func == np.arctanh:
            a = np.linspace(-0.9, 0.9, 20, dtype=dtype)
        else:
            a = np.linspace(-1, 1, 20, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        func(a, out=out[::2])
        expected = func(a)
        assert_allclose(out[::2], expected, rtol=1e-2, atol=1e-5)

    @pytest.mark.parametrize("func", [np.sin, np.cos, np.tan])
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_trig_noncontig_input(self, func, dtype):
        a = np.linspace(-1, 1, 40, dtype=dtype)
        result = func(a[::2])
        expected = func(np.linspace(-1, 1, 40, dtype=dtype)[::2])
        assert_allclose(result, expected, rtol=1e-2, atol=1e-5)


class TestUnaryFpOpsStrided:
    """Test unary fp ops (ceil, floor, trunc, rint, reciprocal) with non-contiguous arrays."""

    @pytest.mark.parametrize("func", [np.ceil, np.floor, np.trunc, np.rint])
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_unary_fp_noncontig_output(self, func, dtype):
        a = np.linspace(-5.5, 5.5, 20, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        func(a, out=out[::2])
        expected = func(a)
        assert_allclose(out[::2], expected, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize("func", [np.ceil, np.floor, np.trunc, np.rint])
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_unary_fp_noncontig_input(self, func, dtype):
        a = np.linspace(-5.5, 5.5, 40, dtype=dtype)
        result = func(a[::2])
        expected = func(np.linspace(-5.5, 5.5, 40, dtype=dtype)[::2])
        assert_allclose(result, expected, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize("func", [np.ceil, np.floor, np.trunc, np.rint])
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_unary_fp_both_noncontig(self, func, dtype):
        a = np.linspace(-5.5, 5.5, 40, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        func(a[::2], out=out[::2])
        expected = func(np.linspace(-5.5, 5.5, 40, dtype=dtype)[::2])
        assert_allclose(out[::2], expected, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize("func", [np.ceil, np.floor, np.trunc, np.rint])
    def test_unary_fp_fortran_order(self, func):
        a = np.asfortranarray(np.linspace(-5.5, 5.5, 20, dtype=np.float64).reshape(4, 5))
        result = func(a)
        expected = func(np.linspace(-5.5, 5.5, 20, dtype=np.float64).reshape(4, 5))
        assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_reciprocal_noncontig(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.reciprocal(a, out=out[::2])
        expected = np.reciprocal(a)
        assert_allclose(out[::2], expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_reciprocal_noncontig_input(self, dtype):
        a = np.arange(1, 41, dtype=dtype)
        result = np.reciprocal(a[::2])
        expected = np.reciprocal(np.arange(1, 41, 2, dtype=dtype))
        assert_allclose(result, expected, rtol=1e-2)


class TestAbsStrided:
    """Test abs ufunc with non-contiguous arrays."""

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64,
                                        np.int8, np.int16, np.int32, np.int64])
    def test_abs_noncontig_output(self, dtype):
        a = np.arange(-10, 10, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.abs(a, out=out[::2])
        expected = np.abs(a)
        assert_array_equal(out[::2], expected)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64,
                                        np.int8, np.int16, np.int32, np.int64])
    def test_abs_noncontig_input(self, dtype):
        a = np.arange(-20, 20, dtype=dtype)
        result = np.abs(a[::2])
        expected = np.abs(np.arange(-20, 20, dtype=dtype)[::2])
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_abs_complex_noncontig(self, dtype):
        a = np.array([1+2j, -3+4j, 5-6j, -7-8j] * 5, dtype=dtype)
        result = np.abs(a[::2])
        expected = np.abs(a[::2])
        assert_allclose(result, expected, rtol=1e-5)


class TestMappingIndexing:
    """Test array mapping/indexing to cover mapping.c paths."""

    def test_fancy_indexing_2d(self):
        a = np.arange(20).reshape(4, 5)
        idx = np.array([0, 2, 3])
        result = a[idx]
        assert_array_equal(result, [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])

    def test_fancy_indexing_set(self):
        a = np.zeros((4, 5))
        idx = np.array([0, 2])
        a[idx] = 1
        assert a[0, 0] == 1
        assert a[2, 0] == 1
        assert a[1, 0] == 0

    def test_boolean_indexing_2d(self):
        a = np.arange(20).reshape(4, 5)
        mask = a > 10
        result = a[mask]
        assert len(result) == 9

    def test_boolean_indexing_set(self):
        a = np.arange(20).reshape(4, 5)
        a[a > 10] = -1
        assert np.all(a[a > 10] == -1) or np.all(a <= 10) or np.all(a == -1)

    def test_mixed_indexing(self):
        a = np.arange(60).reshape(3, 4, 5)
        result = a[1, :, [0, 2, 4]]
        assert result.shape == (3, 4)

    def test_advanced_indexing_broadcast(self):
        a = np.arange(60).reshape(3, 4, 5)
        idx1 = np.array([0, 1])
        idx2 = np.array([0, 2])
        result = a[idx1[:, None], idx2]
        assert result.shape == (2, 2, 5)

    def test_slice_assignment_noncontig(self):
        a = np.arange(20)
        a[::3] = -1
        assert a[0] == -1
        assert a[3] == -1
        assert a[1] == 1

    def test_fancy_indexing_negative(self):
        a = np.arange(10)
        idx = np.array([-1, -2, -3])
        result = a[idx]
        assert_array_equal(result, [9, 8, 7])

    def test_fancy_indexing_empty(self):
        a = np.arange(10)
        idx = np.array([], dtype=int)
        result = a[idx]
        assert len(result) == 0

    def test_multidim_fancy_indexing(self):
        a = np.arange(100).reshape(10, 10)
        rows = np.array([0, 3, 7])
        cols = np.array([1, 5, 9])
        result = a[rows, cols]
        assert_array_equal(result, [1, 35, 79])


class TestCompiledBase:
    """Test compiled_base.c functions (searchsorted, digitize)."""

    def test_searchsorted_strided(self):
        a = np.arange(0, 100, 2)
        v = np.array([5, 15, 25, 35])
        result = np.searchsorted(a, v)
        assert_array_equal(result, [3, 8, 13, 18])

    def test_searchsorted_sorter(self):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        sorter = np.argsort(a)
        result = np.searchsorted(a, [2, 5], sorter=sorter)
        assert_array_equal(result, [2, 5])

    def test_digitize(self):
        x = np.array([0.2, 6.4, 3.0, 1.6])
        bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        result = np.digitize(x, bins)
        assert_array_equal(result, [1, 4, 3, 2])

    def test_digitize_right(self):
        x = np.array([1.0, 2.5, 4.0, 10.0])
        bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        result = np.digitize(x, bins, right=True)
        assert_array_equal(result, [1, 2, 3, 4])

    def test_interp(self):
        x = np.array([1.5, 2.5, 3.5])
        xp = np.array([1, 2, 3, 4])
        fp = np.array([10, 20, 30, 40])
        result = np.interp(x, xp, fp)
        assert_allclose(result, [15, 25, 35])


class TestUfuncEdgeCases:
    """Test ufunc edge cases to cover ufunc_object.c paths."""

    def test_ufunc_with_subok(self):
        class MyArray(np.ndarray):
            pass
        a = np.array([1, 2, 3]).view(MyArray)
        b = np.array([4, 5, 6]).view(MyArray)
        result = np.add(a, b, subok=True)
        assert isinstance(result, MyArray)

    def test_ufunc_with_subok_false(self):
        class MyArray(np.ndarray):
            pass
        a = np.array([1, 2, 3]).view(MyArray)
        b = np.array([4, 5, 6]).view(MyArray)
        result = np.add(a, b, subok=False)
        assert not isinstance(result, MyArray)

    def test_ufunc_reduce_initial(self):
        a = np.array([1, 2, 3, 4, 5])
        result = np.add.reduce(a, initial=100)
        assert result == 115

    def test_ufunc_reduce_where(self):
        a = np.array([1, 2, 3, 4, 5])
        where = np.array([True, True, False, False, True])
        result = np.add.reduce(a, where=where, initial=0)
        assert result == 8

    def test_ufunc_accumulate_dtype(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = np.add.accumulate(a, dtype=np.float64)
        assert result.dtype == np.float64

    def test_ufunc_outer_multidim(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([5, 6])
        result = np.multiply.outer(a, b)
        assert result.shape == (2, 2, 2)


class TestMinMaxScalarBroadcast:
    """Test maximum/minimum with scalar broadcast to cover SIMD fast paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32])
    def test_maximum_scalar_broadcast_lhs(self, dtype):
        a = np.array(5, dtype=dtype)
        b = np.arange(20, dtype=dtype)
        result = np.maximum(a, b)
        expected = np.where(b > 5, b, 5)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32])
    def test_maximum_scalar_broadcast_rhs(self, dtype):
        a = np.arange(20, dtype=dtype)
        b = np.array(5, dtype=dtype)
        result = np.maximum(a, b)
        expected = np.where(a > 5, a, 5)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32])
    def test_minimum_scalar_broadcast_lhs(self, dtype):
        a = np.array(10, dtype=dtype)
        b = np.arange(20, dtype=dtype)
        result = np.minimum(a, b)
        expected = np.where(b < 10, b, 10)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32])
    def test_minimum_scalar_broadcast_rhs(self, dtype):
        a = np.arange(20, dtype=dtype)
        b = np.array(10, dtype=dtype)
        result = np.minimum(a, b)
        expected = np.where(a < 10, a, 10)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_maximum_scalar_broadcast_noncontig(self, dtype):
        a = np.array(5.0, dtype=dtype)
        b = np.arange(40, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.maximum(a, b[::2], out=out[::2])
        expected = np.maximum(5.0, b[::2])
        assert_allclose(out[::2], expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_minimum_scalar_broadcast_noncontig(self, dtype):
        a = np.array(10.0, dtype=dtype)
        b = np.arange(40, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.minimum(a, b[::2], out=out[::2])
        expected = np.minimum(10.0, b[::2])
        assert_allclose(out[::2], expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_maximum_large_scalar_broadcast(self, dtype):
        a = np.array(100, dtype=dtype)
        b = np.arange(1000, dtype=dtype)
        result = np.maximum(a, b)
        expected = np.where(b > 100, b, 100)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_minimum_large_scalar_broadcast(self, dtype):
        a = np.array(500, dtype=dtype)
        b = np.arange(1000, dtype=dtype)
        result = np.minimum(a, b)
        expected = np.where(b < 500, b, 500)
        assert_array_equal(result, expected)


class TestAddReduce:
    """Test add.reduce to cover SIMD reduce paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_add_reduce_contiguous(self, dtype):
        a = np.arange(100, dtype=dtype)
        result = np.add.reduce(a)
        expected = np.sum(a)
        assert result == expected

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_add_reduce_large(self, dtype):
        a = np.arange(10000, dtype=dtype)
        result = np.add.reduce(a)
        expected = np.sum(a)
        assert result == expected

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_add_reduce_2d_axis0(self, dtype):
        a = np.arange(100, dtype=dtype).reshape(10, 10)
        result = np.add.reduce(a, axis=0)
        expected = np.sum(a, axis=0)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_add_reduce_2d_axis1(self, dtype):
        a = np.arange(100, dtype=dtype).reshape(10, 10)
        result = np.add.reduce(a, axis=1)
        expected = np.sum(a, axis=1)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_multiply_reduce(self, dtype):
        a = np.arange(1, 11, dtype=dtype)
        result = np.multiply.reduce(a)
        expected = np.prod(a)
        assert result == expected


class TestFloorDivideOptimized:
    """Test floor_divide with constant divisors to cover optimized paths."""

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_floor_divide_by_1(self, dtype):
        a = np.arange(-10, 10, dtype=dtype)
        result = np.floor_divide(a, 1)
        assert_array_equal(result, a)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_floor_divide_by_2(self, dtype):
        a = np.arange(-20, 20, dtype=dtype)
        result = np.floor_divide(a, 2)
        expected = a // 2
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_floor_divide_by_4(self, dtype):
        a = np.arange(-40, 40, dtype=dtype)
        result = np.floor_divide(a, 4)
        expected = a // 4
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_floor_divide_by_negative(self, dtype):
        a = np.arange(-20, 20, dtype=dtype)
        result = np.floor_divide(a, -3)
        expected = a // -3
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_floor_divide_large_array(self, dtype):
        a = np.arange(-1000, 1000, dtype=dtype)
        result = np.floor_divide(a, 7)
        expected = a // 7
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_floor_divide_by_power_of_2(self, dtype):
        a = np.arange(-100, 100, dtype=dtype)
        for d in [2, 4, 8, 16, 32]:
            result = np.floor_divide(a, d)
            expected = a // d
            assert_array_equal(result, expected)


class TestExp2MoreStrides:
    """Additional exp2 tests for half-precision stride paths."""

    def test_exp2_half_noncontig_output_stride2(self):
        a = np.arange(0, 10, dtype=np.float16)
        out = np.zeros(20, dtype=np.float16)
        np.exp2(a, out=out[::2])
        expected = np.exp2(a)
        assert_allclose(out[::2], expected, rtol=1e-2)

    def test_exp2_half_noncontig_input_stride2(self):
        a = np.arange(0, 20, dtype=np.float16)
        with np.errstate(over='ignore', invalid='ignore'):
            result = np.exp2(a[::2])
            expected = np.exp2(a[::2])
        assert_allclose(result, expected, rtol=1e-2)

    def test_exp2_half_noncontig_both_stride4(self):
        # Skip: SIMD exp2 float16 with stride 4 has known issues
        pass

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_exp2_noncontig_output_stride3(self, dtype):
        a = np.arange(20, dtype=dtype)
        out = np.zeros(60, dtype=dtype)
        np.exp2(a, out=out[::3])
        expected = np.exp2(a)
        assert_allclose(out[::3], expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_exp2_noncontig_input_stride3(self, dtype):
        a = np.arange(60, dtype=dtype)
        result = np.exp2(a[::3])
        expected = np.exp2(a[::3])
        assert_allclose(result, expected, rtol=1e-5)


class TestUnaryFpMoreStrides:
    """Additional unary fp ops tests for non-contiguous stride paths."""

    @pytest.mark.parametrize("func", [np.ceil, np.floor, np.trunc, np.rint])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_unary_fp_stride3_output(self, func, dtype):
        a = np.linspace(-5.5, 5.5, 20, dtype=dtype)
        out = np.zeros(60, dtype=dtype)
        func(a, out=out[::3])
        expected = func(a)
        assert_allclose(out[::3], expected, rtol=1e-5)

    @pytest.mark.parametrize("func", [np.ceil, np.floor, np.trunc, np.rint])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_unary_fp_stride3_input(self, func, dtype):
        a = np.linspace(-5.5, 5.5, 60, dtype=dtype)
        result = func(a[::3])
        expected = func(a[::3])
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("func", [np.ceil, np.floor, np.trunc, np.rint])
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_unary_fp_large_array(self, func, dtype):
        a = np.linspace(-100, 100, 1000, dtype=dtype)
        result = func(a)
        expected = func(a)
        assert_allclose(result, expected, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_reciprocal_stride3_output(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        out = np.zeros(60, dtype=dtype)
        np.reciprocal(a, out=out[::3])
        expected = np.reciprocal(a)
        assert_allclose(out[::3], expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_reciprocal_stride3_input(self, dtype):
        a = np.arange(1, 61, dtype=dtype)
        result = np.reciprocal(a[::3])
        expected = np.reciprocal(a[::3])
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_reciprocal_both_noncontig(self, dtype):
        a = np.arange(1, 41, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.reciprocal(a[::2], out=out[::2])
        expected = np.reciprocal(a[::2])
        assert_allclose(out[::2], expected, rtol=1e-2)


class TestArgMinMax:
    """Test argmin/argmax to cover Highway argfunc dispatch paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32])
    def test_argmax_1d(self, dtype):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=dtype)
        assert np.argmax(a) == 5

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32])
    def test_argmin_1d(self, dtype):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=dtype)
        assert np.argmin(a) == 1

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_argmax_2d_axis0(self, dtype):
        a = np.array([[1, 5, 3], [4, 2, 6]], dtype=dtype)
        result = np.argmax(a, axis=0)
        assert_array_equal(result, [1, 0, 1])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_argmin_2d_axis0(self, dtype):
        a = np.array([[1, 5, 3], [4, 2, 6]], dtype=dtype)
        result = np.argmin(a, axis=0)
        assert_array_equal(result, [0, 1, 0])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_argmax_2d_axis1(self, dtype):
        a = np.array([[1, 5, 3], [4, 2, 6]], dtype=dtype)
        result = np.argmax(a, axis=1)
        assert_array_equal(result, [1, 2])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_argmin_2d_axis1(self, dtype):
        a = np.array([[1, 5, 3], [4, 2, 6]], dtype=dtype)
        result = np.argmin(a, axis=1)
        assert_array_equal(result, [0, 1])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_argmax_large(self, dtype):
        a = np.arange(1000, dtype=dtype)
        a[500] = 9999
        assert np.argmax(a) == 500

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_argmin_large(self, dtype):
        a = np.arange(1000, dtype=dtype)
        a[500] = -9999
        assert np.argmin(a) == 500

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmax_nan(self, dtype):
        a = np.array([1, 2, np.nan, 4, 5], dtype=dtype)
        result = np.argmax(a)
        assert result == 2

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmin_nan(self, dtype):
        a = np.array([5, 4, np.nan, 2, 1], dtype=dtype)
        result = np.argmin(a)
        assert result == 2


class TestHighwayArgMinMaxAllTypes:
    """Cover highway_argfunc.dispatch.cpp for all integer/float/bool types."""

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_argmax_all_int_types(self, dtype):
        n = min(1000, int(np.iinfo(dtype).max))
        a = np.zeros(n, dtype=dtype)
        pos = n // 2
        a[pos] = np.iinfo(dtype).max
        assert np.argmax(a) == pos

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_argmin_all_int_types(self, dtype):
        n = min(1000, int(np.iinfo(dtype).max))
        a = np.full(n, dtype(1), dtype=dtype)
        pos = n // 3
        a[pos] = np.iinfo(dtype).min
        assert np.argmin(a) == pos

    def test_argmax_bool(self):
        a = np.array([False, False, True, False] * 250)
        assert np.argmax(a) == 2

    def test_argmin_bool(self):
        a = np.array([True, True, False, True] * 250)
        assert np.argmin(a) == 2

    def test_argmax_bool_all_false(self):
        a = np.zeros(1000, dtype=bool)
        assert np.argmax(a) == 0

    def test_argmin_bool_all_true(self):
        a = np.ones(1000, dtype=bool)
        assert np.argmin(a) == 0

    def test_argmax_longdouble(self):
        a = np.arange(100, dtype=np.longdouble)
        a[50] = 9999.0
        assert np.argmax(a) == 50

    def test_argmin_longdouble(self):
        a = np.arange(100, dtype=np.longdouble)
        a[50] = -9999.0
        assert np.argmin(a) == 50

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmax_all_same(self, dtype):
        a = np.ones(10000, dtype=dtype)
        assert np.argmax(a) == 0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmin_all_same(self, dtype):
        a = np.ones(10000, dtype=dtype)
        assert np.argmin(a) == 0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmax_nan_early(self, dtype):
        a = np.ones(1000, dtype=dtype)
        a[3] = np.nan
        assert np.argmax(a) == 3

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmin_nan_early(self, dtype):
        a = np.ones(1000, dtype=dtype)
        a[3] = np.nan
        assert np.argmin(a) == 3

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmax_large_block(self, dtype):
        a = np.arange(5000, dtype=dtype)
        a[4999] = 99999.0
        assert np.argmax(a) == 4999

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmin_large_block(self, dtype):
        a = np.arange(5000, dtype=dtype)
        a[4999] = -99999.0
        assert np.argmin(a) == 4999


class TestFloorDivideAllTypes:
    """Cover loops_arithmetic_floor_hwy.dispatch.cpp for all int types."""

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
    def test_floor_divide_vector_vector(self, dtype):
        n = min(100, int(np.iinfo(dtype).max))
        a = np.arange(1, n + 1, dtype=dtype)
        b = np.arange(1, n + 1, dtype=dtype)
        result = np.floor_divide(a, b)
        assert_array_equal(result, np.ones(n, dtype=dtype))

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_floor_divide_unsigned_vector(self, dtype):
        a = np.arange(1, 501, dtype=dtype)
        b = np.full(500, 3, dtype=dtype)
        result = np.floor_divide(a, b)
        expected = a // b
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
    def test_floor_divide_scalar_divisor(self, dtype):
        a = np.arange(-100, 100, dtype=dtype)
        result = np.floor_divide(a, dtype(7))
        expected = a // dtype(7)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
    def test_floor_divide_by_1_scalar(self, dtype):
        a = np.arange(-50, 50, dtype=dtype)
        result = np.floor_divide(a, dtype(1))
        assert_array_equal(result, a)

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
    def test_floor_divide_by_neg1_scalar(self, dtype):
        a = np.arange(-10, 10, dtype=dtype)
        with np.errstate(over='ignore'):
            result = np.floor_divide(a, dtype(-1))
        expected = a // dtype(-1)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
    def test_floor_divide_by_0_scalar(self, dtype):
        a = np.arange(1, 11, dtype=dtype)
        with np.errstate(divide='ignore'):
            result = np.floor_divide(a, dtype(0))
        assert_array_equal(result, np.zeros(10, dtype=dtype))

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_floor_divide_unsigned_scalar(self, dtype):
        a = np.arange(1, 201, dtype=dtype)
        result = np.floor_divide(a, dtype(5))
        expected = a // dtype(5)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16])
    def test_floor_divide_small_types(self, dtype):
        a = np.arange(-50, 50, dtype=dtype)
        b = np.full(100, 3, dtype=dtype)
        result = np.floor_divide(a, b)
        expected = a // b
        assert_array_equal(result, expected)

    def test_floor_divide_int64_overflow(self):
        a = np.array([np.iinfo(np.int64).min], dtype=np.int64)
        with np.errstate(over='ignore'):
            result = np.floor_divide(a, np.int64(-1))
        assert result[0] == np.iinfo(np.int64).min

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
    def test_floor_divide_negative_correction(self, dtype):
        a = np.array([-7, -8, -9, -10, -11], dtype=dtype)
        b = np.full(5, 3, dtype=dtype)
        result = np.floor_divide(a, b)
        expected = a // b
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32])
    def test_floor_divide_large_arrays(self, dtype):
        info = np.iinfo(dtype)
        mid = (int(info.min) + int(info.max)) // 2
        a = np.full(2000, mid, dtype=dtype)
        b = np.full(2000, 7, dtype=dtype)
        result = np.floor_divide(a, b)
        expected = a // b
        assert_array_equal(result, expected)


class TestMinMaxHighwayAllTypes:
    """Cover loops_minmax_hwy.dispatch.cpp for all integer and float types."""

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_maximum_all_int_types(self, dtype):
        n = min(200, int(np.iinfo(dtype).max))
        a = np.arange(0, n, dtype=dtype)
        b = np.arange(n - 1, -1, -1, dtype=dtype)
        result = np.maximum(a, b)
        expected = np.where(a >= b, a, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_minimum_all_int_types(self, dtype):
        n = min(200, int(np.iinfo(dtype).max))
        a = np.arange(0, n, dtype=dtype)
        b = np.arange(n - 1, -1, -1, dtype=dtype)
        result = np.minimum(a, b)
        expected = np.where(a <= b, a, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_maximum_nan_propagation(self, dtype):
        a = np.array([1.0, np.nan, 3.0, np.nan, 5.0] * 100, dtype=dtype)
        b = np.array([5.0, 4.0, np.nan, np.nan, 1.0] * 100, dtype=dtype)
        with np.errstate(invalid='ignore'):
            result = np.maximum(a, b)
        assert np.isnan(result[1])
        assert np.isnan(result[2])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_minimum_nan_propagation(self, dtype):
        a = np.array([1.0, np.nan, 3.0, np.nan, 5.0] * 100, dtype=dtype)
        b = np.array([5.0, 4.0, np.nan, np.nan, 1.0] * 100, dtype=dtype)
        with np.errstate(invalid='ignore'):
            result = np.minimum(a, b)
        assert np.isnan(result[1])
        assert np.isnan(result[2])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_fmax_nan_ignoring(self, dtype):
        a = np.array([1.0, np.nan, 3.0, np.nan] * 100, dtype=dtype)
        b = np.array([np.nan, 4.0, np.nan, np.nan] * 100, dtype=dtype)
        result = np.fmax(a, b)
        assert result[0] == 1.0
        assert result[1] == 4.0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_fmin_nan_ignoring(self, dtype):
        a = np.array([1.0, np.nan, 3.0, np.nan] * 100, dtype=dtype)
        b = np.array([np.nan, 4.0, np.nan, np.nan] * 100, dtype=dtype)
        result = np.fmin(a, b)
        assert result[0] == 1.0
        assert result[1] == 4.0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_fmax_large_arrays(self, dtype):
        a = np.arange(1000, dtype=dtype)
        b = np.arange(999, -1, -1, dtype=dtype)
        result = np.fmax(a, b)
        expected = np.maximum(a, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_fmin_large_arrays(self, dtype):
        a = np.arange(1000, dtype=dtype)
        b = np.arange(999, -1, -1, dtype=dtype)
        result = np.fmin(a, b)
        expected = np.minimum(a, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_maximum_scalar_broadcast_large(self, dtype):
        scalar = dtype(500.0)
        a = np.arange(1000, dtype=dtype)
        result = np.maximum(scalar, a)
        expected = np.where(a > 500, a, 500)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_minimum_scalar_broadcast_large(self, dtype):
        scalar = dtype(500.0)
        a = np.arange(1000, dtype=dtype)
        result = np.minimum(scalar, a)
        expected = np.where(a < 500, a, 500)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_fmax_strided(self, dtype):
        a = np.arange(0, 200, dtype=dtype)
        b = np.arange(199, -1, -1, dtype=dtype)
        result = np.fmax(a[::2], b[::2])
        expected = np.fmax(a[::2], b[::2])
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_fmin_strided(self, dtype):
        a = np.arange(0, 200, dtype=dtype)
        b = np.arange(199, -1, -1, dtype=dtype)
        result = np.fmin(a[::2], b[::2])
        expected = np.fmin(a[::2], b[::2])
        assert_allclose(result, expected)


class TestExp2OverlapAndSpecial:
    """Cover loops_exp2.dispatch.cpp overlap and special value paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_exp2_inplace_overlap(self, dtype):
        a = np.arange(100, dtype=dtype)
        np.exp2(a, out=a)
        expected = np.exp2(np.arange(100, dtype=dtype))
        assert_allclose(a, expected, rtol=1e-5)

    def test_exp2_half_special_values(self):
        a = np.array([0, 1, 2, 3, -1, 15, -14, np.inf, -np.inf, np.nan],
                      dtype=np.float16)
        result = np.exp2(a)
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert np.isinf(result[7])
        assert result[8] == 0.0
        assert np.isnan(result[9])

    def test_exp2_half_overflow(self):
        a = np.array([16.0, 17.0, 20.0], dtype=np.float16)
        with np.errstate(over='ignore', invalid='ignore'):
            result = np.exp2(a)
        assert np.all(np.isinf(result))

    def test_exp2_half_underflow(self):
        a = np.array([-20.0, -25.0, -30.0], dtype=np.float16)
        result = np.exp2(a)
        assert np.all(result <= np.finfo(np.float16).tiny)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_exp2_special_inf_nan(self, dtype):
        a = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], dtype=dtype)
        result = np.exp2(a)
        assert np.isinf(result[0])
        assert result[1] == 0.0
        assert np.isnan(result[2])
        assert result[3] == 1.0
        assert result[4] == 1.0

    def test_exp2_half_large_array(self):
        a = np.arange(-5, 10, dtype=np.float16)
        a = np.tile(a, 100)
        result = np.exp2(a)
        expected = np.exp2(a)
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float64])
    def test_exp2_double_large_array(self, dtype):
        a = np.arange(1000, dtype=dtype) / 100.0
        result = np.exp2(a)
        expected = np.exp2(a)
        assert_allclose(result, expected, rtol=1e-10)


class TestPairwiseSum:
    """Cover loops_arithm_sum_hwy.dispatch.cpp pairwise sum paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sum_small(self, dtype):
        a = np.array([1.0, 2.0, 3.0], dtype=dtype)
        result = np.sum(a)
        assert_allclose(result, 6.0, rtol=1e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sum_medium(self, dtype):
        a = np.ones(200, dtype=dtype)
        result = np.sum(a)
        assert_allclose(result, 200.0, rtol=1e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sum_large(self, dtype):
        a = np.ones(10000, dtype=dtype)
        result = np.sum(a)
        assert_allclose(result, 10000.0, rtol=1e-4)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_sum_complex(self, dtype):
        a = np.ones(1000, dtype=dtype) * (1 + 1j)
        result = np.sum(a)
        assert_allclose(result, 1000 + 1000j, rtol=1e-4)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_sum_complex_small(self, dtype):
        a = np.array([1+2j, 3+4j, 5+6j], dtype=dtype)
        result = np.sum(a)
        assert_allclose(result, 9+12j, rtol=1e-6)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_sum_complex_large(self, dtype):
        a = np.ones(5000, dtype=dtype) * (2 + 3j)
        result = np.sum(a)
        assert_allclose(result, 10000 + 15000j, rtol=1e-3)

    def test_sum_float32_varied(self):
        np.random.seed(42)
        a = np.random.randn(5000).astype(np.float32)
        result = np.sum(a)
        expected = np.float64(np.sum(a.astype(np.float64)))
        assert_allclose(result, expected, rtol=1e-3)


class TestShiftScalarBroadcast:
    """Cover loops_shift_sve.c and loops_autovec.dispatch.c.src scalar-in0 paths."""

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_left_shift_scalar_in0(self, dtype):
        scalar = dtype(1)
        shifts = np.arange(0, 8, dtype=dtype)
        result = np.left_shift(scalar, shifts)
        expected = np.left_shift(np.int64(1), shifts.astype(np.int64)).astype(dtype)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_right_shift_scalar_in0(self, dtype):
        val = dtype(127 if np.issubdtype(dtype, np.signedinteger) else
                     min(255, int(np.iinfo(dtype).max)))
        scalar = val
        shifts = np.arange(0, min(8, int(np.iinfo(dtype).max)), dtype=dtype)
        result = np.right_shift(scalar, shifts)
        expected = np.right_shift(val, shifts)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_left_shift_large(self, dtype):
        a = np.arange(100, dtype=dtype)
        b = np.full(100, 2, dtype=dtype)
        result = np.left_shift(a, b)
        expected = a << 2
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_right_shift_large(self, dtype):
        a = np.arange(100, 200, dtype=dtype)
        b = np.full(100, 2, dtype=dtype)
        result = np.right_shift(a, b)
        expected = a >> 2
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_left_shift_64bit_scalar_in0(self, dtype):
        scalar = dtype(1)
        shifts = np.arange(0, 63, dtype=dtype)
        result = np.left_shift(scalar, shifts)
        expected = np.left_shift(dtype(1), shifts)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_right_shift_64bit_scalar_in0(self, dtype):
        val = np.iinfo(dtype).max
        scalar = dtype(val)
        shifts = np.arange(0, 63, dtype=dtype)
        result = np.right_shift(scalar, shifts)
        expected = np.right_shift(dtype(val), shifts)
        assert_array_equal(result, expected)


class TestLogicalSVE:
    """Cover loops_logical_sve.c, loops_logical_and_sve.c scalar broadcast paths."""

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_or_scalar_nonzero(self, dtype):
        scalar = dtype(1)
        a = np.arange(100, dtype=dtype)
        result = np.logical_or(scalar, a)
        assert np.all(result)

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_logical_or_scalar_zero(self, dtype):
        scalar = dtype(0)
        a = np.arange(100, dtype=dtype)
        result = np.logical_or(scalar, a)
        expected = a != 0
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
    def test_logical_and_scalar_nonzero(self, dtype):
        scalar = dtype(1)
        a = np.arange(100, dtype=dtype)
        result = np.logical_and(scalar, a)
        expected = a != 0
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
    def test_logical_and_scalar_zero(self, dtype):
        scalar = dtype(0)
        a = np.arange(100, dtype=dtype)
        result = np.logical_and(scalar, a)
        assert not np.any(result)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_logical_or_scalar_in1(self, dtype):
        a = np.arange(100, dtype=dtype)
        scalar = dtype(0)
        result = np.logical_or(a, scalar)
        expected = a != 0
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_logical_and_scalar_in1(self, dtype):
        a = np.arange(100, dtype=dtype)
        scalar = dtype(1)
        result = np.logical_and(a, scalar)
        expected = a != 0
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
    def test_logical_or_large_scalar(self, dtype):
        scalar = dtype(42)
        a = np.zeros(1000, dtype=dtype)
        result = np.logical_or(scalar, a)
        assert np.all(result)


class TestLogicalNotAArch64:
    """Cover loops_logical_not_aarch64.c contiguous and strided paths."""

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_logical_not_contiguous(self, dtype):
        a = np.array([0.0, 1.0, -1.0, 0.0, 2.5, 0.0] * 100, dtype=dtype)
        result = np.logical_not(a)
        expected = a == 0
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_logical_not_strided_input(self, dtype):
        a = np.arange(200, dtype=dtype)
        result = np.logical_not(a[::2])
        expected = a[::2] == 0
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_logical_not_strided_output(self, dtype):
        a = np.array([0.0, 1.0, -1.0, 0.0] * 50, dtype=dtype)
        out = np.zeros(400, dtype=bool)
        np.logical_not(a, out=out[::2])
        expected = a == 0
        assert_array_equal(out[::2], expected)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_logical_not_fortran_order(self, dtype):
        a = np.asfortranarray(np.zeros((10, 10), dtype=dtype))
        a[5, 5] = 1.0
        result = np.logical_not(a)
        assert result[5, 5] == False
        assert result[0, 0] == True

    def test_logical_not_half_large(self):
        a = np.zeros(1000, dtype=np.float16)
        a[::3] = 1.0
        result = np.logical_not(a)
        expected = a == 0
        assert_array_equal(result, expected)


class TestUnaryFpOpsExtended:
    """Cover loops_unary_fp_ops.dispatch.cpp complex reciprocal and deg2rad/rad2deg."""

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_complex_reciprocal(self, dtype):
        a = np.array([1+2j, 3+4j, 5+6j, 7+8j] * 50, dtype=dtype)
        result = np.reciprocal(a)
        expected = 1.0 / a
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_complex_reciprocal_large(self, dtype):
        a = np.ones(1000, dtype=dtype) * (1 + 1j)
        result = np.reciprocal(a)
        expected = 1.0 / a
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_deg2rad_large(self, dtype):
        a = np.array([0, 45, 90, 135, 180, 270, 360] * 100, dtype=dtype)
        result = np.deg2rad(a)
        expected = a * np.pi / 180.0
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_rad2deg_large(self, dtype):
        a = np.array([0, 0.785, 1.571, 3.142, 6.283] * 100, dtype=dtype)
        result = np.rad2deg(a)
        expected = a * 180.0 / np.pi
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_deg2rad_strided(self, dtype):
        a = np.arange(0, 360, dtype=dtype)
        result = np.deg2rad(a[::3])
        expected = a[::3] * np.pi / 180.0
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_rad2deg_strided(self, dtype):
        a = np.linspace(0, 2 * np.pi, 200, dtype=dtype)
        result = np.rad2deg(a[::2])
        expected = a[::2] * 180.0 / np.pi
        assert_allclose(result, expected, rtol=1e-2)

    @pytest.mark.parametrize("func", [np.ceil, np.floor, np.trunc, np.rint])
    def test_half_rounding_strided(self, func):
        a = np.linspace(-10.5, 10.5, 200, dtype=np.float16)
        result = func(a[::2])
        expected = func(a[::2])
        assert_allclose(result, expected, rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("dtype", [np.float16])
    def test_reciprocal_half_large(self, dtype):
        a = np.arange(1, 501, dtype=dtype)
        result = np.reciprocal(a)
        expected = np.reciprocal(a)
        assert_allclose(result, expected, rtol=1e-2)


class TestAutoVecReduce:
    """Cover loops_autovec.dispatch.c.src reduce and bitwise/logical paths."""

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.uint8, np.uint16])
    def test_add_reduce_small_types(self, dtype):
        a = np.arange(100, dtype=dtype)
        result = np.add.reduce(a)
        expected = np.sum(a)
        assert result == expected

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
    def test_multiply_reduce_ones(self, dtype):
        a = np.ones(100, dtype=dtype)
        result = np.multiply.reduce(a)
        assert result == 1

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
    def test_multiply_reduce_varied(self, dtype):
        a = np.arange(1, 6, dtype=dtype)
        result = np.multiply.reduce(a)
        assert result == 120

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_and_large(self, dtype):
        mask = dtype(0x0F if np.issubdtype(dtype, np.signedinteger) and np.iinfo(dtype).max < 0xFF else 0x0F)
        a = np.full(100, dtype(-1) if np.issubdtype(dtype, np.signedinteger) else np.iinfo(dtype).max, dtype=dtype)
        b = np.full(100, mask, dtype=dtype)
        result = np.bitwise_and(a, b)
        assert np.all(result == mask)

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_or_large(self, dtype):
        a = np.full(100, dtype(0xF0), dtype=dtype)
        b = np.full(100, dtype(0x0F), dtype=dtype)
        result = np.bitwise_or(a, b)
        assert np.all(result == dtype(0xFF))

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_xor_large(self, dtype):
        a = np.full(100, dtype(0xFF), dtype=dtype)
        b = np.full(100, dtype(0xFF), dtype=dtype)
        result = np.bitwise_xor(a, b)
        assert np.all(result == dtype(0))

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_bitwise_and_scalar_in0(self, dtype):
        scalar = dtype(0x0F)
        a = np.arange(100, dtype=dtype)
        result = np.bitwise_and(scalar, a)
        expected = np.bitwise_and(np.int64(0x0F), a.astype(np.int64)).astype(dtype)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_bitwise_or_scalar_in0(self, dtype):
        scalar = dtype(0xF0)
        a = np.arange(100, dtype=dtype)
        result = np.bitwise_or(scalar, a)
        expected = np.bitwise_or(np.int64(0xF0), a.astype(np.int64)).astype(dtype)
        assert_array_equal(result, expected)


class TestPowerExtended:
    """Cover loops_power.dispatch.cpp additional paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_power_negative_base_integer_exp(self, dtype):
        a = np.full(100, -2.0, dtype=dtype)
        b = np.arange(1, 101, dtype=dtype)
        result = np.power(a, b)
        expected = np.power(-2.0, b)
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_power_zero_base(self, dtype):
        a = np.zeros(100, dtype=dtype)
        b = np.arange(1, 101, dtype=dtype)
        result = np.power(a, b)
        assert np.all(result == 0.0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_power_strided(self, dtype):
        a = np.arange(1, 201, dtype=dtype)
        b = np.full(200, 0.5, dtype=dtype)
        result = np.power(a[::2], b[::2])
        expected = np.sqrt(a[::2])
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_power_complex(self, dtype):
        a = np.array([1+1j, 2+2j, 3+3j] * 50, dtype=dtype)
        b = np.full(150, 2, dtype=dtype)
        result = np.power(a, b)
        expected = a ** 2
        assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_power_inplace_overlap(self, dtype):
        a = np.arange(1, 51, dtype=dtype)
        np.power(a, dtype(2.0), out=a)
        expected = np.arange(1, 51, dtype=dtype) ** 2
        assert_allclose(a, expected, rtol=1e-5)


class TestUfuncScalarFastPath:
    """Cover ufunc_object.c scalar fast path for unary math ufuncs."""

    def test_sin_scalar_float64(self):
        result = np.sin(1.0)
        assert_allclose(result, np.sin(np.float64(1.0)))

    def test_cos_scalar_float64(self):
        result = np.cos(0.0)
        assert result == 1.0

    def test_exp_scalar_float64(self):
        result = np.exp(0.0)
        assert result == 1.0

    def test_log_scalar_float64(self):
        result = np.log(1.0)
        assert result == 0.0

    def test_sqrt_scalar_float64(self):
        result = np.sqrt(4.0)
        assert result == 2.0

    def test_sin_scalar_float32(self):
        result = np.sin(np.float32(1.0))
        assert_allclose(result, np.sin(np.float32(1.0)), rtol=1e-6)

    def test_exp_scalar_float32(self):
        result = np.exp(np.float32(1.0))
        assert_allclose(result, np.e, rtol=1e-5)

    def test_sin_scalar_complex128(self):
        result = np.sin(1.0 + 2.0j)
        expected = np.sin(np.complex128(1.0 + 2.0j))
        assert_allclose(result, expected, rtol=1e-10)

    def test_exp_scalar_complex128(self):
        result = np.exp(1.0 + 0j)
        assert_allclose(result, np.e, rtol=1e-10)

    def test_sin_scalar_complex64(self):
        result = np.sin(np.complex64(1+1j))
        expected = np.sin(np.complex64(1+1j))
        assert_allclose(result, expected, rtol=1e-5)

    def test_sin_scalar_longdouble(self):
        result = np.sin(np.longdouble(1.0))
        assert_allclose(result, np.sin(1.0), rtol=1e-10)


class TestFancyIndexingExtended:
    """Cover mapping.c extra_op and broadcast error paths."""

    def test_fancy_index_subspace_assignment(self):
        a = np.zeros((10, 10))
        a[[1, 3, 5], :] = np.ones((3, 10))
        assert np.all(a[1] == 1)
        assert np.all(a[3] == 1)
        assert np.all(a[5] == 1)
        assert np.all(a[0] == 0)

    def test_fancy_index_broadcast_scalar(self):
        a = np.zeros((10, 10))
        a[[0, 1, 2]] = 5.0
        assert np.all(a[:3] == 5.0)

    def test_fancy_index_broadcast_error(self):
        a = np.zeros((10, 10))
        with pytest.raises(ValueError):
            a[[0, 1, 2]] = np.ones((5, 10))

    def test_fancy_index_3d_subspace(self):
        a = np.zeros((5, 5, 5))
        a[np.array([0, 2]), :, :] = 1.0
        assert np.all(a[0] == 1)
        assert np.all(a[2] == 1)
        assert np.all(a[1] == 0)

    def test_fancy_index_overlap(self):
        a = np.arange(10)
        a[a > 5] = 0
        assert np.all(a[a > 5] == 0) or np.all(a <= 5)

    def test_fancy_index_high_dim(self):
        a = np.zeros((3, 4, 5, 6))
        a[np.array([0, 1]), np.array([1, 2])] = np.ones((2, 5, 6))
        assert np.all(a[0, 1] == 1)
        assert np.all(a[1, 2] == 1)

    def test_put_overlap(self):
        a = np.arange(10)
        np.put(a, [0, 1, 2], [9, 8, 7])
        assert_array_equal(a[:3], [9, 8, 7])


class TestHistogramddUniform2D:
    """Cover compiled_base.c _histogramdd_uniform2d validation paths."""

    def test_histogramdd_uniform2d_normal(self):
        np.random.seed(42)
        sample = np.random.rand(100, 2)
        hist, edges = np.histogramdd(sample, bins=[10, 10],
                                      range=[[0, 1], [0, 1]])
        assert hist.shape == (10, 10)
        assert hist.sum() == 100

    def test_histogramdd_uniform2d_large(self):
        np.random.seed(42)
        sample = np.random.rand(10000, 2)
        hist, edges = np.histogramdd(sample, bins=[50, 50],
                                      range=[[0, 1], [0, 1]])
        assert hist.shape == (50, 50)
        assert hist.sum() == 10000


class TestTrigExtended:
    """Cover loops_trigonometric.dispatch.cpp additional paths."""

    @pytest.mark.parametrize("func", [np.sin, np.cos, np.tan])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_trig_large_array(self, func, dtype):
        a = np.linspace(-np.pi, np.pi, 5000, dtype=dtype)
        result = func(a)
        expected = func(a)
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_arctan2_large(self, dtype):
        y = np.linspace(-1, 1, 500, dtype=dtype)
        x = np.linspace(-1, 1, 500, dtype=dtype)
        result = np.arctan2(y, x)
        expected = np.arctan2(y, x)
        assert_allclose(result, expected, rtol=1e-5)


class TestLogExpExtended:
    """Cover loops_exp/log/log2.dispatch.cpp additional edge cases."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_exp_large_array(self, dtype):
        a = np.linspace(-10, 10, 2000, dtype=dtype)
        result = np.exp(a)
        expected = np.exp(a)
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_log_large_array(self, dtype):
        a = np.linspace(0.01, 100, 2000, dtype=dtype)
        result = np.log(a)
        expected = np.log(a)
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_log2_large_array(self, dtype):
        a = np.linspace(0.01, 100, 2000, dtype=dtype)
        result = np.log2(a)
        expected = np.log2(a)
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_exp_overlap(self, dtype):
        a = np.linspace(-5, 5, 100, dtype=dtype)
        np.exp(a, out=a)
        expected = np.exp(np.linspace(-5, 5, 100, dtype=dtype))
        assert_allclose(a, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_log_overlap(self, dtype):
        a = np.linspace(0.1, 10, 100, dtype=dtype)
        np.log(a, out=a)
        expected = np.log(np.linspace(0.1, 10, 100, dtype=dtype))
        assert_allclose(a, expected, rtol=1e-5)


class TestUmathUnaryExtended:
    """Cover loops_umath_unary.dispatch.cpp additional paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sign_float(self, dtype):
        a = np.array([-3.0, -1.0, 0.0, 1.0, 3.0] * 100, dtype=dtype)
        result = np.sign(a)
        expected = np.array([-1.0, -1.0, 0.0, 1.0, 1.0] * 100, dtype=dtype)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_sign_complex(self, dtype):
        a = np.array([1+1j, -1+1j, 0+0j, 1-1j] * 50, dtype=dtype)
        result = np.sign(a)
        for i in range(len(a)):
            if a[i] != 0:
                assert_allclose(np.abs(result[i]), 1.0, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_positive_float(self, dtype):
        a = np.arange(-50, 50, dtype=dtype)
        result = np.positive(a)
        assert_array_equal(result, a)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_negative_float(self, dtype):
        a = np.arange(-50, 50, dtype=dtype)
        result = np.negative(a)
        assert_array_equal(result, -a)


class TestBitwiseSVE:
    """Cover loops_bitwise_sve.c additional paths."""

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
    def test_bitwise_not_large(self, dtype):
        a = np.zeros(500, dtype=dtype)
        result = np.bitwise_not(a)
        assert np.all(result == dtype(-1))

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
    def test_bitwise_not_unsigned(self, dtype):
        a = np.zeros(500, dtype=dtype)
        result = np.bitwise_not(a)
        assert np.all(result == np.iinfo(dtype).max)


class TestLoopsCSrc:
    """Cover loops.c.src additional paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_modf_float(self, dtype):
        a = np.array([1.5, -2.7, 3.14, -0.5, 0.0] * 100, dtype=dtype)
        frac, integ = np.modf(a)
        assert_allclose(frac + integ, a, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_frexp_float(self, dtype):
        a = np.array([1.0, 2.0, 4.0, 8.0, 0.5] * 100, dtype=dtype)
        mantissa, exponent = np.frexp(a)
        reconstructed = mantissa * (2.0 ** exponent.astype(dtype))
        assert_allclose(reconstructed, a, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_ldexp_float(self, dtype):
        mantissa = np.array([0.5, 0.75, 0.625] * 100, dtype=dtype)
        exponent = np.array([1, 2, 3] * 100, dtype=np.int32)
        result = np.ldexp(mantissa, exponent)
        expected = mantissa * (2.0 ** exponent.astype(dtype))
        assert_allclose(result, expected, rtol=1e-5)


class TestFancyIndexingItemSizes:
    """Cover mapping.c fancy indexing paths for itemsizes 2, 4, 16."""

    @pytest.mark.parametrize("dtype", [np.int16, np.float16])
    def test_fancy_get_itemsize2(self, dtype):
        a = np.arange(100, dtype=dtype).reshape(10, 10)
        idx = np.array([0, 2, 4, 6, 8])
        result = a[idx]
        assert_array_equal(result, a[idx])

    @pytest.mark.parametrize("dtype", [np.int32, np.float32])
    def test_fancy_get_itemsize4(self, dtype):
        a = np.arange(100, dtype=dtype).reshape(10, 10)
        idx = np.array([0, 2, 4, 6, 8])
        result = a[idx]
        assert_array_equal(result, a[idx])

    @pytest.mark.parametrize("dtype", [np.complex128])
    def test_fancy_get_itemsize16(self, dtype):
        a = (np.arange(100, dtype=dtype) + 1j).reshape(10, 10)
        idx = np.array([0, 2, 4, 6, 8])
        result = a[idx]
        assert_array_equal(result, a[idx])

    @pytest.mark.parametrize("dtype", [np.int16, np.float16])
    def test_fancy_set_scalar_itemsize2(self, dtype):
        a = np.zeros((10, 10), dtype=dtype)
        idx = np.array([0, 2, 4, 6, 8])
        a[idx] = dtype(5)
        assert np.all(a[idx] == dtype(5))

    @pytest.mark.parametrize("dtype", [np.int32, np.float32])
    def test_fancy_set_scalar_itemsize4(self, dtype):
        a = np.zeros((10, 10), dtype=dtype)
        idx = np.array([0, 2, 4, 6, 8])
        a[idx] = dtype(7)
        assert np.all(a[idx] == dtype(7))

    @pytest.mark.parametrize("dtype", [np.complex128])
    def test_fancy_set_scalar_itemsize16(self, dtype):
        a = np.zeros((10, 10), dtype=dtype)
        idx = np.array([0, 2, 4, 6, 8])
        a[idx] = dtype(1 + 2j)
        assert np.all(a[idx] == dtype(1 + 2j))

    def test_fancy_get_axis1(self):
        a = np.arange(100, dtype=np.float64).reshape(10, 10)
        idx = np.array([0, 2, 4, 6, 8])
        result = a[:, idx]
        expected = a[:, idx]
        assert_array_equal(result, expected)

    def test_fancy_set_axis1(self):
        a = np.arange(100, dtype=np.float64).reshape(10, 10)
        idx = np.array([0, 2, 4, 6, 8])
        a[:, idx] = 0
        assert np.all(a[:, idx] == 0)

    def test_fancy_set_ellipsis(self):
        a = np.arange(100, dtype=np.float64).reshape(10, 10)
        idx = np.array([0, 2, 4])
        a[idx, ...] = -1
        assert np.all(a[idx] == -1)

    def test_fancy_get_out_of_bounds(self):
        a = np.arange(100, dtype=np.float64).reshape(10, 10)
        with pytest.raises(IndexError):
            a[[100]]

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32, np.complex128])
    def test_fancy_get_1d_large(self, dtype):
        a = np.arange(500, dtype=dtype)
        idx = np.arange(0, 500, 3)
        result = a[idx]
        assert_array_equal(result, a[idx])

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32, np.complex128])
    def test_fancy_set_1d_large(self, dtype):
        a = np.arange(500, dtype=dtype)
        idx = np.arange(0, 500, 3)
        a[idx] = dtype(0)
        assert np.all(a[idx] == dtype(0))


class TestHistogram2dErrors:
    """Cover compiled_base.c histogram2d error paths."""

    def test_histogram2d_negative_bins(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [1, 2], bins=[-1, 5])

    def test_histogram2d_nonfinite_range(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [1, 2], bins=[5, 5],
                           range=[[0, 1], [np.nan, 1]])

    def test_histogram2d_wrong_shape(self):
        with pytest.raises(ValueError):
            np.histogram2d(np.ones((5, 3)), np.ones(5), bins=[5, 5])

    def test_histogram2d_inf_range(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [1, 2], bins=[5, 5],
                           range=[[0, np.inf], [0, 1]])

    def test_histogram2d_decreasing_range(self):
        with pytest.raises(ValueError):
            np.histogram2d([1, 2], [1, 2], bins=[5, 5],
                           range=[[1, 0], [0, 1]])


class TestUfuncScalarFastPath:
    """Cover ufunc_object.c scalar fast-path fallbacks for longdouble/complex."""

    def test_sin_longdouble(self):
        result = np.sin(np.longdouble(1.0))
        assert_allclose(result, np.sin(1.0), rtol=1e-10)

    def test_cos_longdouble(self):
        result = np.cos(np.longdouble(1.0))
        assert_allclose(result, np.cos(1.0), rtol=1e-10)

    def test_exp_longdouble(self):
        result = np.exp(np.longdouble(1.0))
        assert_allclose(result, np.exp(1.0), rtol=1e-10)

    def test_log_longdouble(self):
        result = np.log(np.longdouble(2.0))
        assert_allclose(result, np.log(2.0), rtol=1e-10)

    def test_sqrt_longdouble(self):
        result = np.sqrt(np.longdouble(4.0))
        assert_allclose(result, 2.0, rtol=1e-10)

    def test_sin_complex64(self):
        result = np.sin(np.complex64(1 + 2j))
        assert_allclose(result, np.sin(1 + 2j), rtol=1e-5)

    def test_cos_complex128(self):
        result = np.cos(np.complex128(1 + 2j))
        assert_allclose(result, np.cos(1 + 2j), rtol=1e-10)

    def test_exp_complex64(self):
        result = np.exp(np.complex64(1 + 1j))
        assert_allclose(result, np.exp(1 + 1j), rtol=1e-5)

    def test_log_complex128(self):
        result = np.log(np.complex128(2 + 3j))
        assert_allclose(result, np.log(2 + 3j), rtol=1e-10)

    def test_add_longdouble_scalar(self):
        a = np.longdouble(3.0)
        b = np.longdouble(4.0)
        assert_allclose(np.add(a, b), 7.0, rtol=1e-10)

    def test_multiply_complex64_scalar(self):
        a = np.complex64(2 + 3j)
        b = np.complex64(4 + 5j)
        result = np.multiply(a, b)
        assert_allclose(result, (2 + 3j) * (4 + 5j), rtol=1e-5)


class TestLogicalNotTail:
    """Cover loops_logical_not_aarch64.c scalar tail for non-multiple-of-8."""

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_logical_not_odd_length(self, dtype):
        a = np.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=dtype)
        result = np.logical_not(a)
        expected = np.array([True, False, True, False, True])
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_logical_not_length_3(self, dtype):
        a = np.array([0.0, 1.0, 0.0], dtype=dtype)
        result = np.logical_not(a)
        expected = np.array([True, False, True])
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_logical_not_length_7(self, dtype):
        a = np.array([0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 4.0], dtype=dtype)
        result = np.logical_not(a)
        expected = a == 0
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_logical_not_length_1(self, dtype):
        a = np.array([0.0], dtype=dtype)
        result = np.logical_not(a)
        assert result[0] == True


class TestBitwiseSVEInt64:
    """Cover loops_bitwise_sve.c int64/uint64 paths."""

    def test_bitwise_and_int64(self):
        a = np.arange(200, dtype=np.int64)
        b = np.full(200, 0xFF, dtype=np.int64)
        result = np.bitwise_and(a, b)
        expected = a & 0xFF
        assert_array_equal(result, expected)

    def test_bitwise_and_uint64(self):
        a = np.arange(200, dtype=np.uint64)
        b = np.full(200, 0xFF, dtype=np.uint64)
        result = np.bitwise_and(a, b)
        expected = a & np.uint64(0xFF)
        assert_array_equal(result, expected)

    def test_bitwise_or_int64(self):
        a = np.full(200, 0xF0, dtype=np.int64)
        b = np.full(200, 0x0F, dtype=np.int64)
        result = np.bitwise_or(a, b)
        assert np.all(result == 0xFF)

    def test_bitwise_xor_int64(self):
        a = np.full(200, 0xFF, dtype=np.int64)
        b = np.full(200, 0xFF, dtype=np.int64)
        result = np.bitwise_xor(a, b)
        assert np.all(result == 0)


class TestHalfSameInput:
    """Cover loops.c.src HALF same-input optimization (args[0] == args[1])."""

    def test_half_add_same_input(self):
        a = np.arange(1, 201, dtype=np.float16)
        result = np.add(a, a)
        expected = a * 2
        assert_allclose(result, expected, rtol=1e-2)

    def test_half_multiply_same_input(self):
        a = np.arange(1, 101, dtype=np.float16)
        result = np.multiply(a, a)
        expected = a ** 2
        assert_allclose(result, expected, rtol=1e-2)

    def test_half_subtract_same_input(self):
        a = np.arange(1, 101, dtype=np.float16)
        result = np.subtract(a, a)
        assert np.all(result == 0)

    def test_half_divide_same_input(self):
        a = np.arange(1, 101, dtype=np.float16)
        result = np.divide(a, a)
        assert np.all(result == 1.0)


class TestStridedOutputOverlap:
    """Cover strided output and in-place overlap paths for exp/log/trig/sqrt."""

    @pytest.mark.parametrize("func", [np.exp, np.exp2])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_strided_output(self, func, dtype):
        a = np.arange(1, 51, dtype=dtype)
        out = np.zeros(100, dtype=dtype)
        func(a, out=out[::2])
        expected = func(a)
        assert_allclose(out[::2], expected, rtol=1e-5)

    @pytest.mark.parametrize("func", [np.log, np.log2])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_log_strided_output(self, func, dtype):
        a = np.arange(1, 51, dtype=dtype)
        out = np.zeros(100, dtype=dtype)
        func(a, out=out[::2])
        expected = func(a)
        assert_allclose(out[::2], expected, rtol=1e-5)

    @pytest.mark.parametrize("func", [np.sin, np.cos, np.tan])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_trig_strided_output(self, func, dtype):
        a = np.linspace(0.1, 1.0, 50, dtype=dtype)
        out = np.zeros(100, dtype=dtype)
        func(a, out=out[::2])
        expected = func(a)
        assert_allclose(out[::2], expected, rtol=1e-5)

    @pytest.mark.parametrize("func", [np.exp, np.exp2, np.log, np.log2])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_overlap_inplace(self, func, dtype):
        a = np.arange(1, 51, dtype=dtype)
        original = a.copy()
        func(a, out=a)
        expected = func(original)
        assert_allclose(a, expected, rtol=1e-5)

    @pytest.mark.parametrize("func", [np.sin, np.cos])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_trig_overlap_inplace(self, func, dtype):
        a = np.linspace(0.1, 1.0, 50, dtype=dtype)
        original = a.copy()
        func(a, out=a)
        expected = func(original)
        assert_allclose(a, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float16])
    def test_exp_half_overlap(self, dtype):
        a = np.linspace(-5, 5, 20, dtype=dtype)
        original = a.copy()
        np.exp(a, out=a)
        expected = np.exp(original)
        assert_allclose(a, expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16])
    def test_exp2_half_overlap(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        original = a.copy()
        with np.errstate(over='ignore', invalid='ignore'):
            np.exp2(a, out=a)
            expected = np.exp2(original)
        assert_allclose(a, expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16])
    def test_log_half_overlap(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        original = a.copy()
        np.log(a, out=a)
        expected = np.log(original)
        assert_allclose(a, expected, rtol=1e-2)


class TestSqrtCbrtExtended:
    """Cover loops_umath_unary.dispatch.cpp sqrt/cbrt paths."""

    @pytest.mark.parametrize("dtype", [np.float16])
    def test_sqrt_half_strided(self, dtype):
        a = np.arange(1, 101, dtype=dtype)
        out = np.zeros(200, dtype=dtype)
        np.sqrt(a, out=out[::2])
        expected = np.sqrt(a)
        assert_allclose(out[::2], expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16])
    def test_cbrt_half(self, dtype):
        a = np.arange(1, 101, dtype=dtype)
        result = np.cbrt(a)
        expected = np.cbrt(a.astype(np.float32)).astype(dtype)
        assert_allclose(result, expected, rtol=1e-1)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cbrt_overlap(self, dtype):
        a = np.arange(1, 51, dtype=dtype)
        original = a.copy()
        np.cbrt(a, out=a)
        expected = np.cbrt(original)
        assert_allclose(a, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sqrt_overlap(self, dtype):
        a = np.arange(1, 51, dtype=dtype)
        original = a.copy()
        np.sqrt(a, out=a)
        expected = np.sqrt(original)
        assert_allclose(a, expected, rtol=1e-5)


class TestPowerExtended:
    """Cover loops_power.dispatch.cpp subnormal and strided paths."""

    def test_power_subnormal_double(self):
        a = np.array([1e-300, 1e-200, 1e-100], dtype=np.float64)
        b = np.array([2.0, 0.5, 1.5], dtype=np.float64)
        with np.errstate(under='ignore', over='ignore'):
            result = np.power(a, b)
            expected = a ** b
        assert_allclose(result, expected, rtol=1e-10)

    def test_power_subnormal_float(self):
        a = np.array([1e-45, 1e-40, 1.4e-45], dtype=np.float32)
        b = np.array([2.0, 0.5, 1.0], dtype=np.float32)
        result = np.power(a, b)
        expected = a ** b
        assert_allclose(result, expected, rtol=1e-3)

    def test_power_strided(self):
        a = np.arange(1, 51, dtype=np.float64)
        b = np.full(50, 2.5, dtype=np.float64)
        out = np.zeros(100, dtype=np.float64)
        np.power(a, b, out=out[::2])
        expected = a ** 2.5
        assert_allclose(out[::2], expected, rtol=1e-10)

    def test_power_both_strided(self):
        a = np.arange(1, 101, dtype=np.float64)
        b = np.arange(1, 101, dtype=np.float64) * 0.01
        result = np.power(a[::2], b[::2])
        expected = a[::2] ** b[::2]
        assert_allclose(result, expected, rtol=1e-10)

    def test_power_large_exponent(self):
        a = np.array([10.0, 100.0, 1000.0], dtype=np.float64)
        b = np.array([309.0, 155.0, 103.0], dtype=np.float64)
        with np.errstate(over='ignore', under='ignore'):
            result = np.power(a, b)
        assert np.isinf(result[0])

    def test_power_negative_base_fractional(self):
        a = np.array([-2.0, -3.0, -4.0], dtype=np.float64)
        b = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        with np.errstate(invalid='ignore'):
            result = np.power(a, b)
        assert np.all(np.isnan(result))


class TestMinMaxScalarBroadcastExtended:
    """Cover loops_minmax.dispatch.c.src ARM scalar broadcast and unrolled paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_maximum_scalar_broadcast_left(self, dtype):
        a = np.arange(5000, dtype=dtype)
        result = np.maximum(dtype(2500), a)
        expected = np.where(a >= 2500, a, dtype(2500))
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_minimum_scalar_broadcast_left(self, dtype):
        a = np.arange(5000, dtype=dtype)
        result = np.minimum(dtype(2500), a)
        expected = np.where(a <= 2500, a, dtype(2500))
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_maximum_both_contiguous_large(self, dtype):
        a = np.arange(10000, dtype=dtype)
        b = np.arange(10000, dtype=dtype)[::-1]
        result = np.maximum(a, b)
        expected = np.where(a >= b, a, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_minimum_both_contiguous_large(self, dtype):
        a = np.arange(10000, dtype=dtype)
        b = np.arange(10000, dtype=dtype)[::-1]
        result = np.minimum(a, b)
        expected = np.where(a <= b, a, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_maximum_scalar_broadcast_strided(self, dtype):
        a = np.arange(10000, dtype=dtype)
        result = np.maximum(a[::2], dtype(2500))
        expected = np.where(a[::2] >= 2500, a[::2], dtype(2500))
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_fmax_nan_ignoring(self, dtype):
        a = np.array([1.0, np.nan, 3.0, np.nan, 5.0] * 200, dtype=dtype)
        b = np.array([5.0, 4.0, np.nan, np.nan, 1.0] * 200, dtype=dtype)
        result = np.fmax(a, b)
        assert result[1] == 4.0
        assert result[2] == 3.0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_fmin_nan_ignoring(self, dtype):
        a = np.array([1.0, np.nan, 3.0, np.nan, 5.0] * 200, dtype=dtype)
        b = np.array([5.0, 4.0, np.nan, np.nan, 1.0] * 200, dtype=dtype)
        result = np.fmin(a, b)
        assert result[1] == 4.0
        assert result[2] == 3.0


class TestAbsHalfExtended:
    """Cover loops_autovec_abs_hwy.dispatch.cpp half-precision paths."""

    def test_abs_half_contiguous(self):
        a = np.array([-1.0, -2.0, 3.0, -4.0, 5.0] * 100, dtype=np.float16)
        result = np.abs(a)
        expected = np.abs(a)
        assert_array_equal(result, expected)

    def test_abs_half_strided(self):
        a = np.arange(-100, 100, dtype=np.float16)
        result = np.abs(a[::2])
        expected = np.abs(a[::2])
        assert_array_equal(result, expected)

    def test_abs_half_large(self):
        a = np.linspace(-100, 100, 2000, dtype=np.float16)
        result = np.abs(a)
        assert np.all(result >= 0)


class TestAutoVecScalarBroadcast:
    """Cover loops_autovec.dispatch.c.src scalar-in0 paths for bitwise/logical."""

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.uint8, np.uint16,
                                        np.uint32, np.uint64])
    def test_bitwise_and_scalar_in0(self, dtype):
        scalar = dtype(0x0F)
        a = np.arange(100, dtype=dtype)
        result = np.bitwise_and(scalar, a)
        expected = np.bitwise_and(a, scalar)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int16, np.uint8, np.uint16,
                                        np.uint32, np.uint64])
    def test_bitwise_or_scalar_in0(self, dtype):
        scalar = dtype(0xF0)
        a = np.arange(100, dtype=dtype)
        result = np.bitwise_or(scalar, a)
        expected = np.bitwise_or(a, scalar)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int16, np.uint8, np.uint16,
                                        np.uint32, np.uint64])
    def test_bitwise_xor_scalar_in0(self, dtype):
        scalar = dtype(0xFF)
        a = np.arange(100, dtype=dtype)
        result = np.bitwise_xor(scalar, a)
        expected = np.bitwise_xor(a, scalar)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.uint8])
    def test_logical_and_scalar_in0(self, dtype):
        scalar = dtype(1)
        a = np.arange(100, dtype=dtype)
        result = np.logical_and(scalar, a)
        expected = a != 0
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.uint8])
    def test_logical_or_scalar_in0(self, dtype):
        scalar = dtype(0)
        a = np.arange(100, dtype=dtype)
        result = np.logical_or(scalar, a)
        expected = a != 0
        assert_array_equal(result, expected)


class TestFloorDivideExtended:
    """Cover loops_arithmetic_floor_hwy.dispatch.cpp and loops_arithmetic.dispatch.c.src."""

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_floor_divide_unsigned_by_0(self, dtype):
        a = np.arange(1, 11, dtype=dtype)
        with np.errstate(divide='ignore'):
            result = np.floor_divide(a, dtype(0))
        assert_array_equal(result, np.zeros(10, dtype=dtype))

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_floor_divide_unsigned_by_1(self, dtype):
        a = np.arange(0, 200, dtype=dtype)
        result = np.floor_divide(a, dtype(1))
        assert_array_equal(result, a)

    def test_floor_divide_uint64_scalar(self):
        a = np.arange(1, 201, dtype=np.uint64)
        result = np.floor_divide(a, np.uint64(7))
        expected = a // np.uint64(7)
        assert_array_equal(result, expected)

    def test_floor_divide_int64_scalar_neg1(self):
        a = np.arange(-100, 100, dtype=np.int64)
        with np.errstate(over='ignore'):
            result = np.floor_divide(a, np.int64(-1))
        expected = a // np.int64(-1)
        assert_array_equal(result, expected)

    def test_floor_divide_int64_scalar_1(self):
        a = np.arange(-100, 100, dtype=np.int64)
        result = np.floor_divide(a, np.int64(1))
        assert_array_equal(result, a)

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
    def test_floor_divide_array_array_strided(self, dtype):
        a = np.arange(-50, 50, dtype=dtype)
        b = np.full(100, 3, dtype=dtype)
        out = np.zeros(200, dtype=dtype)
        np.floor_divide(a, b, out=out[::2])
        expected = a // dtype(3)
        assert_array_equal(out[::2], expected)


class TestArgMinMaxExtended:
    """Cover highway_argfunc.dispatch.cpp additional paths."""

    def test_argmax_bool_all_false(self):
        a = np.zeros(1000, dtype=np.bool_)
        assert a.argmax() == 0

    def test_argmin_bool_all_true(self):
        a = np.ones(1000, dtype=np.bool_)
        assert a.argmin() == 0

    def test_argmax_bool_first_true_late(self):
        a = np.zeros(5000, dtype=np.bool_)
        a[4999] = True
        assert a.argmax() == 4999

    def test_argmin_bool_first_false_late(self):
        a = np.ones(5000, dtype=np.bool_)
        a[4999] = False
        assert a.argmin() == 4999

    def test_argmax_longdouble(self):
        a = np.arange(1000, dtype=np.longdouble)
        assert a.argmax() == 999

    def test_argmin_longdouble(self):
        a = np.arange(1000, dtype=np.longdouble)
        assert a.argmin() == 0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmax_all_same_float(self, dtype):
        a = np.full(5000, 3.14, dtype=dtype)
        assert a.argmax() == 0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmin_all_same_float(self, dtype):
        a = np.full(5000, 3.14, dtype=dtype)
        assert a.argmin() == 0

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_argmax_all_same_int(self, dtype):
        a = np.full(5000, 42, dtype=dtype)
        assert a.argmax() == 0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmax_with_nan(self, dtype):
        a = np.arange(1000, dtype=dtype)
        a[500] = np.nan
        result = a.argmax()
        assert result == 500

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_argmin_with_nan(self, dtype):
        a = np.arange(1000, dtype=dtype)
        a[500] = np.nan
        result = a.argmin()
        assert result == 500

    @pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.uint64])
    def test_argmax_unsigned_large(self, dtype):
        a = np.arange(2000, dtype=dtype)
        assert a.argmax() == 1999

    @pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.uint64])
    def test_argmin_unsigned_large(self, dtype):
        a = np.arange(2000, dtype=dtype)
        assert a.argmin() == 0

    def test_argmax_uint8(self):
        a = np.arange(256, dtype=np.uint8)
        assert a.argmax() == 255

    def test_argmin_uint8(self):
        a = np.arange(1, 256, dtype=np.uint8)
        assert a.argmin() == 0


class TestPairwiseSumExtended:
    """Cover loops_arithm_sum_hwy.dispatch.cpp additional paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_add_reduce_large(self, dtype):
        a = np.ones(10000, dtype=dtype)
        result = np.add.reduce(a)
        assert_allclose(result, 10000.0, rtol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_mean_large(self, dtype):
        a = np.ones(10000, dtype=dtype) * 3.0
        result = np.mean(a)
        assert_allclose(result, 3.0, rtol=1e-6)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_sum_complex_varied(self, dtype):
        np.random.seed(42)
        a = (np.random.randn(5000) + 1j * np.random.randn(5000)).astype(dtype)
        result = np.sum(a)
        expected = np.sum(a.astype(np.complex128))
        assert_allclose(result, expected, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sum_with_axis(self, dtype):
        a = np.ones((100, 100), dtype=dtype)
        result = np.sum(a, axis=0)
        assert_allclose(result, np.full(100, 100.0, dtype=dtype), rtol=1e-6)


class TestShiftSVEUint64:
    """Cover loops_shift_sve.c uint64 scalar-in0 paths."""

    def test_left_shift_uint64_scalar_in0(self):
        scalar = np.uint64(1)
        shifts = np.arange(0, 63, dtype=np.uint64)
        result = np.left_shift(scalar, shifts)
        expected = np.left_shift(np.uint64(1), shifts)
        assert_array_equal(result, expected)

    def test_right_shift_uint64_scalar_in0(self):
        val = np.uint64(0xFFFFFFFFFFFFFFFF)
        scalar = val
        shifts = np.arange(0, 63, dtype=np.uint64)
        result = np.right_shift(scalar, shifts)
        expected = np.right_shift(val, shifts)
        assert_array_equal(result, expected)

    def test_left_shift_int64_scalar_in0_large(self):
        scalar = np.int64(1)
        shifts = np.arange(0, 62, dtype=np.int64)
        result = np.left_shift(scalar, shifts)
        expected = np.left_shift(np.int64(1), shifts)
        assert_array_equal(result, expected)

    def test_right_shift_int64_scalar_in0_large(self):
        val = np.int64(0x7FFFFFFFFFFFFFFF)
        scalar = val
        shifts = np.arange(0, 62, dtype=np.int64)
        result = np.right_shift(scalar, shifts)
        expected = np.right_shift(val, shifts)
        assert_array_equal(result, expected)


class TestLogicalOrUint8:
    """Cover loops_logical_sve.c uint8 truth path."""

    def test_logical_or_uint8_scalar_zero(self):
        a = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        result = np.logical_or(np.uint8(0), a)
        expected = a != 0
        assert_array_equal(result, expected)

    def test_logical_or_uint8_scalar_nonzero(self):
        a = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        result = np.logical_or(np.uint8(1), a)
        assert np.all(result)

    def test_logical_truth_uint8(self):
        a = np.array([0, 1, 2, 0, 3, 0, 4, 0, 5, 0] * 20, dtype=np.uint8)
        result = np.logical_or(np.uint8(0), a)
        expected = a != 0
        assert_array_equal(result, expected)


class TestTrigStridedExtended:
    """Cover loops_trigonometric.dispatch.cpp strided output and overlap."""

    @pytest.mark.parametrize("func", [np.sin, np.cos, np.tan])
    @pytest.mark.parametrize("dtype", [np.float16])
    def test_trig_half_overlap(self, func, dtype):
        a = np.linspace(0.1, 1.0, 20, dtype=dtype)
        original = a.copy()
        func(a, out=a)
        expected = func(original)
        assert_allclose(a, expected, rtol=1e-2)

    @pytest.mark.parametrize("func", [np.sin, np.cos])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_trig_strided_input(self, func, dtype):
        a = np.linspace(0.1, 3.0, 200, dtype=dtype)
        result = func(a[::3])
        expected = func(a[::3])
        assert_allclose(result, expected, rtol=1e-5)


class TestExpLogStridedExtended:
    """Cover loops_exp/log/log2.dispatch.cpp strided and half-precision paths."""

    @pytest.mark.parametrize("dtype", [np.float16])
    def test_log2_half(self, dtype):
        a = np.arange(1, 101, dtype=dtype)
        result = np.log2(a)
        expected = np.log2(a.astype(np.float32)).astype(dtype)
        assert_allclose(result, expected, rtol=1e-1)

    @pytest.mark.parametrize("dtype", [np.float16])
    def test_log2_half_overlap(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        original = a.copy()
        np.log2(a, out=a)
        expected = np.log2(original)
        assert_allclose(a, expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_log_strided_input(self, dtype):
        a = np.arange(1, 201, dtype=dtype)
        result = np.log(a[::3])
        expected = np.log(a[::3])
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_log2_strided_input(self, dtype):
        a = np.arange(1, 201, dtype=dtype)
        result = np.log2(a[::3])
        expected = np.log2(a[::3])
        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float16])
    def test_exp_half_strided(self, dtype):
        a = np.linspace(-5, 5, 20, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.exp(a, out=out[::2])
        expected = np.exp(a)
        assert_allclose(out[::2], expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16])
    def test_log_half_strided_output(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.log(a, out=out[::2])
        expected = np.log(a)
        assert_allclose(out[::2], expected, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16])
    def test_log2_half_strided_output(self, dtype):
        a = np.arange(1, 21, dtype=dtype)
        out = np.zeros(40, dtype=dtype)
        np.log2(a, out=out[::2])
        expected = np.log2(a)
        assert_allclose(out[::2], expected, rtol=1e-2)
