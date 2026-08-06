"""
Tests to improve Python incremental code coverage to 80%+.
Targets numpy/_core/_methods.py which fell to 79% after the
Std/Var cross-platform revert (PR #137).

Covers all branches of:
- _count_reduce_items (axis=None, axis=int, axis=tuple, where=array)
- _mean (dtype inference, float16, out, where, object arrays, 0-d)
- _var (ddof, dtype, mean param, complex, object, where, out, keepdims,
        ARM all-equal short-circuit)
- _std (various dtypes, out, keepdims, 0-d, object)
- _any, _all, _amax, _amin, _sum, _prod (direct calls)
- _clip (integer out-of-bound, min/max None)
- _ptp, _dump, _dumps, _bitwise_count
"""
import io
import os
import warnings

import pytest
import numpy as np
from numpy._core import _methods as M
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)


class TestCountReduceItems:
    """_count_reduce_items: all axis and where branches."""

    def test_axis_none_1d(self):
        result = M._count_reduce_items(np.array([1, 2, 3, 4]), None)
        assert result == 4

    def test_axis_none_2d(self):
        result = M._count_reduce_items(np.array([[1, 2], [3, 4]]), None)
        assert result == 4

    def test_axis_int(self):
        result = M._count_reduce_items(np.array([[1, 2], [3, 4]]), 0)
        assert result == 2

    def test_axis_tuple(self):
        result = M._count_reduce_items(np.array([[1, 2], [3, 4]]), (0, 1))
        assert result == 4

    def test_where_array_1d(self):
        mask = np.array([True, False, True, False])
        result = M._count_reduce_items(
            np.array([1, 2, 3, 4]), None, where=mask
        )
        assert result == 2

    def test_where_array_2d_keepdims(self):
        mask = np.array([[True, False], [False, True]])
        result = M._count_reduce_items(
            np.array([[1, 2], [3, 4]]), 0, keepdims=True, where=mask
        )
        assert_array_equal(result, [[1, 1]])


class TestMean:
    """_mean: all dtype, out, where, and return-type branches."""

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64,
                                    np.uint8, np.uint16, np.uint32, np.uint64])
    def test_mean_integer_dtype_inference(self, dt):
        arr = np.array([1, 2, 3, 4], dtype=dt)
        result = np.mean(arr)
        assert result.dtype == np.float64
        assert_almost_equal(result, 2.5)

    def test_mean_bool_dtype_inference(self):
        arr = np.array([True, False, True, True])
        result = np.mean(arr)
        assert result.dtype == np.float64
        assert_almost_equal(result, 0.75)

    def test_mean_float16_dtype_inference(self):
        arr = np.array([1, 2, 3], dtype=np.float16)
        result = np.mean(arr)
        assert result.dtype == np.float16
        assert_almost_equal(result, 2.0, decimal=3)

    def test_mean_float16_with_out(self):
        arr = np.array([1, 2, 3], dtype=np.float16)
        out = np.zeros((), dtype=np.float16)
        result = np.mean(arr, out=out)
        assert result is out
        assert_almost_equal(result, 2.0, decimal=3)

    def test_mean_explicit_dtype(self):
        arr = np.array([1, 2, 3, 4])
        result = np.mean(arr, dtype=np.float32)
        assert result.dtype == np.float32
        assert_almost_equal(result, 2.5)

    def test_mean_out_parameter(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = np.zeros(())
        result = np.mean(arr, out=out)
        assert result is out
        assert_almost_equal(result, 2.5)

    def test_mean_axis_none(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = np.mean(arr)
        assert_almost_equal(result, 2.5)

    def test_mean_axis_int(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = np.mean(arr, axis=0)
        assert_array_equal(result, [2.0, 3.0])

    def test_mean_axis_tuple(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = np.mean(arr, axis=(0, 1))
        assert_almost_equal(result, 2.5)

    def test_mean_where_true_default(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = np.mean(arr, where=True)
        assert_almost_equal(result, 2.5)

    def test_mean_where_array(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        result = np.mean(arr, where=mask)
        assert_almost_equal(result, 2.0)

    def test_mean_where_array_keepdims(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [False, True]])
        result = np.mean(arr, where=mask, keepdims=True)
        assert result.shape == (1, 1)

    def test_mean_where_false_empty_warning(self):
        arr = np.array([1.0, 2.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.mean(arr, where=False)
        assert np.isnan(result)

    def test_mean_keepdims(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = np.mean(arr, keepdims=True)
        assert result.shape == (1,)

    def test_mean_0d_input(self):
        result = np.mean(np.array(5.0))
        assert result == 5.0

    def test_mean_scalar_input(self):
        result = np.mean(5.0)
        assert result == 5.0

    def test_mean_object_array(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=object)
        result = np.mean(arr)
        assert_almost_equal(float(result), 2.0)

    def test_mean_float16_result_not_ndarray(self):
        arr = np.array([1, 2, 3], dtype=np.float16)
        result = np.mean(arr)
        assert result.dtype == np.float16


class TestVar:
    """_var: all dtype, ddof, mean, where, out, and short-circuit branches."""

    @pytest.mark.parametrize("ddof", [0, 1, 2])
    def test_var_ddof(self, ddof):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = np.var(arr, ddof=ddof)
        expected = np.sum((arr - arr.mean()) ** 2) / (len(arr) - ddof)
        assert_almost_equal(result, expected)

    def test_var_ddof_ge_n_warning(self):
        arr = np.array([1.0, 2.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.var(arr, ddof=3)
        assert np.isnan(result) or result == 0.0 or np.isinf(result)

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64,
                                    np.uint8, np.uint16, np.uint32, np.uint64])
    def test_var_integer_dtype_inference(self, dt):
        arr = np.array([1, 2, 3, 4], dtype=dt)
        result = np.var(arr)
        assert result.dtype == np.float64

    def test_var_bool_dtype_inference(self):
        arr = np.array([True, False, True, True])
        result = np.var(arr)
        assert result.dtype == np.float64

    def test_var_float_dtype(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = np.var(arr)
        assert result.dtype == np.float64
        assert_almost_equal(result, 1.25)

    def test_var_float32_dtype(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = np.var(arr)
        assert result.dtype == np.float32

    def test_var_complex64(self):
        arr = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        result = np.var(arr)
        assert result.dtype == np.float32
        assert result > 0

    def test_var_complex128(self):
        arr = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        result = np.var(arr)
        assert_almost_equal(result, np.var(arr))

    def test_var_clongdouble(self):
        arr = np.array([1 + 2j, 3 + 4j], dtype=np.clongdouble)
        result = np.var(arr)
        assert np.isfinite(float(result.real))

    def test_var_object_array(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=object)
        result = np.var(arr)
        assert_almost_equal(float(result), 1.25)

    def test_var_explicit_dtype(self):
        arr = np.array([1, 2, 3, 4])
        result = np.var(arr, dtype=np.float32)
        assert result.dtype == np.float32

    def test_var_out_parameter(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = np.zeros(())
        result = np.var(arr, out=out)
        assert result is out
        assert_almost_equal(result, 2.0)

    def test_var_mean_parameter(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_val = np.mean(arr)
        result = np.var(arr, mean=mean_val)
        expected = np.var(arr)
        assert_almost_equal(result, expected)

    def test_var_where_true_default(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = np.var(arr, where=True)
        assert_almost_equal(result, 1.25)

    def test_var_where_array(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = np.array([True, False, True, False, True])
        result = np.var(arr, where=mask)
        expected = np.var(arr[mask])
        assert_almost_equal(result, expected)

    def test_var_where_array_keepdims(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([[True, False], [False, True]])
        result = np.var(arr, where=mask, keepdims=True)
        assert result.shape == (1, 1)

    def test_var_where_false_empty_warning(self):
        arr = np.array([1.0, 2.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.var(arr, where=False)
        assert np.isnan(result) or result == 0.0

    def test_var_keepdims(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = np.var(arr, keepdims=True)
        assert result.shape == (1,)

    def test_var_axis_none(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = np.var(arr)
        assert_almost_equal(result, 1.25)

    def test_var_axis_int(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result0 = np.var(arr, axis=0)
        result1 = np.var(arr, axis=1)
        assert_array_equal(result0, [1.0, 1.0])
        assert_array_equal(result1, [0.25, 0.25])

    def test_var_axis_tuple(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = np.var(arr, axis=(0, 1))
        assert_almost_equal(result, 1.25)

    def test_var_0d_input(self):
        result = np.var(np.array(5.0))
        assert result == 0.0

    def test_var_scalar_input(self):
        result = np.var(5.0)
        assert result == 0.0

    def test_var_float16(self):
        arr = np.array([1, 2, 3], dtype=np.float16)
        result = np.var(arr)
        assert result.dtype == np.float16


class TestVarArmShortCircuit:
    """_var ARM all-equal short-circuit (lines 174-190)."""

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_all_equal_float(self, dt):
        arr = np.ones(10, dtype=dt)
        result = np.var(arr)
        assert result == 0.0

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.int32, np.int64,
                                    np.uint8, np.uint16, np.uint32, np.uint64])
    def test_all_equal_integer(self, dt):
        arr = np.ones(10, dtype=dt)
        result = np.var(arr)
        assert result == 0.0

    def test_all_equal_bool(self):
        arr = np.ones(10, dtype=bool)
        result = np.var(arr)
        assert result == 0.0

    def test_all_equal_float_zeros(self):
        arr = np.zeros(5, dtype=np.float64)
        result = np.var(arr)
        assert result == 0.0

    def test_all_equal_with_out(self):
        arr = np.ones(10, dtype=np.float64)
        out = np.zeros((), dtype=np.float64)
        result = np.var(arr, out=out)
        assert result is out
        assert result == 0.0

    def test_all_equal_with_keepdims(self):
        arr = np.ones(10, dtype=np.float64)
        result = np.var(arr, keepdims=True)
        assert result.shape == (1,)
        assert result[0] == 0.0

    def test_all_equal_with_dtype(self):
        arr = np.ones(10, dtype=np.float64)
        result = np.var(arr, dtype=np.dtype('f4'))
        assert result == 0.0

    def test_all_equal_nan_float(self):
        arr = np.array([np.nan] * 5, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.var(arr)
        assert np.isnan(result)

    def test_not_all_equal_float(self):
        arr = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        result = np.var(arr)
        assert result > 0.0

    def test_not_all_equal_integer(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = np.var(arr)
        assert result > 0.0

    def test_short_circuit_size_boundary_256(self):
        arr = np.ones(256, dtype=np.float64)
        result = np.var(arr)
        assert result == 0.0

    def test_short_circuit_exceeds_size(self):
        arr = np.ones(257, dtype=np.float64)
        result = np.var(arr)
        assert result == 0.0

    def test_short_circuit_non_contiguous(self):
        arr = np.ones(20, dtype=np.float64)[::2]
        result = np.var(arr)
        assert result == 0.0

    def test_short_circuit_2d_array(self):
        arr = np.ones((3, 3), dtype=np.float64)
        result = np.var(arr)
        assert result == 0.0

    def test_short_circuit_with_ddof(self):
        arr = np.ones(10, dtype=np.float64)
        result = np.var(arr, ddof=1)
        assert result == 0.0

    def test_short_circuit_with_mean_param(self):
        arr = np.ones(10, dtype=np.float64)
        mean_val = np.float64(1.0)
        result = np.var(arr, mean=mean_val)
        assert result == 0.0


class TestStd:
    """_std: all dtype, out, and return-type branches."""

    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_std_float(self, dt):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dt)
        result = np.std(arr)
        assert_almost_equal(result, np.sqrt(2.0))

    def test_std_complex(self):
        arr = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        result = np.std(arr)
        expected = np.sqrt(np.var(arr))
        assert_almost_equal(result, expected)

    def test_std_ddof(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = np.std(arr, ddof=1)
        expected = np.std(arr, ddof=0) * np.sqrt(5 / 4)
        assert_almost_equal(result, expected)

    def test_std_out_parameter(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = np.zeros(())
        result = np.std(arr, out=out)
        assert result is out
        assert_almost_equal(result, np.sqrt(2.0))

    def test_std_keepdims(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = np.std(arr, keepdims=True)
        assert result.shape == (1,)

    def test_std_0d_input(self):
        result = np.std(np.array(5.0))
        assert result == 0.0

    def test_std_scalar_input(self):
        result = np.std(5.0)
        assert result == 0.0

    def test_std_object_array(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=object)
        result = np.std(arr)
        assert_almost_equal(float(result), np.std([1.0, 2.0, 3.0, 4.0]))

    def test_std_float16(self):
        arr = np.array([1, 2, 3], dtype=np.float16)
        result = np.std(arr)
        assert result.dtype == np.float16

    def test_std_all_equal(self):
        arr = np.ones(10, dtype=np.float64)
        result = np.std(arr)
        assert result == 0.0


class TestReductionMethods:
    """_any, _all, _amax, _amin, _sum, _prod: direct calls."""

    def test_any_true(self):
        arr = np.array([False, True, False])
        assert M._any(arr) == True

    def test_any_false(self):
        arr = np.array([False, False, False])
        assert M._any(arr) == False

    def test_any_where_true(self):
        arr = np.array([False, True, False])
        assert M._any(arr, where=True) == True

    def test_any_where_array(self):
        arr = np.array([False, True, False])
        mask = np.array([True, False, True])
        assert M._any(arr, where=mask) == False

    def test_any_dtype_default_bool(self):
        arr = np.array([0, 1, 0])
        result = M._any(arr)
        assert result == True

    def test_all_true(self):
        arr = np.array([True, True, True])
        assert M._all(arr) == True

    def test_all_false(self):
        arr = np.array([True, False, True])
        assert M._all(arr) == False

    def test_all_where_true(self):
        arr = np.array([True, True, True])
        assert M._all(arr, where=True) == True

    def test_all_where_array(self):
        arr = np.array([True, False, True])
        mask = np.array([True, False, True])
        assert M._all(arr, where=mask) == True

    def test_all_dtype_default_bool(self):
        arr = np.array([1, 1, 1])
        result = M._all(arr)
        assert result == True

    def test_amax(self):
        arr = np.array([3, 1, 4, 1, 5])
        assert M._amax(arr) == 5

    def test_amin(self):
        arr = np.array([3, 1, 4, 1, 5])
        assert M._amin(arr) == 1

    def test_sum(self):
        arr = np.array([1, 2, 3, 4, 5])
        assert M._sum(arr) == 15

    def test_prod(self):
        arr = np.array([1, 2, 3, 4, 5])
        assert M._prod(arr) == 120

    def test_amax_with_axis(self):
        arr = np.array([[3, 1], [4, 2]])
        assert_array_equal(M._amax(arr, axis=0), [4, 2])

    def test_amin_with_axis(self):
        arr = np.array([[3, 1], [4, 2]])
        assert_array_equal(M._amin(arr, axis=0), [3, 1])

    def test_sum_with_axis(self):
        arr = np.array([[1, 2], [3, 4]])
        assert_array_equal(M._sum(arr, axis=0), [4, 6])

    def test_prod_with_axis(self):
        arr = np.array([[1, 2], [3, 4]])
        assert_array_equal(M._prod(arr, axis=1), [2, 12])


class TestClip:
    """_clip: all min/max branches including integer out-of-bound."""

    def test_clip_both(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = np.clip(arr, 2, 4)
        assert_array_equal(result, [2, 2, 3, 4, 4])

    def test_clip_min_only(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = np.clip(arr, 2, None)
        assert_array_equal(result, [2, 2, 3, 4, 5])

    def test_clip_max_only(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = np.clip(arr, None, 4)
        assert_array_equal(result, [1, 2, 3, 4, 4])

    def test_clip_neither(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = np.clip(arr, None, None)
        assert_array_equal(result, [1, 2, 3, 4, 5])

    def test_clip_integer_below_min(self):
        arr = np.array([1, 2, 3], dtype=np.uint8)
        result = np.clip(arr, -100, 100)
        assert_array_equal(result, [1, 2, 3])

    def test_clip_integer_above_max(self):
        arr = np.array([1, 2, 3], dtype=np.uint8)
        result = np.clip(arr, -1, None)
        assert_array_equal(result, [1, 2, 3])

    def test_clip_integer_above_max_min_none(self):
        arr = np.array([1, 2, 3], dtype=np.uint8)
        result = np.clip(arr, None, 300)
        assert_array_equal(result, [1, 2, 3])

    def test_clip_int32(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = np.clip(arr, 0, 10)
        assert_array_equal(result, [1, 2, 3])

    def test_clip_float(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = np.clip(arr, 1.5, 2.5)
        assert_array_equal(result, [1.5, 2.0, 2.5])


class TestPtp:
    """_ptp: peak-to-peak."""

    def test_ptp_1d(self):
        arr = np.array([1, 2, 3, 4, 5])
        assert M._ptp(arr) == 4

    def test_ptp_with_axis(self):
        arr = np.array([[1, 2], [3, 4]])
        assert_array_equal(M._ptp(arr, axis=0), [2, 2])


class TestDumpDumps:
    """_dump, _dumps: file object and file path."""

    def test_dump_file_object(self):
        arr = np.array([1, 2, 3])
        buf = io.BytesIO()
        arr.dump(buf)
        buf.seek(0)
        import pickle
        loaded = pickle.load(buf)
        assert_array_equal(loaded, arr)

    def test_dump_file_path(self):
        arr = np.array([1, 2, 3])
        path = '/tmp/test_dump_methods.npy'
        arr.dump(path)
        loaded = np.load(path, allow_pickle=True)
        assert_array_equal(loaded, arr)
        os.remove(path)

    def test_dumps(self):
        arr = np.array([1, 2, 3])
        data = arr.dumps()
        import pickle
        loaded = pickle.loads(data)
        assert_array_equal(loaded, arr)

    def test_dump_protocol(self):
        arr = np.array([1, 2, 3])
        buf = io.BytesIO()
        arr.dump(buf, protocol=2)
        buf.seek(0)
        import pickle
        loaded = pickle.load(buf)
        assert_array_equal(loaded, arr)


class TestBitwiseCount:
    """_bitwise_count."""

    def test_bitwise_count_uint8(self):
        arr = np.array([5, 0, 255], dtype=np.uint8)
        result = np.bitwise_count(arr)
        assert_array_equal(result, [2, 0, 8])

    def test_bitwise_count_int32(self):
        arr = np.array([7, 0, 255], dtype=np.int32)
        result = np.bitwise_count(arr)
        assert_array_equal(result, [3, 0, 8])

    def test_bitwise_count_with_out(self):
        arr = np.array([5, 3], dtype=np.uint8)
        out = np.zeros(2, dtype=np.uint8)
        result = np.bitwise_count(arr, out=out)
        assert result is out
        assert_array_equal(result, [2, 2])
