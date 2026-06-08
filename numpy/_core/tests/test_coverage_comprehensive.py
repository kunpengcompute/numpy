"""
Comprehensive tests to improve C/C++ coverage across multiple numpy modules.
Targets missing lines in:
- numpy/_core/src/multiarray/multiarraymodule.c
- numpy/_core/src/multiarray/arraytypes.c.src
- numpy/_core/src/multiarray/scalartypes.c.src
- numpy/_core/src/multiarray/ctors.c
- numpy/_core/src/multiarray/descriptor.c
- numpy/_core/src/multiarray/dtype_transfer.c
- numpy/_core/src/umath/ufunc_object.c
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises


class TestArrayConcatenation:
    """Test array concatenation to cover multiarraymodule.c paths."""

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64,
                                        np.float32, np.float64, np.complex64, np.complex128])
    def test_concatenate_dtypes(self, dtype):
        """Test concatenation with various dtypes."""
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([4, 5, 6], dtype=dtype)
        result = np.concatenate([a, b])
        assert result.dtype == dtype
        assert len(result) == 6

    def test_concatenate_empty_error(self):
        """Test concatenation with empty list raises error."""
        with pytest.raises(ValueError, match="need at least one array"):
            np.concatenate([])

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_concatenate_axis(self, axis):
        """Test concatenation along different axes."""
        a = np.ones((2, 3))
        b = np.ones((2, 3))
        if axis == 0:
            result = np.concatenate([a, b], axis=axis)
            assert result.shape == (4, 3)
        elif axis == 1 or axis == -1:
            result = np.concatenate([a, b], axis=axis)
            assert result.shape == (2, 6)

    def test_concatenate_mixed_dtypes(self):
        """Test concatenation with mixed dtypes."""
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3.5, 4.5], dtype=np.float64)
        result = np.concatenate([a, b])
        assert result.dtype == np.float64

    def test_concatenate_complex(self):
        """Test concatenation with complex arrays."""
        a = np.array([1+2j, 3+4j], dtype=np.complex128)
        b = np.array([5+6j, 7+8j], dtype=np.complex128)
        result = np.concatenate([a, b])
        assert result.dtype == np.complex128
        assert len(result) == 4


class TestArrayCreation:
    """Test array creation to cover ctors.c paths."""

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64,
                                        np.float16, np.float32, np.float64,
                                        np.complex64, np.complex128,
                                        np.bool_, np.object_])
    def test_array_dtypes(self, dtype):
        """Test array creation with various dtypes."""
        if dtype == np.object_:
            arr = np.array([1, "hello", 3.14], dtype=dtype)
        else:
            arr = np.array([1, 2, 3], dtype=dtype)
        assert arr.dtype == dtype

    def test_array_from_list(self):
        """Test array creation from nested lists."""
        arr = np.array([[1, 2], [3, 4]])
        assert arr.shape == (2, 2)

    def test_array_from_tuple(self):
        """Test array creation from tuples."""
        arr = np.array(((1, 2), (3, 4)))
        assert arr.shape == (2, 2)

    def test_array_fortran_order(self):
        """Test array creation with Fortran order."""
        arr = np.array([[1, 2], [3, 4]], order='F')
        assert arr.flags['F_CONTIGUOUS']

    def test_array_c_order(self):
        """Test array creation with C order."""
        arr = np.array([[1, 2], [3, 4]], order='C')
        assert arr.flags['C_CONTIGUOUS']

    def test_zeros_various_shapes(self):
        """Test zeros with various shapes."""
        for shape in [(0,), (1,), (5,), (2, 3), (2, 3, 4)]:
            arr = np.zeros(shape)
            assert arr.shape == shape
            assert np.all(arr == 0)

    def test_ones_various_shapes(self):
        """Test ones with various shapes."""
        for shape in [(0,), (1,), (5,), (2, 3), (2, 3, 4)]:
            arr = np.ones(shape)
            assert arr.shape == shape
            assert np.all(arr == 1)

    def test_empty_various_shapes(self):
        """Test empty with various shapes."""
        for shape in [(0,), (1,), (5,), (2, 3), (2, 3, 4)]:
            arr = np.empty(shape)
            assert arr.shape == shape

    def test_full_various_shapes(self):
        """Test full with various shapes and fill values."""
        for shape in [(2,), (2, 3), (2, 3, 4)]:
            arr = np.full(shape, 7)
            assert arr.shape == shape
            assert np.all(arr == 7)


class TestDtypeOperations:
    """Test dtype operations to cover descriptor.c paths."""

    def test_dtype_creation(self):
        """Test dtype creation with various specifications."""
        dt = np.dtype('float64')
        assert dt == np.float64

        dt = np.dtype([('x', 'f8'), ('y', 'i4')])
        assert dt.names == ('x', 'y')

    def test_dtype_from_type(self):
        """Test dtype creation from Python types."""
        assert np.dtype(int) == np.int_
        assert np.dtype(float) == np.float64
        assert np.dtype(complex) == np.complex128

    def test_structured_dtype(self):
        """Test structured dtype creation and usage."""
        dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f8')])
        arr = np.array([('Alice', 25, 55.5), ('Bob', 30, 70.2)], dtype=dt)
        assert arr['name'][0] == 'Alice'
        assert arr['age'][1] == 30

    def test_dtype_byteorder(self):
        """Test dtype byte order specifications."""
        dt_native = np.dtype('f8')
        dt_little = np.dtype('<f8')
        dt_big = np.dtype('>f8')
        
        # On little-endian systems, '<' is native
        import sys
        if sys.byteorder == 'little':
            assert dt_little.byteorder in ['<', '=']
            assert dt_big.byteorder == '>'
        else:
            assert dt_little.byteorder == '<'
            assert dt_big.byteorder in ['>', '=']

    def test_dtype_alignment(self):
        """Test dtype alignment."""
        dt = np.dtype([('a', 'i1'), ('b', 'i8')], align=True)
        assert dt.isalignedstruct


class TestTypeCasting:
    """Test type casting to cover dtype_transfer.c paths."""

    @pytest.mark.parametrize("from_dtype,to_dtype", [
        (np.int8, np.int16),
        (np.int16, np.int32),
        (np.int32, np.int64),
        (np.float32, np.float64),
        (np.float64, np.complex128),
        (np.int32, np.float64),
        (np.complex64, np.complex128),
    ])
    def test_astype(self, from_dtype, to_dtype):
        """Test astype with various dtype conversions."""
        arr = np.array([1, 2, 3], dtype=from_dtype)
        result = arr.astype(to_dtype)
        assert result.dtype == to_dtype

    def test_casting_rules(self):
        """Test casting rules."""
        arr_int = np.array([1, 2, 3], dtype=np.int32)
        arr_float = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        
        # Safe casting
        result = arr_int.astype(np.float64, casting='safe')
        assert result.dtype == np.float64

    def test_unsafe_casting(self):
        """Test unsafe casting."""
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        result = arr.astype(np.int32, casting='unsafe')
        assert result.dtype == np.int32
        assert_array_equal(result, [1, 2, 3])


class TestUfuncOperations:
    """Test ufunc operations to cover ufunc_object.c paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_add_dtypes(self, dtype):
        """Test addition with various dtypes."""
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([4, 5, 6], dtype=dtype)
        result = np.add(a, b)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_multiply_dtypes(self, dtype):
        """Test multiplication with various dtypes."""
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([4, 5, 6], dtype=dtype)
        result = np.multiply(a, b)
        assert result.dtype == dtype

    @pytest.mark.parametrize("ufunc", [np.sin, np.cos, np.exp, np.log, np.sqrt])
    def test_math_ufuncs(self, ufunc):
        """Test various math ufuncs."""
        a = np.array([0.1, 0.5, 1.0, 2.0])
        result = ufunc(a)
        assert result.shape == a.shape

    @pytest.mark.parametrize("ufunc", [np.sin, np.cos, np.exp, np.log, np.sqrt])
    def test_math_ufuncs_complex(self, ufunc):
        """Test math ufuncs with complex numbers."""
        a = np.array([0.1+0.1j, 0.5+0.5j, 1.0+1.0j])
        result = ufunc(a)
        assert result.dtype == np.complex128

    def test_ufunc_out_parameter(self):
        """Test ufunc with out parameter."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        out = np.empty(3, dtype=np.int64)
        result = np.add(a, b, out=out)
        assert result is out
        assert_array_equal(out, [5, 7, 9])

    def test_ufunc_where_parameter(self):
        """Test ufunc with where parameter."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        where = np.array([True, False, True])
        out = np.zeros(3, dtype=np.int64)
        result = np.add(a, b, out=out, where=where)
        assert_array_equal(result, [5, 0, 9])

    def test_ufunc_reduce(self):
        """Test ufunc reduce."""
        a = np.array([1, 2, 3, 4, 5])
        result = np.add.reduce(a)
        assert result == 15

    def test_ufunc_accumulate(self):
        """Test ufunc accumulate."""
        a = np.array([1, 2, 3, 4, 5])
        result = np.add.accumulate(a)
        assert_array_equal(result, [1, 3, 6, 10, 15])

    def test_ufunc_outer(self):
        """Test ufunc outer."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        result = np.multiply.outer(a, b)
        assert result.shape == (3, 2)

    def test_ufunc_at(self):
        """Test ufunc at."""
        a = np.array([1, 2, 3, 4, 5])
        indices = np.array([0, 2, 4])
        np.add.at(a, indices, 10)
        assert_array_equal(a, [11, 2, 13, 4, 15])


class TestArrayIndexing:
    """Test array indexing to cover various code paths."""

    def test_basic_indexing(self):
        """Test basic indexing."""
        a = np.arange(10)
        assert a[5] == 5
        assert a[-1] == 9

    def test_slice_indexing(self):
        """Test slice indexing."""
        a = np.arange(10)
        assert_array_equal(a[2:5], [2, 3, 4])
        assert_array_equal(a[::2], [0, 2, 4, 6, 8])

    def test_fancy_indexing(self):
        """Test fancy indexing."""
        a = np.arange(10)
        indices = np.array([1, 3, 5, 7])
        assert_array_equal(a[indices], [1, 3, 5, 7])

    def test_boolean_indexing(self):
        """Test boolean indexing."""
        a = np.arange(10)
        mask = a > 5
        assert_array_equal(a[mask], [6, 7, 8, 9])

    def test_multidim_indexing(self):
        """Test multidimensional indexing."""
        a = np.arange(12).reshape(3, 4)
        assert a[1, 2] == 6
        assert_array_equal(a[1, :], [4, 5, 6, 7])
        assert_array_equal(a[:, 2], [2, 6, 10])


class TestArrayReshaping:
    """Test array reshaping operations."""

    def test_reshape(self):
        """Test reshape."""
        a = np.arange(12)
        b = a.reshape(3, 4)
        assert b.shape == (3, 4)

    def test_transpose(self):
        """Test transpose."""
        a = np.arange(12).reshape(3, 4)
        b = a.T
        assert b.shape == (4, 3)

    def test_ravel(self):
        """Test ravel."""
        a = np.arange(12).reshape(3, 4)
        b = a.ravel()
        assert b.shape == (12,)

    def test_flatten(self):
        """Test flatten."""
        a = np.arange(12).reshape(3, 4)
        b = a.flatten()
        assert b.shape == (12,)

    def test_squeeze(self):
        """Test squeeze."""
        a = np.arange(12).reshape(1, 3, 1, 4)
        b = np.squeeze(a)
        assert b.shape == (3, 4)

    def test_expand_dims(self):
        """Test expand_dims."""
        a = np.arange(12).reshape(3, 4)
        b = np.expand_dims(a, axis=0)
        assert b.shape == (1, 3, 4)


class TestArrayMath:
    """Test array math operations."""

    def test_sum(self):
        """Test sum."""
        a = np.arange(12).reshape(3, 4)
        assert a.sum() == 66
        assert_array_equal(a.sum(axis=0), [12, 15, 18, 21])
        assert_array_equal(a.sum(axis=1), [6, 22, 38])

    def test_mean(self):
        """Test mean."""
        a = np.arange(12).reshape(3, 4)
        assert a.mean() == 5.5

    def test_std(self):
        """Test std."""
        a = np.arange(12).reshape(3, 4)
        assert a.std() > 0

    def test_min_max(self):
        """Test min and max."""
        a = np.arange(12).reshape(3, 4)
        assert a.min() == 0
        assert a.max() == 11

    def test_argmin_argmax(self):
        """Test argmin and argmax."""
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        assert a.argmin() == 1
        assert a.argmax() == 5

    def test_cumsum(self):
        """Test cumsum."""
        a = np.array([1, 2, 3, 4])
        assert_array_equal(a.cumsum(), [1, 3, 6, 10])

    def test_prod(self):
        """Test prod."""
        a = np.array([1, 2, 3, 4])
        assert a.prod() == 24


class TestSorting:
    """Test sorting operations."""

    @pytest.mark.parametrize("dtype", [np.int32, np.float64, np.complex128])
    def test_sort(self, dtype):
        """Test sort with various dtypes."""
        if dtype == np.complex128:
            a = np.array([3+1j, 1+2j, 4+0j, 1+5j], dtype=dtype)
        else:
            a = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=dtype)
        result = np.sort(a)
        assert np.all(np.diff(result.real) >= 0)

    def test_argsort(self):
        """Test argsort."""
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        indices = np.argsort(a)
        assert_array_equal(a[indices], np.sort(a))

    def test_sort_axis(self):
        """Test sort along different axes."""
        a = np.array([[3, 1, 4], [1, 5, 9]])
        result = np.sort(a, axis=0)
        assert_array_equal(result[0], [1, 1, 4])
        assert_array_equal(result[1], [3, 5, 9])


class TestSearchSorted:
    """Test searchsorted operations."""

    def test_searchsorted_basic(self):
        """Test basic searchsorted."""
        a = np.array([1, 2, 3, 4, 5])
        assert np.searchsorted(a, 3) == 2

    def test_searchsorted_side(self):
        """Test searchsorted with different sides."""
        a = np.array([1, 2, 2, 2, 3, 4, 5])
        assert np.searchsorted(a, 2, side='left') == 1
        assert np.searchsorted(a, 2, side='right') == 4

    def test_searchsorted_multiple(self):
        """Test searchsorted with multiple values."""
        a = np.array([1, 2, 3, 4, 5])
        v = np.array([2, 4])
        result = np.searchsorted(a, v)
        assert_array_equal(result, [1, 3])


class TestLinearAlgebra:
    """Test linear algebra operations."""

    def test_dot(self):
        """Test dot product."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        assert np.dot(a, b) == 32

    def test_matmul(self):
        """Test matrix multiplication."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = np.matmul(a, b)
        assert_array_equal(result, [[19, 22], [43, 50]])

    def test_inner(self):
        """Test inner product."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        assert np.inner(a, b) == 32

    def test_outer(self):
        """Test outer product."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        result = np.outer(a, b)
        assert result.shape == (3, 2)

    def test_cross(self):
        """Test cross product."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.cross(a, b)
        assert_array_equal(result, [-3, 6, -3])


class TestFFT:
    """Test FFT operations."""

    def test_fft(self):
        """Test FFT."""
        a = np.array([1, 2, 3, 4])
        result = np.fft.fft(a)
        assert result.shape == (4,)
        assert result.dtype == np.complex128

    def test_ifft(self):
        """Test inverse FFT."""
        a = np.array([1, 2, 3, 4])
        fft_result = np.fft.fft(a)
        ifft_result = np.fft.ifft(fft_result)
        assert_allclose(ifft_result.real, a, rtol=1e-10)

    def test_fft2(self):
        """Test 2D FFT."""
        a = np.arange(16).reshape(4, 4)
        result = np.fft.fft2(a)
        assert result.shape == (4, 4)

    def test_rfft(self):
        """Test real FFT."""
        a = np.array([1, 2, 3, 4])
        result = np.fft.rfft(a)
        assert result.shape == (3,)


class TestRandom:
    """Test random number generation."""

    def test_random_uniform(self):
        """Test uniform random."""
        result = np.random.uniform(0, 1, 100)
        assert result.shape == (100,)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_random_normal(self):
        """Test normal random."""
        result = np.random.normal(0, 1, 1000)
        assert result.shape == (1000,)
        assert abs(result.mean()) < 0.2

    def test_random_integers(self):
        """Test random integers."""
        result = np.random.randint(0, 10, 100)
        assert result.shape == (100,)
        assert np.all(result >= 0) and np.all(result < 10)

    def test_random_choice(self):
        """Test random choice."""
        a = np.arange(10)
        result = np.random.choice(a, 5, replace=False)
        assert len(result) == 5
        assert len(np.unique(result)) == 5

    def test_random_shuffle(self):
        """Test random shuffle."""
        a = np.arange(10)
        original = a.copy()
        np.random.shuffle(a)
        assert len(a) == 10
        assert set(a) == set(original)


class TestStringOperations:
    """Test string array operations."""

    def test_string_array(self):
        """Test string array creation."""
        arr = np.array(['hello', 'world', 'numpy'])
        assert arr.dtype.kind == 'U'
        assert len(arr) == 3

    def test_string_concatenation(self):
        """Test string concatenation."""
        a = np.array(['hello', 'world'])
        b = np.array([' there', ' numpy'])
        result = np.char.add(a, b)
        assert_array_equal(result, ['hello there', 'world numpy'])

    def test_string_upper(self):
        """Test string upper."""
        arr = np.array(['hello', 'world'])
        result = np.char.upper(arr)
        assert_array_equal(result, ['HELLO', 'WORLD'])

    def test_string_lower(self):
        """Test string lower."""
        arr = np.array(['HELLO', 'WORLD'])
        result = np.char.lower(arr)
        assert_array_equal(result, ['hello', 'world'])


class TestDatetimeOperations:
    """Test datetime operations."""

    def test_datetime_creation(self):
        """Test datetime creation."""
        dt = np.datetime64('2023-01-01')
        assert str(dt) == '2023-01-01'

    def test_datetime_arithmetic(self):
        """Test datetime arithmetic."""
        dt1 = np.datetime64('2023-01-01')
        dt2 = np.datetime64('2023-01-10')
        delta = dt2 - dt1
        assert delta == np.timedelta64(9, 'D')

    def test_datetime_array(self):
        """Test datetime array."""
        dates = np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[D]')
        assert dates.dtype == np.dtype('datetime64[D]')
        assert len(dates) == 3


class TestMaskedArrays:
    """Test masked array operations."""

    def test_masked_array_creation(self):
        """Test masked array creation."""
        a = np.ma.array([1, 2, 3, 4], mask=[False, True, False, True])
        assert a[1] is np.ma.masked
        assert a[0] == 1

    def test_masked_array_operations(self):
        """Test masked array operations."""
        a = np.ma.array([1, 2, 3, 4], mask=[False, True, False, True])
        assert a.sum() == 4  # 1 + 3
        assert a.mean() == 2.0  # (1 + 3) / 2

    def test_masked_array_fill(self):
        """Test masked array fill value."""
        a = np.ma.array([1, 2, 3, 4], mask=[False, True, False, True], fill_value=-999)
        filled = a.filled()
        assert_array_equal(filled, [1, -999, 3, -999])
