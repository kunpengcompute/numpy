"""950 platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_ufunc import (
    Broadcast as _Official_bench_ufunc_Broadcast,
    CustomComparison as _Official_bench_ufunc_CustomComparison,
    CustomScalarFloorDivideInt as _Official_bench_ufunc_CustomScalarFloorDivideInt,
    CustomScalarFloorDivideUInt as _Official_bench_ufunc_CustomScalarFloorDivideUInt,
    MethodsV0 as _Official_bench_ufunc_MethodsV0,
    MethodsV1 as _Official_bench_ufunc_MethodsV1,
    MethodsV1IntOnly as _Official_bench_ufunc_MethodsV1IntOnly,
    NDArrayGetItem as _Official_bench_ufunc_NDArrayGetItem,
    NDArrayLRShifts as _Official_bench_ufunc_NDArrayLRShifts,
    NDArraySetItem as _Official_bench_ufunc_NDArraySetItem,
    UFunc as _Official_bench_ufunc_UFunc,
)
from ..bench_ufunc_strides import (
    BinaryFP as _Official_bench_ufunc_strides_BinaryFP,
    BinaryInt as _Official_bench_ufunc_strides_BinaryInt,
    BinaryIntContig as _Official_bench_ufunc_strides_BinaryIntContig,
    UnaryFP as _Official_bench_ufunc_strides_UnaryFP,
    UnaryFPSpecial as _Official_bench_ufunc_strides_UnaryFPSpecial,
)


class Abs(_AggregateBenchmark):
    """Aggregate abs with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (0, 3)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__abs__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (0, 5)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__abs__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (0, 0)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__abs__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (0, 2)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__abs__', 'int32')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (1,)),  # bench_ufunc.UFunc.time_ufunc_types('absolute')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (30,)),  # bench_ufunc.UFunc.time_ufunc_types('fabs')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (0, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (17, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (0, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (17, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 2, 'f')
    )
    run_repeat = (
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__abs__', 'float32')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__abs__', 'float64')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__abs__', 'int16')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__abs__', 'int32')
        15,  # bench_ufunc.UFunc.time_ufunc_types('absolute')
        10,  # bench_ufunc.UFunc.time_ufunc_types('fabs')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 1, 'd')
        14,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 2, 'd')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 1, 'd')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 2, 'd')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'absolute'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'fabs'>, 4, 2, 'f')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 1, 'd')
        14,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 2, 'd')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 1, 'd')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 2, 'd')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'absolute'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'fabs'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ufunc_types',
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Add(_AggregateBenchmark):
    """Aggregate add with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (0, 3)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__add__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (0, 5)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__add__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (0, 0)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__add__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (0, 2)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__add__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (0, 4)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__add__', 'int64')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (0, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (0, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (0, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (0, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (0, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (0, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (0, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (0, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (0, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (0, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (0, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (0, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (0, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (0, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (0, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (0, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (0, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (0, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (0, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (0, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'q')
    )
    run_repeat = (
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__add__', 'float32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__add__', 'float64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__add__', 'int16')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__add__', 'int32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__add__', 'int64')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'B')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'H')
        556,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'I')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'L')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'Q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'b')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'h')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'i')
        358,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'l')
        358,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'add'>, 1, 1, 1, 'q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'B')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'H')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'I')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'L')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'Q')
        667,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'b')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'h')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'i')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'l')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'add'>, 1, 1, 1, 'q')
    )
    case_methods = (
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
    )
    case_types = (
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
    )


class Bitwise(_AggregateBenchmark):
    """Aggregate bitwise with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_MethodsV1IntOnly, 'time_ndarray_meth', (0, 0)),  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__and__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1IntOnly, 'time_ndarray_meth', (0, 1)),  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__and__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1IntOnly, 'time_ndarray_meth', (0, 2)),  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__and__', 'int64')
        _select_case_params(_Official_bench_ufunc_MethodsV1IntOnly, 'time_ndarray_meth', (1, 0)),  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__or__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1IntOnly, 'time_ndarray_meth', (1, 1)),  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__or__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1IntOnly, 'time_ndarray_meth', (1, 2)),  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__or__', 'int64')
        _select_case_params(_Official_bench_ufunc_MethodsV1IntOnly, 'time_ndarray_meth', (2, 0)),  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__xor__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1IntOnly, 'time_ndarray_meth', (2, 1)),  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__xor__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1IntOnly, 'time_ndarray_meth', (2, 2)),  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__xor__', 'int64')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (10,)),  # bench_ufunc.UFunc.time_ufunc_types('bitwise_and')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (12,)),  # bench_ufunc.UFunc.time_ufunc_types('bitwise_not')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (13,)),  # bench_ufunc.UFunc.time_ufunc_types('bitwise_or')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (14,)),  # bench_ufunc.UFunc.time_ufunc_types('bitwise_xor')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (3, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (3, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (3, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (3, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (3, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (3, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (3, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (3, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (3, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (3, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (4, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (4, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (4, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (4, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (4, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (4, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (4, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (4, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (4, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (4, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (5, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (5, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (5, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (5, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (5, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (5, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (5, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (5, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (5, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (5, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (3, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (3, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (3, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (3, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (3, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (3, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (3, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (3, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (3, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (3, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (4, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (4, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (4, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (4, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (4, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (4, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (4, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (4, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (4, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (4, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (5, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (5, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (5, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (5, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (5, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (5, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (5, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (5, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (5, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (5, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'q')
    )
    run_repeat = (
        1000,  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__and__', 'int16')
        1000,  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__and__', 'int32')
        1000,  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__and__', 'int64')
        1000,  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__or__', 'int16')
        1000,  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__or__', 'int32')
        1000,  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__or__', 'int64')
        1000,  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__xor__', 'int16')
        1000,  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__xor__', 'int32')
        1000,  # bench_ufunc.MethodsV1IntOnly.time_ndarray_meth('__xor__', 'int64')
        70,  # bench_ufunc.UFunc.time_ufunc_types('bitwise_and')
        98,  # bench_ufunc.UFunc.time_ufunc_types('bitwise_not')
        70,  # bench_ufunc.UFunc.time_ufunc_types('bitwise_or')
        70,  # bench_ufunc.UFunc.time_ufunc_types('bitwise_xor')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'B')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'H')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'I')
        358,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'L')
        358,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'Q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'b')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'h')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'i')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'l')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_and'>, 1, 1, 1, 'q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'B')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'H')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'I')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'L')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'Q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'b')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'h')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'i')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'l')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_or'>, 1, 1, 1, 'q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'B')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'H')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'I')
        218,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'L')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'Q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'b')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'h')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'i')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'l')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'bitwise_xor'>, 1, 1, 1, 'q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'B')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'H')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'I')
        323,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'L')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'Q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'b')
        625,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'h')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'i')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'l')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_and'>, 1, 1, 1, 'q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'B')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'H')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'I')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'L')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'Q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'b')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'h')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'i')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'l')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_or'>, 1, 1, 1, 'q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'B')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'H')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'I')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'L')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'Q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'b')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'h')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'i')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'l')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'bitwise_xor'>, 1, 1, 1, 'q')
    )
    case_methods = (
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ufunc_types',
        'time_ufunc_types',
        'time_ufunc_types',
        'time_ufunc_types',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
    )
    case_types = (
        _Official_bench_ufunc_MethodsV1IntOnly,
        _Official_bench_ufunc_MethodsV1IntOnly,
        _Official_bench_ufunc_MethodsV1IntOnly,
        _Official_bench_ufunc_MethodsV1IntOnly,
        _Official_bench_ufunc_MethodsV1IntOnly,
        _Official_bench_ufunc_MethodsV1IntOnly,
        _Official_bench_ufunc_MethodsV1IntOnly,
        _Official_bench_ufunc_MethodsV1IntOnly,
        _Official_bench_ufunc_MethodsV1IntOnly,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
    )


class BitwiseCount(_AggregateBenchmark):
    """Aggregate bitwise_count with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (11,)),  # bench_ufunc.UFunc.time_ufunc_types('bitwise_count')
    )
    run_repeat = (
        1,  # bench_ufunc.UFunc.time_ufunc_types('bitwise_count')
    )
    case_methods = (
        'time_ufunc_types',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
    )


class Broadcast(_AggregateBenchmark):
    """Aggregate broadcast with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_Broadcast, 'time_broadcast', ()),  # bench_ufunc.Broadcast.time_broadcast
    )
    run_repeat = (
        1,  # bench_ufunc.Broadcast.time_broadcast
    )
    case_methods = (
        'time_broadcast',
    )
    case_types = (
        _Official_bench_ufunc_Broadcast,
    )


class Cbrt(_AggregateBenchmark):
    """Aggregate cbrt with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (15,)),  # bench_ufunc.UFunc.time_ufunc_types('cbrt')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (7, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (7, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 2, 'f')
    )
    run_repeat = (
        2,  # bench_ufunc.UFunc.time_ufunc_types('cbrt')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cbrt'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cbrt'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Ceil(_AggregateBenchmark):
    """Aggregate ceil with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (16,)),  # bench_ufunc.UFunc.time_ufunc_types('ceil')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (8, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (8, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 2, 'f')
    )
    run_repeat = (
        22,  # bench_ufunc.UFunc.time_ufunc_types('ceil')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'ceil'>, 4, 2, 'f')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'ceil'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Comparison(_AggregateBenchmark):
    """Aggregate Comparison with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_binary', (10,)),  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.bool'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_binary', (8,)),  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.float32'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_binary', (9,)),  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_binary', (1,)),  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.int16'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_binary', (2,)),  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.int32'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_binary', (3,)),  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.int64'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_binary', (0,)),  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.int8'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_binary', (5,)),  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.uint16'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_binary', (6,)),  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.uint32'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_binary', (7,)),  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.uint64'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_binary', (4,)),  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.uint8'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar1', (10,)),  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.bool'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar1', (8,)),  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.float32'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar1', (9,)),  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar1', (1,)),  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.int16'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar1', (2,)),  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.int32'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar1', (3,)),  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.int64'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar1', (0,)),  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.int8'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar1', (5,)),  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.uint16'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar1', (6,)),  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.uint32'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar1', (7,)),  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.uint64'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar1', (4,)),  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.uint8'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar2', (10,)),  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.bool'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar2', (8,)),  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.float32'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar2', (9,)),  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar2', (1,)),  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.int16'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar2', (2,)),  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.int32'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar2', (3,)),  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.int64'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar2', (0,)),  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.int8'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar2', (5,)),  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.uint16'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar2', (6,)),  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.uint32'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar2', (7,)),  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.uint64'>)
        _select_case_params(_Official_bench_ufunc_CustomComparison, 'time_less_than_scalar2', (4,)),  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.uint8'>)
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (1, 3)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__eq__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (1, 5)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__eq__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (1, 0)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__eq__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (1, 2)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__eq__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (1, 4)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__eq__', 'int64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (2, 3)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__ge__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (2, 5)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__ge__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (2, 0)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__ge__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (2, 2)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__ge__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (2, 4)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__ge__', 'int64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (3, 3)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__gt__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (3, 5)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__gt__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (3, 0)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__gt__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (3, 2)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__gt__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (3, 4)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__gt__', 'int64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (4, 3)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__le__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (4, 5)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__le__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (4, 0)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__le__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (4, 2)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__le__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (4, 4)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__le__', 'int64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (5, 3)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__lt__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (5, 5)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__lt__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (5, 0)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__lt__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (5, 2)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__lt__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (5, 4)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__lt__', 'int64')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (26,)),  # bench_ufunc.UFunc.time_ufunc_types('equal')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (39,)),  # bench_ufunc.UFunc.time_ufunc_types('greater')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (40,)),  # bench_ufunc.UFunc.time_ufunc_types('greater_equal')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (51,)),  # bench_ufunc.UFunc.time_ufunc_types('less')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (52,)),  # bench_ufunc.UFunc.time_ufunc_types('less_equal')
    )
    run_repeat = (
        313,  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.bool'>)
        167,  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.float32'>)
        77,  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.float64'>)
        295,  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.int16'>)
        173,  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.int32'>)
        79,  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.int64'>)
        455,  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.int8'>)
        304,  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.uint16'>)
        170,  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.uint32'>)
        79,  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.uint64'>)
        477,  # bench_ufunc.CustomComparison.time_less_than_binary(<class 'numpy.uint8'>)
        400,  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.bool'>)
        182,  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.float32'>)
        103,  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.float64'>)
        304,  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.int16'>)
        186,  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.int32'>)
        106,  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.int64'>)
        385,  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.int8'>)
        313,  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.uint16'>)
        186,  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.uint32'>)
        103,  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.uint64'>)
        400,  # bench_ufunc.CustomComparison.time_less_than_scalar1(<class 'numpy.uint8'>)
        400,  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.bool'>)
        186,  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.float32'>)
        105,  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.float64'>)
        313,  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.int16'>)
        182,  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.int32'>)
        104,  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.int64'>)
        400,  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.int8'>)
        304,  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.uint16'>)
        186,  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.uint32'>)
        104,  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.uint64'>)
        400,  # bench_ufunc.CustomComparison.time_less_than_scalar2(<class 'numpy.uint8'>)
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__eq__', 'float32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__eq__', 'float64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__eq__', 'int16')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__eq__', 'int32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__eq__', 'int64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__ge__', 'float32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__ge__', 'float64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__ge__', 'int16')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__ge__', 'int32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__ge__', 'int64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__gt__', 'float32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__gt__', 'float64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__gt__', 'int16')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__gt__', 'int32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__gt__', 'int64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__le__', 'float32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__le__', 'float64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__le__', 'int16')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__le__', 'int32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__le__', 'int64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__lt__', 'float32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__lt__', 'float64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__lt__', 'int16')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__lt__', 'int32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__lt__', 'int64')
        13,  # bench_ufunc.UFunc.time_ufunc_types('equal')
        12,  # bench_ufunc.UFunc.time_ufunc_types('greater')
        12,  # bench_ufunc.UFunc.time_ufunc_types('greater_equal')
        12,  # bench_ufunc.UFunc.time_ufunc_types('less')
        12,  # bench_ufunc.UFunc.time_ufunc_types('less_equal')
    )
    case_methods = (
        'time_less_than_binary',
        'time_less_than_binary',
        'time_less_than_binary',
        'time_less_than_binary',
        'time_less_than_binary',
        'time_less_than_binary',
        'time_less_than_binary',
        'time_less_than_binary',
        'time_less_than_binary',
        'time_less_than_binary',
        'time_less_than_binary',
        'time_less_than_scalar1',
        'time_less_than_scalar1',
        'time_less_than_scalar1',
        'time_less_than_scalar1',
        'time_less_than_scalar1',
        'time_less_than_scalar1',
        'time_less_than_scalar1',
        'time_less_than_scalar1',
        'time_less_than_scalar1',
        'time_less_than_scalar1',
        'time_less_than_scalar1',
        'time_less_than_scalar2',
        'time_less_than_scalar2',
        'time_less_than_scalar2',
        'time_less_than_scalar2',
        'time_less_than_scalar2',
        'time_less_than_scalar2',
        'time_less_than_scalar2',
        'time_less_than_scalar2',
        'time_less_than_scalar2',
        'time_less_than_scalar2',
        'time_less_than_scalar2',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ufunc_types',
        'time_ufunc_types',
        'time_ufunc_types',
        'time_ufunc_types',
        'time_ufunc_types',
    )
    case_types = (
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_CustomComparison,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
    )


class Deg(_AggregateBenchmark):
    """Aggregate deg with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (22,)),  # bench_ufunc.UFunc.time_ufunc_types('deg2rad')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (23,)),  # bench_ufunc.UFunc.time_ufunc_types('degrees')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (12, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (13, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (12, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (13, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 2, 'f')
    )
    run_repeat = (
        10,  # bench_ufunc.UFunc.time_ufunc_types('deg2rad')
        10,  # bench_ufunc.UFunc.time_ufunc_types('degrees')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'deg2rad'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'degrees'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'deg2rad'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'degrees'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Exp2(_AggregateBenchmark):
    """Aggregate exp2 with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (28,)),  # bench_ufunc.UFunc.time_ufunc_types('exp2')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (15, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (15, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 2, 'f')
    )
    run_repeat = (
        2,  # bench_ufunc.UFunc.time_ufunc_types('exp2')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp2'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp2'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Floor(_AggregateBenchmark):
    """Aggregate floor with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (32,)),  # bench_ufunc.UFunc.time_ufunc_types('floor')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (18, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (18, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 2, 'f')
    )
    run_repeat = (
        22,  # bench_ufunc.UFunc.time_ufunc_types('floor')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'floor'>, 4, 2, 'f')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'floor'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class FloorDivide(_AggregateBenchmark):
    """Aggregate FloorDivide with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (1, 3)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, -43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (1, 1)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, -8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (1, 2)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, 43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (1, 0)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, 8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (2, 3)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, -43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (2, 1)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, -8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (2, 2)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, 43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (2, 0)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, 8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (4, 3)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, -43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (4, 1)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, -8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (4, 2)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, 43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (4, 0)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, 8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (0, 3)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, -43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (0, 1)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, -8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (0, 2)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, 43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (0, 0)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, 8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (3, 3)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, -43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (3, 1)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, -8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (3, 2)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, 43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideInt, 'time_floor_divide_int', (3, 0)),  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, 8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideUInt, 'time_floor_divide_uint', (1, 1)),  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint16'>, 43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideUInt, 'time_floor_divide_uint', (1, 0)),  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint16'>, 8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideUInt, 'time_floor_divide_uint', (2, 1)),  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint32'>, 43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideUInt, 'time_floor_divide_uint', (2, 0)),  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint32'>, 8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideUInt, 'time_floor_divide_uint', (3, 1)),  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint64'>, 43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideUInt, 'time_floor_divide_uint', (3, 0)),  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint64'>, 8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideUInt, 'time_floor_divide_uint', (0, 1)),  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint8'>, 43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideUInt, 'time_floor_divide_uint', (0, 0)),  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint8'>, 8)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideUInt, 'time_floor_divide_uint', (4, 1)),  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.ulonglong'>, 43)
        _select_case_params(_Official_bench_ufunc_CustomScalarFloorDivideUInt, 'time_floor_divide_uint', (4, 0)),  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.ulonglong'>, 8)
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (33,)),  # bench_ufunc.UFunc.time_ufunc_types('floor_divide')
    )
    run_repeat = (
        264,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, -43)
        264,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, -8)
        257,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, 43)
        264,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, 8)
        152,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, -43)
        150,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, -8)
        150,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, 43)
        150,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, 8)
        18,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, -43)
        18,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, -8)
        18,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, 43)
        18,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, 8)
        400,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, -43)
        400,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, -8)
        400,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, 43)
        400,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, 8)
        18,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, -43)
        18,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, -8)
        18,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, 43)
        18,  # bench_ufunc.CustomScalarFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, 8)
        435,  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint16'>, 43)
        435,  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint16'>, 8)
        295,  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint32'>, 43)
        286,  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint32'>, 8)
        100,  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint64'>, 43)
        100,  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint64'>, 8)
        556,  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint8'>, 43)
        589,  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.uint8'>, 8)
        100,  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.ulonglong'>, 43)
        100,  # bench_ufunc.CustomScalarFloorDivideUInt.time_floor_divide_uint(<class 'numpy.ulonglong'>, 8)
        5,  # bench_ufunc.UFunc.time_ufunc_types('floor_divide')
    )
    case_methods = (
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_int',
        'time_floor_divide_uint',
        'time_floor_divide_uint',
        'time_floor_divide_uint',
        'time_floor_divide_uint',
        'time_floor_divide_uint',
        'time_floor_divide_uint',
        'time_floor_divide_uint',
        'time_floor_divide_uint',
        'time_floor_divide_uint',
        'time_floor_divide_uint',
        'time_ufunc_types',
    )
    case_types = (
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideInt,
        _Official_bench_ufunc_CustomScalarFloorDivideUInt,
        _Official_bench_ufunc_CustomScalarFloorDivideUInt,
        _Official_bench_ufunc_CustomScalarFloorDivideUInt,
        _Official_bench_ufunc_CustomScalarFloorDivideUInt,
        _Official_bench_ufunc_CustomScalarFloorDivideUInt,
        _Official_bench_ufunc_CustomScalarFloorDivideUInt,
        _Official_bench_ufunc_CustomScalarFloorDivideUInt,
        _Official_bench_ufunc_CustomScalarFloorDivideUInt,
        _Official_bench_ufunc_CustomScalarFloorDivideUInt,
        _Official_bench_ufunc_CustomScalarFloorDivideUInt,
        _Official_bench_ufunc_UFunc,
    )


class Fmax(_AggregateBenchmark):
    """Aggregate fmax with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (34,)),  # bench_ufunc.UFunc.time_ufunc_types('fmax')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 0, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (2, 1, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 0, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (2, 1, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 4, 'f')
    )
    run_repeat = (
        10,  # bench_ufunc.UFunc.time_ufunc_types('fmax')
        3,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 1, 'd')
        5,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 2, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 1, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 4, 'd')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 1, 4, 4, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 1, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 1, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 4, 'd')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmax'>, 2, 4, 4, 'f')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 1, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 2, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 4, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 1, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 1, 4, 4, 'f')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 1, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 2, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 4, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 1, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmax'>, 2, 4, 4, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
    )


class Fmin(_AggregateBenchmark):
    """Aggregate fmin with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (35,)),  # bench_ufunc.UFunc.time_ufunc_types('fmin')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 0, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (3, 1, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 0, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (3, 1, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 4, 'f')
    )
    run_repeat = (
        10,  # bench_ufunc.UFunc.time_ufunc_types('fmin')
        3,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 1, 'd')
        5,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 2, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 1, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 4, 'd')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 1, 4, 4, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 1, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 1, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 4, 'd')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'fmin'>, 2, 4, 4, 'f')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 1, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 2, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 4, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 1, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 1, 4, 4, 'f')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 1, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 2, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 4, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 1, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'fmin'>, 2, 4, 4, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
    )


class Getitem(_AggregateBenchmark):
    """Aggregate getitem with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_NDArrayGetItem, 'time_methods_getitem', (2, 1)),  # bench_ufunc.NDArrayGetItem.time_methods_getitem((-1, 0), 'big')
        _select_case_params(_Official_bench_ufunc_NDArrayGetItem, 'time_methods_getitem', (2, 0)),  # bench_ufunc.NDArrayGetItem.time_methods_getitem((-1, 0), 'small')
        _select_case_params(_Official_bench_ufunc_NDArrayGetItem, 'time_methods_getitem', (1, 1)),  # bench_ufunc.NDArrayGetItem.time_methods_getitem((0, 0), 'big')
        _select_case_params(_Official_bench_ufunc_NDArrayGetItem, 'time_methods_getitem', (1, 0)),  # bench_ufunc.NDArrayGetItem.time_methods_getitem((0, 0), 'small')
        _select_case_params(_Official_bench_ufunc_NDArrayGetItem, 'time_methods_getitem', (0, 1)),  # bench_ufunc.NDArrayGetItem.time_methods_getitem(0, 'big')
        _select_case_params(_Official_bench_ufunc_NDArrayGetItem, 'time_methods_getitem', (0, 0)),  # bench_ufunc.NDArrayGetItem.time_methods_getitem(0, 'small')
        _select_case_params(_Official_bench_ufunc_NDArrayGetItem, 'time_methods_getitem', (3, 1)),  # bench_ufunc.NDArrayGetItem.time_methods_getitem([0, -1], 'big')
        _select_case_params(_Official_bench_ufunc_NDArrayGetItem, 'time_methods_getitem', (3, 0)),  # bench_ufunc.NDArrayGetItem.time_methods_getitem([0, -1], 'small')
    )
    run_repeat = (
        1000,  # bench_ufunc.NDArrayGetItem.time_methods_getitem((-1, 0), 'big')
        1000,  # bench_ufunc.NDArrayGetItem.time_methods_getitem((-1, 0), 'small')
        1000,  # bench_ufunc.NDArrayGetItem.time_methods_getitem((0, 0), 'big')
        1000,  # bench_ufunc.NDArrayGetItem.time_methods_getitem((0, 0), 'small')
        1000,  # bench_ufunc.NDArrayGetItem.time_methods_getitem(0, 'big')
        1000,  # bench_ufunc.NDArrayGetItem.time_methods_getitem(0, 'small')
        527,  # bench_ufunc.NDArrayGetItem.time_methods_getitem([0, -1], 'big')
        589,  # bench_ufunc.NDArrayGetItem.time_methods_getitem([0, -1], 'small')
    )
    case_methods = (
        'time_methods_getitem',
        'time_methods_getitem',
        'time_methods_getitem',
        'time_methods_getitem',
        'time_methods_getitem',
        'time_methods_getitem',
        'time_methods_getitem',
        'time_methods_getitem',
    )
    case_types = (
        _Official_bench_ufunc_NDArrayGetItem,
        _Official_bench_ufunc_NDArrayGetItem,
        _Official_bench_ufunc_NDArrayGetItem,
        _Official_bench_ufunc_NDArrayGetItem,
        _Official_bench_ufunc_NDArrayGetItem,
        _Official_bench_ufunc_NDArrayGetItem,
        _Official_bench_ufunc_NDArrayGetItem,
        _Official_bench_ufunc_NDArrayGetItem,
    )


class Invert(_AggregateBenchmark):
    """Aggregate invert with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (43,)),  # bench_ufunc.UFunc.time_ufunc_types('invert')
    )
    run_repeat = (
        1,  # bench_ufunc.UFunc.time_ufunc_types('invert')
    )
    case_methods = (
        'time_ufunc_types',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
    )


class Is(_AggregateBenchmark):
    """Aggregate is with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (44,)),  # bench_ufunc.UFunc.time_ufunc_types('isfinite')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (45,)),  # bench_ufunc.UFunc.time_ufunc_types('isinf')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (46,)),  # bench_ufunc.UFunc.time_ufunc_types('isnan')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (47,)),  # bench_ufunc.UFunc.time_ufunc_types('isnat')
    )
    run_repeat = (
        19,  # bench_ufunc.UFunc.time_ufunc_types('isfinite')
        19,  # bench_ufunc.UFunc.time_ufunc_types('isinf')
        20,  # bench_ufunc.UFunc.time_ufunc_types('isnan')
        1000,  # bench_ufunc.UFunc.time_ufunc_types('isnat')
    )
    case_methods = (
        'time_ufunc_types',
        'time_ufunc_types',
        'time_ufunc_types',
        'time_ufunc_types',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
    )


class Log2(_AggregateBenchmark):
    """Aggregate log2 with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (56,)),  # bench_ufunc.UFunc.time_ufunc_types('log2')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (22, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (22, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 2, 'f')
    )
    run_repeat = (
        2,  # bench_ufunc.UFunc.time_ufunc_types('log2')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log2'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log2'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Logical(_AggregateBenchmark):
    """Aggregate logical with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (59,)),  # bench_ufunc.UFunc.time_ufunc_types('logical_and')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (60,)),  # bench_ufunc.UFunc.time_ufunc_types('logical_not')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (61,)),  # bench_ufunc.UFunc.time_ufunc_types('logical_or')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (62,)),  # bench_ufunc.UFunc.time_ufunc_types('logical_xor')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (6, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (6, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (6, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (6, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (6, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (6, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (6, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (6, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (6, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (6, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (7, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (7, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (7, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (7, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (7, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (7, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (7, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (7, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (7, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (7, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (8, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (8, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (8, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (8, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (8, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (8, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (8, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (8, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (8, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (8, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (6, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (6, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (6, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (6, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (6, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (6, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (6, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (6, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (6, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (6, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (7, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (7, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (7, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (7, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (7, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (7, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (7, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (7, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (7, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (7, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (8, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (8, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (8, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (8, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (8, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (8, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (8, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (8, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (8, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (8, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (23, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 2, 'f')
    )
    run_repeat = (
        9,  # bench_ufunc.UFunc.time_ufunc_types('logical_and')
        16,  # bench_ufunc.UFunc.time_ufunc_types('logical_not')
        10,  # bench_ufunc.UFunc.time_ufunc_types('logical_or')
        10,  # bench_ufunc.UFunc.time_ufunc_types('logical_xor')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'B')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'H')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'I')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'L')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'Q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'b')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'h')
        477,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'i')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'l')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_and'>, 1, 1, 1, 'q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'B')
        770,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'H')
        500,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'I')
        304,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'L')
        304,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'Q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'b')
        770,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'h')
        500,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'i')
        304,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'l')
        304,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_or'>, 1, 1, 1, 'q')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'B')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'H')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'I')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'L')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'Q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'b')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'h')
        477,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'i')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'l')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'logical_xor'>, 1, 1, 1, 'q')
        667,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'B')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'H')
        417,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'I')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'L')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'Q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'b')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'h')
        435,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'i')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'l')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_and'>, 1, 1, 1, 'q')
        667,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'B')
        556,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'H')
        417,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'I')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'L')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'Q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'b')
        556,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'h')
        417,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'i')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'l')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_or'>, 1, 1, 1, 'q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'B')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'H')
        435,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'I')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'L')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'Q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'b')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'h')
        435,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'i')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'l')
        278,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'logical_xor'>, 1, 1, 1, 'q')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 1, 'e')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 1, 'e')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'logical_not'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_ufunc_types',
        'time_ufunc_types',
        'time_ufunc_types',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
    )


class Minimum(_AggregateBenchmark):
    """Aggregate minimum with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (66,)),  # bench_ufunc.UFunc.time_ufunc_types('minimum')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 0, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (1, 1, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 0, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (1, 1, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 1, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 1, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 1, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 1, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 1, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 1, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 1, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 1, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 1, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 0, 1, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 0, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 0, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 0, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 0, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 0, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 0, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 0, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 0, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 0, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 0, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 1, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 1, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 1, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 1, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 1, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 1, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 1, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 1, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 1, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 0, 1, 1, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 0, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 0, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 0, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 0, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 0, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 0, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 0, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 0, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 0, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 0, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 1, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 1, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 1, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 1, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 1, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 1, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 1, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 1, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 1, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 0, 1, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 0, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 0, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 0, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 0, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 0, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 0, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 0, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 0, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 0, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 0, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 1, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 1, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 1, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 1, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 1, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 1, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 1, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 1, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 1, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (1, 1, 1, 1, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'q')
    )
    run_repeat = (
        10,  # bench_ufunc.UFunc.time_ufunc_types('minimum')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'd')
        5,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 1, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 4, 'd')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 1, 4, 4, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 1, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 4, 'd')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'minimum'>, 2, 4, 4, 'f')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 1, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 2, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 4, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 1, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 1, 4, 4, 'f')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 1, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 2, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 4, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 1, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'minimum'>, 2, 4, 4, 'f')
        218,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'B')
        141,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'H')
        59,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'I')
        23,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'L')
        23,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'Q')
        218,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'b')
        137,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'h')
        59,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'i')
        23,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'l')
        23,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 1, 'q')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'B')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'H')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'I')
        16,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'L')
        16,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'b')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'h')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'i')
        16,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'l')
        16,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 1, 2, 'q')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'B')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'H')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'I')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'L')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'b')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'h')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'i')
        16,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'l')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 1, 'q')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'B')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'H')
        23,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'I')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'L')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'b')
        20,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'h')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'i')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'l')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 1, 2, 2, 'q')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'B')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'H')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'I')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'L')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'b')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'h')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'i')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'l')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 1, 'q')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'B')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'H')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'I')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'L')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'b')
        20,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'h')
        23,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'i')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'l')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 1, 2, 'q')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'B')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'H')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'I')
        13,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'L')
        13,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'b')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'h')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'i')
        13,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'l')
        13,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 1, 'q')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'B')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'H')
        20,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'I')
        10,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'L')
        10,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'b')
        20,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'h')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'i')
        10,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'l')
        10,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'minimum'>, 2, 2, 2, 'q')
    )
    case_methods = (
        'time_ufunc_types',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryFP,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
        _Official_bench_ufunc_strides_BinaryInt,
    )


class Mul(_AggregateBenchmark):
    """Aggregate mul with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (7, 3)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__mul__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (7, 5)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__mul__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (7, 0)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__mul__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (7, 2)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__mul__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (7, 4)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__mul__', 'int64')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (2, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (2, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (2, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (2, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (2, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (2, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (2, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (2, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (2, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (2, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (2, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (2, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (2, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (2, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (2, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (2, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (2, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (2, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (2, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (2, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'q')
    )
    run_repeat = (
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__mul__', 'float32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__mul__', 'float64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__mul__', 'int16')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__mul__', 'int32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__mul__', 'int64')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'B')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'H')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'I')
        218,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'L')
        218,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'Q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'b')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'h')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'i')
        218,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'l')
        218,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'multiply'>, 1, 1, 1, 'q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'B')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'H')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'I')
        193,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'L')
        197,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'Q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'b')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'h')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'i')
        197,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'l')
        193,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'multiply'>, 1, 1, 1, 'q')
    )
    case_methods = (
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
    )
    case_types = (
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
    )


class Neg(_AggregateBenchmark):
    """Aggregate neg with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (0, 4)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__abs__', 'int64')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (1, 3)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__neg__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (1, 5)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__neg__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (1, 0)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__neg__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (1, 2)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__neg__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (1, 4)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__neg__', 'int64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (8, 3)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__ne__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (8, 5)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__ne__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (8, 0)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__ne__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (8, 2)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__ne__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (8, 4)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__ne__', 'int64')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (70,)),  # bench_ufunc.UFunc.time_ufunc_types('negative')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (24, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 2, 'f')
    )
    run_repeat = (
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__abs__', 'int64')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__neg__', 'float32')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__neg__', 'float64')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__neg__', 'int16')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__neg__', 'int32')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__neg__', 'int64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__ne__', 'float32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__ne__', 'float64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__ne__', 'int16')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__ne__', 'int32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__ne__', 'int64')
        17,  # bench_ufunc.UFunc.time_ufunc_types('negative')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 1, 'd')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 2, 'd')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 1, 'd')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 2, 'd')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'negative'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
    )


class Pos(_AggregateBenchmark):
    """Aggregate pos with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (2, 3)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__pos__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (2, 5)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__pos__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (2, 0)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__pos__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (2, 2)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__pos__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV0, 'time_ndarray_meth', (2, 4)),  # bench_ufunc.MethodsV0.time_ndarray_meth('__pos__', 'int64')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (73,)),  # bench_ufunc.UFunc.time_ufunc_types('positive')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (25, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 2, 'f')
    )
    run_repeat = (
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__pos__', 'float32')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__pos__', 'float64')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__pos__', 'int16')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__pos__', 'int32')
        1000,  # bench_ufunc.MethodsV0.time_ndarray_meth('__pos__', 'int64')
        16,  # bench_ufunc.UFunc.time_ufunc_types('positive')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 1, 'd')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 2, 'd')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 2, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 1, 'd')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 2, 'd')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'positive'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_MethodsV0,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
    )


class Rad(_AggregateBenchmark):
    """Aggregate rad with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (75,)),  # bench_ufunc.UFunc.time_ufunc_types('rad2deg')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (76,)),  # bench_ufunc.UFunc.time_ufunc_types('radians')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (26, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (27, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (26, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (27, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 2, 'f')
    )
    run_repeat = (
        10,  # bench_ufunc.UFunc.time_ufunc_types('rad2deg')
        10,  # bench_ufunc.UFunc.time_ufunc_types('radians')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rad2deg'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'radians'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rad2deg'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'radians'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Reciprocal(_AggregateBenchmark):
    """Aggregate reciprocal with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (77,)),  # bench_ufunc.UFunc.time_ufunc_types('reciprocal')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (28, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (28, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 2, 'f')
    )
    run_repeat = (
        7,  # bench_ufunc.UFunc.time_ufunc_types('reciprocal')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'reciprocal'>, 4, 2, 'f')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'reciprocal'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Rint(_AggregateBenchmark):
    """Aggregate rint with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (80,)),  # bench_ufunc.UFunc.time_ufunc_types('rint')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (29, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (29, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 2, 'f')
    )
    run_repeat = (
        12,  # bench_ufunc.UFunc.time_ufunc_types('rint')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'rint'>, 4, 2, 'f')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'rint'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Setitem(_AggregateBenchmark):
    """Aggregate setitem with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_NDArraySetItem, 'time_methods_setitem', (2, 1)),  # bench_ufunc.NDArraySetItem.time_methods_setitem((-1, 0), 'big')
        _select_case_params(_Official_bench_ufunc_NDArraySetItem, 'time_methods_setitem', (2, 0)),  # bench_ufunc.NDArraySetItem.time_methods_setitem((-1, 0), 'small')
        _select_case_params(_Official_bench_ufunc_NDArraySetItem, 'time_methods_setitem', (1, 1)),  # bench_ufunc.NDArraySetItem.time_methods_setitem((0, 0), 'big')
        _select_case_params(_Official_bench_ufunc_NDArraySetItem, 'time_methods_setitem', (1, 0)),  # bench_ufunc.NDArraySetItem.time_methods_setitem((0, 0), 'small')
        _select_case_params(_Official_bench_ufunc_NDArraySetItem, 'time_methods_setitem', (0, 1)),  # bench_ufunc.NDArraySetItem.time_methods_setitem(0, 'big')
        _select_case_params(_Official_bench_ufunc_NDArraySetItem, 'time_methods_setitem', (0, 0)),  # bench_ufunc.NDArraySetItem.time_methods_setitem(0, 'small')
        _select_case_params(_Official_bench_ufunc_NDArraySetItem, 'time_methods_setitem', (3, 1)),  # bench_ufunc.NDArraySetItem.time_methods_setitem([0, -1], 'big')
        _select_case_params(_Official_bench_ufunc_NDArraySetItem, 'time_methods_setitem', (3, 0)),  # bench_ufunc.NDArraySetItem.time_methods_setitem([0, -1], 'small')
    )
    run_repeat = (
        1000,  # bench_ufunc.NDArraySetItem.time_methods_setitem((-1, 0), 'big')
        1000,  # bench_ufunc.NDArraySetItem.time_methods_setitem((-1, 0), 'small')
        1000,  # bench_ufunc.NDArraySetItem.time_methods_setitem((0, 0), 'big')
        1000,  # bench_ufunc.NDArraySetItem.time_methods_setitem((0, 0), 'small')
        1000,  # bench_ufunc.NDArraySetItem.time_methods_setitem(0, 'big')
        1000,  # bench_ufunc.NDArraySetItem.time_methods_setitem(0, 'small')
        527,  # bench_ufunc.NDArraySetItem.time_methods_setitem([0, -1], 'big')
        1000,  # bench_ufunc.NDArraySetItem.time_methods_setitem([0, -1], 'small')
    )
    case_methods = (
        'time_methods_setitem',
        'time_methods_setitem',
        'time_methods_setitem',
        'time_methods_setitem',
        'time_methods_setitem',
        'time_methods_setitem',
        'time_methods_setitem',
        'time_methods_setitem',
    )
    case_types = (
        _Official_bench_ufunc_NDArraySetItem,
        _Official_bench_ufunc_NDArraySetItem,
        _Official_bench_ufunc_NDArraySetItem,
        _Official_bench_ufunc_NDArraySetItem,
        _Official_bench_ufunc_NDArraySetItem,
        _Official_bench_ufunc_NDArraySetItem,
        _Official_bench_ufunc_NDArraySetItem,
        _Official_bench_ufunc_NDArraySetItem,
    )


class Shift(_AggregateBenchmark):
    """Aggregate shift with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (0, 2)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'int16')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (0, 3)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'int32')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (0, 4)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'int64')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (0, 1)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'int8')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (0, 0)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'intp')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (0, 6)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'uint16')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (0, 7)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'uint32')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (0, 8)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'uint64')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (0, 5)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'uint8')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (1, 2)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'int16')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (1, 3)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'int32')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (1, 4)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'int64')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (1, 1)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'int8')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (1, 0)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'intp')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (1, 6)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'uint16')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (1, 7)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'uint32')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (1, 8)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'uint64')
        _select_case_params(_Official_bench_ufunc_NDArrayLRShifts, 'time_ndarray_meth', (1, 5)),  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'uint8')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (50,)),  # bench_ufunc.UFunc.time_ufunc_types('left_shift')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (79,)),  # bench_ufunc.UFunc.time_ufunc_types('right_shift')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (10, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (10, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (10, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (10, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (10, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (10, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (10, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (10, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (10, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (10, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (9, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (9, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (9, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (9, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (9, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (9, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (9, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (9, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (9, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (9, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (10, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (10, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (10, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (10, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (10, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (10, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (10, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (10, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (10, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (10, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (9, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (9, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (9, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (9, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (9, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (9, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (9, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (9, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (9, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (9, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'q')
    )
    run_repeat = (
        770,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'int16')
        770,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'int32')
        770,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'int64')
        910,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'int8')
        770,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'intp')
        834,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'uint16')
        770,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'uint32')
        715,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'uint64')
        910,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__lshift__', 'uint8')
        834,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'int16')
        770,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'int32')
        715,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'int64')
        910,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'int8')
        770,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'intp')
        834,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'uint16')
        770,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'uint32')
        625,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'uint64')
        910,  # bench_ufunc.NDArrayLRShifts.time_ndarray_meth('__rshift__', 'uint8')
        58,  # bench_ufunc.UFunc.time_ufunc_types('left_shift')
        51,  # bench_ufunc.UFunc.time_ufunc_types('right_shift')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'B')
        477,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'H')
        500,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'I')
        218,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'L')
        286,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'Q')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'b')
        477,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'h')
        527,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'i')
        218,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'l')
        286,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'left_shift'>, 1, 1, 1, 'q')
        400,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'B')
        435,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'H')
        477,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'I')
        218,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'L')
        264,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'Q')
        385,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'b')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'h')
        358,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'i')
        182,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'l')
        189,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'right_shift'>, 1, 1, 1, 'q')
        435,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'B')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'H')
        435,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'I')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'L')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'Q')
        435,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'b')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'h')
        435,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'i')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'l')
        271,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'left_shift'>, 1, 1, 1, 'q')
        400,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'B')
        400,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'H')
        417,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'I')
        239,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'L')
        197,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'Q')
        400,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'b')
        400,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'h')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'i')
        239,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'l')
        239,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'right_shift'>, 1, 1, 1, 'q')
    )
    case_methods = (
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ufunc_types',
        'time_ufunc_types',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
    )
    case_types = (
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_NDArrayLRShifts,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
    )


class Sub(_AggregateBenchmark):
    """Aggregate sub with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (10, 3)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__sub__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (10, 5)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__sub__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (10, 0)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__sub__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (10, 2)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__sub__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (10, 4)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__sub__', 'int64')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (1, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (1, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (1, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (1, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (1, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (1, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (1, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (1, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (1, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary', (1, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (1, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (1, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (1, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (1, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (1, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (1, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (1, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (1, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (1, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryIntContig, 'time_binary_scalar_in0', (1, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'q')
    )
    run_repeat = (
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__sub__', 'float32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__sub__', 'float64')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__sub__', 'int16')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__sub__', 'int32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__sub__', 'int64')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'B')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'H')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'I')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'L')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'Q')
        1000,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'b')
        834,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'h')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'i')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'l')
        371,  # bench_ufunc_strides.BinaryIntContig.time_binary(<ufunc 'subtract'>, 1, 1, 1, 'q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'B')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'H')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'I')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'L')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'Q')
        715,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'b')
        589,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'h')
        455,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'i')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'l')
        313,  # bench_ufunc_strides.BinaryIntContig.time_binary_scalar_in0(<ufunc 'subtract'>, 1, 1, 1, 'q')
    )
    case_methods = (
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
    )
    case_types = (
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
        _Official_bench_ufunc_strides_BinaryIntContig,
    )


class Truediv(_AggregateBenchmark):
    """Aggregate truediv with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (11, 3)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__truediv__', 'float32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (11, 5)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__truediv__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (11, 0)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__truediv__', 'int16')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (11, 2)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__truediv__', 'int32')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (11, 4)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__truediv__', 'int64')
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (91,)),  # bench_ufunc.UFunc.time_ufunc_types('true_divide')
    )
    run_repeat = (
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__truediv__', 'float32')
        1000,  # bench_ufunc.MethodsV1.time_ndarray_meth('__truediv__', 'float64')
        834,  # bench_ufunc.MethodsV1.time_ndarray_meth('__truediv__', 'int16')
        834,  # bench_ufunc.MethodsV1.time_ndarray_meth('__truediv__', 'int32')
        834,  # bench_ufunc.MethodsV1.time_ndarray_meth('__truediv__', 'int64')
        5,  # bench_ufunc.UFunc.time_ufunc_types('true_divide')
    )
    case_methods = (
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ndarray_meth',
        'time_ufunc_types',
    )
    case_types = (
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_UFunc,
    )


class Trunc(_AggregateBenchmark):
    """Aggregate trunc with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (92,)),  # bench_ufunc.UFunc.time_ufunc_types('trunc')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (37, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (37, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 2, 'f')
    )
    run_repeat = (
        22,  # bench_ufunc.UFunc.time_ufunc_types('trunc')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'trunc'>, 4, 2, 'f')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 1, 'e')
        7,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 2, 'e')
        4,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 2, 'e')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'trunc'>, 4, 2, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )
