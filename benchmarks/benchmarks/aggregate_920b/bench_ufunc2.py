"""920b platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_ufunc import (
    CustomArrayFloorDivideInt as _Official_bench_ufunc_CustomArrayFloorDivideInt,
    NDArrayAsType as _Official_bench_ufunc_NDArrayAsType,
    UFunc as _Official_bench_ufunc_UFunc,
    UFuncSmall as _Official_bench_ufunc_UFuncSmall,
)
from ..bench_ufunc_strides import (
    BinaryFP as _Official_bench_ufunc_strides_BinaryFP,
    BinaryFPSpecial as _Official_bench_ufunc_strides_BinaryFPSpecial,
    BinaryInt as _Official_bench_ufunc_strides_BinaryInt,
    UnaryFP as _Official_bench_ufunc_strides_UnaryFP,
    UnaryFPSpecial as _Official_bench_ufunc_strides_UnaryFPSpecial,
)


class Abs(_AggregateBenchmark):
    """Aggregate abs with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (0,)),  # bench_ufunc.UFunc.time_ufunc_types('abs')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_numpy_scalar', (0,)),  # bench_ufunc.UFuncSmall.time_ufunc_numpy_scalar('abs')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_python_float', (0,)),  # bench_ufunc.UFuncSmall.time_ufunc_python_float('abs')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_small_array', (0,)),  # bench_ufunc.UFuncSmall.time_ufunc_small_array('abs')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_small_array_inplace', (0,)),  # bench_ufunc.UFuncSmall.time_ufunc_small_array_inplace('abs')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_small_int_array', (0,)),  # bench_ufunc.UFuncSmall.time_ufunc_small_int_array('abs')
    )
    run_repeat = (
        9,  # bench_ufunc.UFunc.time_ufunc_types('abs')
        1000,  # bench_ufunc.UFuncSmall.time_ufunc_numpy_scalar('abs')
        1000,  # bench_ufunc.UFuncSmall.time_ufunc_python_float('abs')
        834,  # bench_ufunc.UFuncSmall.time_ufunc_small_array('abs')
        834,  # bench_ufunc.UFuncSmall.time_ufunc_small_array_inplace('abs')
        770,  # bench_ufunc.UFuncSmall.time_ufunc_small_int_array('abs')
    )
    case_methods = (
        'time_ufunc_types',
        'time_ufunc_numpy_scalar',
        'time_ufunc_python_float',
        'time_ufunc_small_array',
        'time_ufunc_small_array_inplace',
        'time_ufunc_small_int_array',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
    )


class Add(_AggregateBenchmark):
    """Aggregate add with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (2,)),  # bench_ufunc.UFunc.time_ufunc_types('add')
    )
    run_repeat = (
        1,  # bench_ufunc.UFunc.time_ufunc_types('add')
    )
    case_methods = (
        'time_ufunc_types',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
    )


class Astype(_AggregateBenchmark):
    """Aggregate astype with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_NDArrayAsType, 'time_astype', (29,)),  # bench_ufunc.NDArrayAsType.time_astype(('float32', 'float64'))
        _select_case_params(_Official_bench_ufunc_NDArrayAsType, 'time_astype', (28,)),  # bench_ufunc.NDArrayAsType.time_astype(('float32', 'int64'))
        _select_case_params(_Official_bench_ufunc_NDArrayAsType, 'time_astype', (3,)),  # bench_ufunc.NDArrayAsType.time_astype(('int16', 'float32'))
        _select_case_params(_Official_bench_ufunc_NDArrayAsType, 'time_astype', (5,)),  # bench_ufunc.NDArrayAsType.time_astype(('int16', 'float64'))
        _select_case_params(_Official_bench_ufunc_NDArrayAsType, 'time_astype', (2,)),  # bench_ufunc.NDArrayAsType.time_astype(('int16', 'int32'))
        _select_case_params(_Official_bench_ufunc_NDArrayAsType, 'time_astype', (4,)),  # bench_ufunc.NDArrayAsType.time_astype(('int16', 'int64'))
        _select_case_params(_Official_bench_ufunc_NDArrayAsType, 'time_astype', (19,)),  # bench_ufunc.NDArrayAsType.time_astype(('int32', 'float32'))
        _select_case_params(_Official_bench_ufunc_NDArrayAsType, 'time_astype', (21,)),  # bench_ufunc.NDArrayAsType.time_astype(('int32', 'float64'))
        _select_case_params(_Official_bench_ufunc_NDArrayAsType, 'time_astype', (20,)),  # bench_ufunc.NDArrayAsType.time_astype(('int32', 'int64'))
        _select_case_params(_Official_bench_ufunc_NDArrayAsType, 'time_astype', (37,)),  # bench_ufunc.NDArrayAsType.time_astype(('int64', 'float64'))
    )
    run_repeat = (
        257,  # bench_ufunc.NDArrayAsType.time_astype(('float32', 'float64'))
        167,  # bench_ufunc.NDArrayAsType.time_astype(('float32', 'int64'))
        271,  # bench_ufunc.NDArrayAsType.time_astype(('int16', 'float32'))
        213,  # bench_ufunc.NDArrayAsType.time_astype(('int16', 'float64'))
        334,  # bench_ufunc.NDArrayAsType.time_astype(('int16', 'int32'))
        250,  # bench_ufunc.NDArrayAsType.time_astype(('int16', 'int64'))
        286,  # bench_ufunc.NDArrayAsType.time_astype(('int32', 'float32'))
        228,  # bench_ufunc.NDArrayAsType.time_astype(('int32', 'float64'))
        250,  # bench_ufunc.NDArrayAsType.time_astype(('int32', 'int64'))
        218,  # bench_ufunc.NDArrayAsType.time_astype(('int64', 'float64'))
    )
    case_methods = (
        'time_astype',
        'time_astype',
        'time_astype',
        'time_astype',
        'time_astype',
        'time_astype',
        'time_astype',
        'time_astype',
        'time_astype',
        'time_astype',
    )
    case_types = (
        _Official_bench_ufunc_NDArrayAsType,
        _Official_bench_ufunc_NDArrayAsType,
        _Official_bench_ufunc_NDArrayAsType,
        _Official_bench_ufunc_NDArrayAsType,
        _Official_bench_ufunc_NDArrayAsType,
        _Official_bench_ufunc_NDArrayAsType,
        _Official_bench_ufunc_NDArrayAsType,
        _Official_bench_ufunc_NDArrayAsType,
        _Official_bench_ufunc_NDArrayAsType,
        _Official_bench_ufunc_NDArrayAsType,
    )


class Cos(_AggregateBenchmark):
    """Aggregate cos with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_numpy_scalar', (2,)),  # bench_ufunc.UFuncSmall.time_ufunc_numpy_scalar('cos')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_python_float', (2,)),  # bench_ufunc.UFuncSmall.time_ufunc_python_float('cos')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_small_array', (2,)),  # bench_ufunc.UFuncSmall.time_ufunc_small_array('cos')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_small_array_inplace', (2,)),  # bench_ufunc.UFuncSmall.time_ufunc_small_array_inplace('cos')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_small_int_array', (2,)),  # bench_ufunc.UFuncSmall.time_ufunc_small_int_array('cos')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (10, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cos'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (10, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cos'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (10, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cos'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (10, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cos'>, 1, 1, 'f')
    )
    run_repeat = (
        1000,  # bench_ufunc.UFuncSmall.time_ufunc_numpy_scalar('cos')
        1000,  # bench_ufunc.UFuncSmall.time_ufunc_python_float('cos')
        770,  # bench_ufunc.UFuncSmall.time_ufunc_small_array('cos')
        770,  # bench_ufunc.UFuncSmall.time_ufunc_small_array_inplace('cos')
        527,  # bench_ufunc.UFuncSmall.time_ufunc_small_int_array('cos')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cos'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'cos'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cos'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'cos'>, 1, 1, 'f')
    )
    case_methods = (
        'time_ufunc_numpy_scalar',
        'time_ufunc_python_float',
        'time_ufunc_small_array',
        'time_ufunc_small_array_inplace',
        'time_ufunc_small_int_array',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Divide(_AggregateBenchmark):
    """Aggregate divide with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (24,)),  # bench_ufunc.UFunc.time_ufunc_types('divide')
    )
    run_repeat = (
        1,  # bench_ufunc.UFunc.time_ufunc_types('divide')
    )
    case_methods = (
        'time_ufunc_types',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
    )


class Exp(_AggregateBenchmark):
    """Aggregate exp with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (27,)),  # bench_ufunc.UFunc.time_ufunc_types('exp')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (14, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (14, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 2, 'f')
    )
    run_repeat = (
        2,  # bench_ufunc.UFunc.time_ufunc_types('exp')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'exp'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'exp'>, 4, 2, 'f')
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
    """Aggregate floor_divide with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (1, 0)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, 100)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (1, 1)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, 10000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (1, 2)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, 1000000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (2, 0)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, 100)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (2, 1)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, 10000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (2, 2)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, 1000000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (4, 0)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, 100)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (4, 1)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, 10000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (4, 2)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, 1000000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (0, 0)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, 100)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (0, 1)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, 10000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (0, 2)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, 1000000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (3, 0)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, 100)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (3, 1)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, 10000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (3, 2)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, 1000000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (6, 0)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint16'>, 100)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (6, 1)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint16'>, 10000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (6, 2)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint16'>, 1000000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (7, 0)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint32'>, 100)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (7, 1)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint32'>, 10000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (7, 2)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint32'>, 1000000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (8, 0)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint64'>, 100)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (8, 1)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint64'>, 10000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (8, 2)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint64'>, 1000000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (5, 0)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint8'>, 100)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (5, 1)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint8'>, 10000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (5, 2)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint8'>, 1000000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (9, 0)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.ulonglong'>, 100)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (9, 1)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.ulonglong'>, 10000)
        _select_case_params(_Official_bench_ufunc_CustomArrayFloorDivideInt, 'time_floor_divide_int', (9, 2)),  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.ulonglong'>, 1000000)
    )
    run_repeat = (
        477,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, 100)
        19,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, 10000)
        1,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int16'>, 1000000)
        477,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, 100)
        20,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, 10000)
        1,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int32'>, 1000000)
        455,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, 100)
        16,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, 10000)
        1,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int64'>, 1000000)
        477,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, 100)
        20,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, 10000)
        1,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.int8'>, 1000000)
        455,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, 100)
        16,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, 10000)
        1,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.longlong'>, 1000000)
        556,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint16'>, 100)
        35,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint16'>, 10000)
        1,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint16'>, 1000000)
        527,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint32'>, 100)
        25,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint32'>, 10000)
        1,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint32'>, 1000000)
        477,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint64'>, 100)
        16,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint64'>, 10000)
        1,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint64'>, 1000000)
        556,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint8'>, 100)
        37,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint8'>, 10000)
        1,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.uint8'>, 1000000)
        477,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.ulonglong'>, 100)
        16,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.ulonglong'>, 10000)
        1,  # bench_ufunc.CustomArrayFloorDivideInt.time_floor_divide_int(<class 'numpy.ulonglong'>, 1000000)
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
    )
    case_types = (
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
        _Official_bench_ufunc_CustomArrayFloorDivideInt,
    )


class Log(_AggregateBenchmark):
    """Aggregate log with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (53,)),  # bench_ufunc.UFunc.time_ufunc_types('log')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (19, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (19, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (19, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (19, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log'>, 1, 1, 'f')
    )
    run_repeat = (
        2,  # bench_ufunc.UFunc.time_ufunc_types('log')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'log'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'log'>, 1, 1, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Maximum(_AggregateBenchmark):
    """Aggregate maximum with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (65,)),  # bench_ufunc.UFunc.time_ufunc_types('maximum')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFPSpecial, 'time_binary', (0, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryFPSpecial.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFPSpecial, 'time_binary', (0, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryFPSpecial.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFPSpecial, 'time_binary_scalar_in0', (0, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryFPSpecial.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFPSpecial, 'time_binary_scalar_in0', (0, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryFPSpecial.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary_scalar_in0', (0, 0, 0, 0, 1)),  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary_scalar_in0', (0, 0, 0, 0, 3)),  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary_scalar_in0', (0, 0, 0, 0, 5)),  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary_scalar_in0', (0, 0, 0, 0, 7)),  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary_scalar_in0', (0, 0, 0, 0, 9)),  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary_scalar_in0', (0, 0, 0, 0, 0)),  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary_scalar_in0', (0, 0, 0, 0, 2)),  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary_scalar_in0', (0, 0, 0, 0, 4)),  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary_scalar_in0', (0, 0, 0, 0, 6)),  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary_scalar_in0', (0, 0, 0, 0, 8)),  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'q')
    )
    run_repeat = (
        7,  # bench_ufunc.UFunc.time_ufunc_types('maximum')
        3,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'd')
        5,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'f')
        3,  # bench_ufunc_strides.BinaryFPSpecial.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'd')
        5,  # bench_ufunc_strides.BinaryFPSpecial.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFPSpecial.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'd')
        3,  # bench_ufunc_strides.BinaryFPSpecial.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'f')
        145,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'B')
        94,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'H')
        52,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'I')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'L')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'Q')
        143,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'b')
        93,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'h')
        51,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'i')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'l')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 1, 'q')
        20,  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'B')
        21,  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'H')
        20,  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'I')
        19,  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'L')
        19,  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'Q')
        15,  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'b')
        15,  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'h')
        20,  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'i')
        19,  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'l')
        19,  # bench_ufunc_strides.BinaryInt.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 1, 'q')
    )
    case_methods = (
        'time_ufunc_types',
        'time_binary',
        'time_binary',
        'time_binary_scalar_in0',
        'time_binary_scalar_in0',
        'time_binary',
        'time_binary',
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
        _Official_bench_ufunc_strides_BinaryFPSpecial,
        _Official_bench_ufunc_strides_BinaryFPSpecial,
        _Official_bench_ufunc_strides_BinaryFPSpecial,
        _Official_bench_ufunc_strides_BinaryFPSpecial,
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


class Multiply(_AggregateBenchmark):
    """Aggregate multiply with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (69,)),  # bench_ufunc.UFunc.time_ufunc_types('multiply')
    )
    run_repeat = (
        1,  # bench_ufunc.UFunc.time_ufunc_types('multiply')
    )
    case_methods = (
        'time_ufunc_types',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
    )


class Sin(_AggregateBenchmark):
    """Aggregate sin with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (83,)),  # bench_ufunc.UFunc.time_ufunc_types('sin')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (31, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (31, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 2, 'f')
    )
    run_repeat = (
        1,  # bench_ufunc.UFunc.time_ufunc_types('sin')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sin'>, 4, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 1, 2, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 1, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 2, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 2, 'e')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sin'>, 4, 2, 'f')
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


class Sqrt(_AggregateBenchmark):
    """Aggregate sqrt with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (86,)),  # bench_ufunc.UFunc.time_ufunc_types('sqrt')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_numpy_scalar', (1,)),  # bench_ufunc.UFuncSmall.time_ufunc_numpy_scalar('sqrt')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_python_float', (1,)),  # bench_ufunc.UFuncSmall.time_ufunc_python_float('sqrt')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_small_array', (1,)),  # bench_ufunc.UFuncSmall.time_ufunc_small_array('sqrt')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_small_array_inplace', (1,)),  # bench_ufunc.UFuncSmall.time_ufunc_small_array_inplace('sqrt')
        _select_case_params(_Official_bench_ufunc_UFuncSmall, 'time_ufunc_small_int_array', (1,)),  # bench_ufunc.UFuncSmall.time_ufunc_small_int_array('sqrt')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (33, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sqrt'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (33, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sqrt'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (33, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sqrt'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (33, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sqrt'>, 1, 1, 'f')
    )
    run_repeat = (
        3,  # bench_ufunc.UFunc.time_ufunc_types('sqrt')
        1000,  # bench_ufunc.UFuncSmall.time_ufunc_numpy_scalar('sqrt')
        1000,  # bench_ufunc.UFuncSmall.time_ufunc_python_float('sqrt')
        770,  # bench_ufunc.UFuncSmall.time_ufunc_small_array('sqrt')
        834,  # bench_ufunc.UFuncSmall.time_ufunc_small_array_inplace('sqrt')
        527,  # bench_ufunc.UFuncSmall.time_ufunc_small_int_array('sqrt')
        2,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sqrt'>, 1, 1, 'd')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'sqrt'>, 1, 1, 'f')
        2,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sqrt'>, 1, 1, 'd')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'sqrt'>, 1, 1, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_ufunc_numpy_scalar',
        'time_ufunc_python_float',
        'time_ufunc_small_array',
        'time_ufunc_small_array_inplace',
        'time_ufunc_small_int_array',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_UFuncSmall,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )


class Subtract(_AggregateBenchmark):
    """Aggregate subtract with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (88,)),  # bench_ufunc.UFunc.time_ufunc_types('subtract')
    )
    run_repeat = (
        1,  # bench_ufunc.UFunc.time_ufunc_types('subtract')
    )
    case_methods = (
        'time_ufunc_types',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
    )


class Tan(_AggregateBenchmark):
    """Aggregate tan with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_UFunc, 'time_ufunc_types', (89,)),  # bench_ufunc.UFunc.time_ufunc_types('tan')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (35, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'tan'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (35, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'tan'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (35, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'tan'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (35, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'tan'>, 1, 1, 'f')
    )
    run_repeat = (
        1,  # bench_ufunc.UFunc.time_ufunc_types('tan')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'tan'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc 'tan'>, 1, 1, 'f')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'tan'>, 1, 1, 'd')
        1,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc 'tan'>, 1, 1, 'f')
    )
    case_methods = (
        'time_ufunc_types',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
    )
    case_types = (
        _Official_bench_ufunc_UFunc,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFP,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
        _Official_bench_ufunc_strides_UnaryFPSpecial,
    )
