"""950 platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_reduce import (
    AddReduce as _Official_bench_reduce_AddReduce,
    AddReduceSeparate as _Official_bench_reduce_AddReduceSeparate,
    AnyAll as _Official_bench_reduce_AnyAll,
    ArgMin as _Official_bench_reduce_ArgMin,
    FMinMax as _Official_bench_reduce_FMinMax,
    StatsReductions as _Official_bench_reduce_StatsReductions,
)


class AddReduce(_AggregateBenchmark):
    """Aggregate AddReduce with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_reduce_AddReduce, 'time_axis_0', ()),  # bench_reduce.AddReduce.time_axis_0
        _select_case_params(_Official_bench_reduce_AddReduce, 'time_axis_1', ()),  # bench_reduce.AddReduce.time_axis_1
        _select_case_params(_Official_bench_reduce_AddReduceSeparate, 'time_reduce', (0, 3)),  # bench_reduce.AddReduceSeparate.time_reduce(0, 'float32')
        _select_case_params(_Official_bench_reduce_AddReduceSeparate, 'time_reduce', (0, 5)),  # bench_reduce.AddReduceSeparate.time_reduce(0, 'float64')
        _select_case_params(_Official_bench_reduce_AddReduceSeparate, 'time_reduce', (0, 0)),  # bench_reduce.AddReduceSeparate.time_reduce(0, 'int16')
        _select_case_params(_Official_bench_reduce_AddReduceSeparate, 'time_reduce', (0, 2)),  # bench_reduce.AddReduceSeparate.time_reduce(0, 'int32')
        _select_case_params(_Official_bench_reduce_AddReduceSeparate, 'time_reduce', (0, 4)),  # bench_reduce.AddReduceSeparate.time_reduce(0, 'int64')
        _select_case_params(_Official_bench_reduce_AddReduceSeparate, 'time_reduce', (1, 3)),  # bench_reduce.AddReduceSeparate.time_reduce(1, 'float32')
        _select_case_params(_Official_bench_reduce_AddReduceSeparate, 'time_reduce', (1, 5)),  # bench_reduce.AddReduceSeparate.time_reduce(1, 'float64')
        _select_case_params(_Official_bench_reduce_AddReduceSeparate, 'time_reduce', (1, 0)),  # bench_reduce.AddReduceSeparate.time_reduce(1, 'int16')
        _select_case_params(_Official_bench_reduce_AddReduceSeparate, 'time_reduce', (1, 2)),  # bench_reduce.AddReduceSeparate.time_reduce(1, 'int32')
        _select_case_params(_Official_bench_reduce_AddReduceSeparate, 'time_reduce', (1, 4)),  # bench_reduce.AddReduceSeparate.time_reduce(1, 'int64')
    )
    run_repeat = (
        1,  # bench_reduce.AddReduce.time_axis_0
        1,  # bench_reduce.AddReduce.time_axis_1
        9,  # bench_reduce.AddReduceSeparate.time_reduce(0, 'float32')
        6,  # bench_reduce.AddReduceSeparate.time_reduce(0, 'float64')
        4,  # bench_reduce.AddReduceSeparate.time_reduce(0, 'int16')
        3,  # bench_reduce.AddReduceSeparate.time_reduce(0, 'int32')
        5,  # bench_reduce.AddReduceSeparate.time_reduce(0, 'int64')
        5,  # bench_reduce.AddReduceSeparate.time_reduce(1, 'float32')
        5,  # bench_reduce.AddReduceSeparate.time_reduce(1, 'float64')
        4,  # bench_reduce.AddReduceSeparate.time_reduce(1, 'int16')
        4,  # bench_reduce.AddReduceSeparate.time_reduce(1, 'int32')
        5,  # bench_reduce.AddReduceSeparate.time_reduce(1, 'int64')
    )
    case_methods = (
        'time_axis_0',
        'time_axis_1',
        'time_reduce',
        'time_reduce',
        'time_reduce',
        'time_reduce',
        'time_reduce',
        'time_reduce',
        'time_reduce',
        'time_reduce',
        'time_reduce',
        'time_reduce',
    )
    case_types = (
        _Official_bench_reduce_AddReduce,
        _Official_bench_reduce_AddReduce,
        _Official_bench_reduce_AddReduceSeparate,
        _Official_bench_reduce_AddReduceSeparate,
        _Official_bench_reduce_AddReduceSeparate,
        _Official_bench_reduce_AddReduceSeparate,
        _Official_bench_reduce_AddReduceSeparate,
        _Official_bench_reduce_AddReduceSeparate,
        _Official_bench_reduce_AddReduceSeparate,
        _Official_bench_reduce_AddReduceSeparate,
        _Official_bench_reduce_AddReduceSeparate,
        _Official_bench_reduce_AddReduceSeparate,
    )


class AnyAll(_AggregateBenchmark):
    """Aggregate AnyAll with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_reduce_AnyAll, 'time_all_fast', ()),  # bench_reduce.AnyAll.time_all_fast
        _select_case_params(_Official_bench_reduce_AnyAll, 'time_all_slow', ()),  # bench_reduce.AnyAll.time_all_slow
        _select_case_params(_Official_bench_reduce_AnyAll, 'time_any_fast', ()),  # bench_reduce.AnyAll.time_any_fast
        _select_case_params(_Official_bench_reduce_AnyAll, 'time_any_slow', ()),  # bench_reduce.AnyAll.time_any_slow
    )
    run_repeat = (
        910,  # bench_reduce.AnyAll.time_all_fast
        239,  # bench_reduce.AnyAll.time_all_slow
        910,  # bench_reduce.AnyAll.time_any_fast
        250,  # bench_reduce.AnyAll.time_any_slow
    )
    case_methods = (
        'time_all_fast',
        'time_all_slow',
        'time_any_fast',
        'time_any_slow',
    )
    case_types = (
        _Official_bench_reduce_AnyAll,
        _Official_bench_reduce_AnyAll,
        _Official_bench_reduce_AnyAll,
        _Official_bench_reduce_AnyAll,
    )


class Argmin(_AggregateBenchmark):
    """Aggregate argmin with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_reduce_ArgMin, 'time_argmin', (10,)),  # bench_reduce.ArgMin.time_argmin(<class 'bool'>)
        _select_case_params(_Official_bench_reduce_ArgMin, 'time_argmin', (8,)),  # bench_reduce.ArgMin.time_argmin(<class 'numpy.float32'>)
        _select_case_params(_Official_bench_reduce_ArgMin, 'time_argmin', (9,)),  # bench_reduce.ArgMin.time_argmin(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_reduce_ArgMin, 'time_argmin', (2,)),  # bench_reduce.ArgMin.time_argmin(<class 'numpy.int16'>)
        _select_case_params(_Official_bench_reduce_ArgMin, 'time_argmin', (4,)),  # bench_reduce.ArgMin.time_argmin(<class 'numpy.int32'>)
        _select_case_params(_Official_bench_reduce_ArgMin, 'time_argmin', (6,)),  # bench_reduce.ArgMin.time_argmin(<class 'numpy.int64'>)
        _select_case_params(_Official_bench_reduce_ArgMin, 'time_argmin', (0,)),  # bench_reduce.ArgMin.time_argmin(<class 'numpy.int8'>)
        _select_case_params(_Official_bench_reduce_ArgMin, 'time_argmin', (3,)),  # bench_reduce.ArgMin.time_argmin(<class 'numpy.uint16'>)
        _select_case_params(_Official_bench_reduce_ArgMin, 'time_argmin', (5,)),  # bench_reduce.ArgMin.time_argmin(<class 'numpy.uint32'>)
        _select_case_params(_Official_bench_reduce_ArgMin, 'time_argmin', (7,)),  # bench_reduce.ArgMin.time_argmin(<class 'numpy.uint64'>)
        _select_case_params(_Official_bench_reduce_ArgMin, 'time_argmin', (1,)),  # bench_reduce.ArgMin.time_argmin(<class 'numpy.uint8'>)
    )
    run_repeat = (
        167,  # bench_reduce.ArgMin.time_argmin(<class 'bool'>)
        25,  # bench_reduce.ArgMin.time_argmin(<class 'numpy.float32'>)
        11,  # bench_reduce.ArgMin.time_argmin(<class 'numpy.float64'>)
        80,  # bench_reduce.ArgMin.time_argmin(<class 'numpy.int16'>)
        40,  # bench_reduce.ArgMin.time_argmin(<class 'numpy.int32'>)
        19,  # bench_reduce.ArgMin.time_argmin(<class 'numpy.int64'>)
        141,  # bench_reduce.ArgMin.time_argmin(<class 'numpy.int8'>)
        80,  # bench_reduce.ArgMin.time_argmin(<class 'numpy.uint16'>)
        40,  # bench_reduce.ArgMin.time_argmin(<class 'numpy.uint32'>)
        20,  # bench_reduce.ArgMin.time_argmin(<class 'numpy.uint64'>)
        141,  # bench_reduce.ArgMin.time_argmin(<class 'numpy.uint8'>)
    )
    case_methods = (
        'time_argmin',
        'time_argmin',
        'time_argmin',
        'time_argmin',
        'time_argmin',
        'time_argmin',
        'time_argmin',
        'time_argmin',
        'time_argmin',
        'time_argmin',
        'time_argmin',
    )
    case_types = (
        _Official_bench_reduce_ArgMin,
        _Official_bench_reduce_ArgMin,
        _Official_bench_reduce_ArgMin,
        _Official_bench_reduce_ArgMin,
        _Official_bench_reduce_ArgMin,
        _Official_bench_reduce_ArgMin,
        _Official_bench_reduce_ArgMin,
        _Official_bench_reduce_ArgMin,
        _Official_bench_reduce_ArgMin,
        _Official_bench_reduce_ArgMin,
        _Official_bench_reduce_ArgMin,
    )


class Max(_AggregateBenchmark):
    """Aggregate max with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_reduce_FMinMax, 'time_max', (0,)),  # bench_reduce.FMinMax.time_max(<class 'numpy.float32'>)
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_max', (5,)),  # bench_reduce.StatsReductions.time_max('bool_')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_max', (2,)),  # bench_reduce.StatsReductions.time_max('float32')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_max', (1,)),  # bench_reduce.StatsReductions.time_max('uint64')
    )
    run_repeat = (
        435,  # bench_reduce.FMinMax.time_max(<class 'numpy.float32'>)
        477,  # bench_reduce.StatsReductions.time_max('bool_')
        477,  # bench_reduce.StatsReductions.time_max('float32')
        477,  # bench_reduce.StatsReductions.time_max('uint64')
    )
    case_methods = (
        'time_max',
        'time_max',
        'time_max',
        'time_max',
    )
    case_types = (
        _Official_bench_reduce_FMinMax,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
    )


class Min(_AggregateBenchmark):
    """Aggregate min with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_reduce_FMinMax, 'time_min', (0,)),  # bench_reduce.FMinMax.time_min(<class 'numpy.float32'>)
        _select_case_params(_Official_bench_reduce_FMinMax, 'time_min', (1,)),  # bench_reduce.FMinMax.time_min(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_min', (5,)),  # bench_reduce.StatsReductions.time_min('bool_')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_min', (2,)),  # bench_reduce.StatsReductions.time_min('float32')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_min', (3,)),  # bench_reduce.StatsReductions.time_min('float64')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_min', (0,)),  # bench_reduce.StatsReductions.time_min('int64')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_min', (1,)),  # bench_reduce.StatsReductions.time_min('uint64')
    )
    run_repeat = (
        417,  # bench_reduce.FMinMax.time_min(<class 'numpy.float32'>)
        295,  # bench_reduce.FMinMax.time_min(<class 'numpy.float64'>)
        500,  # bench_reduce.StatsReductions.time_min('bool_')
        477,  # bench_reduce.StatsReductions.time_min('float32')
        477,  # bench_reduce.StatsReductions.time_min('float64')
        477,  # bench_reduce.StatsReductions.time_min('int64')
        477,  # bench_reduce.StatsReductions.time_min('uint64')
    )
    case_methods = (
        'time_min',
        'time_min',
        'time_min',
        'time_min',
        'time_min',
        'time_min',
        'time_min',
    )
    case_types = (
        _Official_bench_reduce_FMinMax,
        _Official_bench_reduce_FMinMax,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
    )


class Prod(_AggregateBenchmark):
    """Aggregate prod with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_prod', (5,)),  # bench_reduce.StatsReductions.time_prod('bool_')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_prod', (2,)),  # bench_reduce.StatsReductions.time_prod('float32')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_prod', (3,)),  # bench_reduce.StatsReductions.time_prod('float64')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_prod', (0,)),  # bench_reduce.StatsReductions.time_prod('int64')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_prod', (1,)),  # bench_reduce.StatsReductions.time_prod('uint64')
    )
    run_repeat = (
        400,  # bench_reduce.StatsReductions.time_prod('bool_')
        455,  # bench_reduce.StatsReductions.time_prod('float32')
        435,  # bench_reduce.StatsReductions.time_prod('float64')
        435,  # bench_reduce.StatsReductions.time_prod('int64')
        455,  # bench_reduce.StatsReductions.time_prod('uint64')
    )
    case_methods = (
        'time_prod',
        'time_prod',
        'time_prod',
        'time_prod',
        'time_prod',
    )
    case_types = (
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
    )


class Var(_AggregateBenchmark):
    """Aggregate var with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_var', (5,)),  # bench_reduce.StatsReductions.time_var('bool_')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_var', (2,)),  # bench_reduce.StatsReductions.time_var('float32')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_var', (3,)),  # bench_reduce.StatsReductions.time_var('float64')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_var', (0,)),  # bench_reduce.StatsReductions.time_var('int64')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_var', (1,)),  # bench_reduce.StatsReductions.time_var('uint64')
    )
    run_repeat = (
        112,  # bench_reduce.StatsReductions.time_var('bool_')
        112,  # bench_reduce.StatsReductions.time_var('float32')
        130,  # bench_reduce.StatsReductions.time_var('float64')
        120,  # bench_reduce.StatsReductions.time_var('int64')
        120,  # bench_reduce.StatsReductions.time_var('uint64')
    )
    case_methods = (
        'time_var',
        'time_var',
        'time_var',
        'time_var',
        'time_var',
    )
    case_types = (
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
    )
