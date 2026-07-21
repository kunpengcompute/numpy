"""920b platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_core import StatsMethods as _Official_bench_core_StatsMethods
from ..bench_function_base import Mean as _Official_bench_function_base_Mean
from ..bench_reduce import (
    FMinMax as _Official_bench_reduce_FMinMax,
    StatsReductions as _Official_bench_reduce_StatsReductions,
)


class Max(_AggregateBenchmark):
    """Aggregate max with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_core_StatsMethods, 'time_max', (3, 0)),  # bench_core.StatsMethods.time_max('float64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_max', (3, 1)),  # bench_core.StatsMethods.time_max('float64', 10000)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_max', (0, 0)),  # bench_core.StatsMethods.time_max('int64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_max', (0, 1)),  # bench_core.StatsMethods.time_max('int64', 10000)
        _select_case_params(_Official_bench_reduce_FMinMax, 'time_max', (1,)),  # bench_reduce.FMinMax.time_max(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_max', (3,)),  # bench_reduce.StatsReductions.time_max('float64')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_max', (0,)),  # bench_reduce.StatsReductions.time_max('int64')
    )
    run_repeat = (
        385,  # bench_core.StatsMethods.time_max('float64', 100)
        223,  # bench_core.StatsMethods.time_max('float64', 10000)
        385,  # bench_core.StatsMethods.time_max('int64', 100)
        209,  # bench_core.StatsMethods.time_max('int64', 10000)
        176,  # bench_reduce.FMinMax.time_max(<class 'numpy.float64'>)
        257,  # bench_reduce.StatsReductions.time_max('float64')
        257,  # bench_reduce.StatsReductions.time_max('int64')
    )
    case_methods = (
        'time_max',
        'time_max',
        'time_max',
        'time_max',
        'time_max',
        'time_max',
        'time_max',
    )
    case_types = (
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_reduce_FMinMax,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
    )


class Mean(_AggregateBenchmark):
    """Aggregate mean with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_core_StatsMethods, 'time_mean', (5, 0)),  # bench_core.StatsMethods.time_mean('bool_', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_mean', (5, 1)),  # bench_core.StatsMethods.time_mean('bool_', 10000)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_mean', (2, 0)),  # bench_core.StatsMethods.time_mean('float32', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_mean', (2, 1)),  # bench_core.StatsMethods.time_mean('float32', 10000)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_mean', (3, 0)),  # bench_core.StatsMethods.time_mean('float64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_mean', (3, 1)),  # bench_core.StatsMethods.time_mean('float64', 10000)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_mean', (0, 0)),  # bench_core.StatsMethods.time_mean('int64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_mean', (0, 1)),  # bench_core.StatsMethods.time_mean('int64', 10000)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_mean', (1, 0)),  # bench_core.StatsMethods.time_mean('uint64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_mean', (1, 1)),  # bench_core.StatsMethods.time_mean('uint64', 10000)
        _select_case_params(_Official_bench_function_base_Mean, 'time_mean', (0,)),  # bench_function_base.Mean.time_mean(1)
        _select_case_params(_Official_bench_function_base_Mean, 'time_mean', (1,)),  # bench_function_base.Mean.time_mean(10)
        _select_case_params(_Official_bench_function_base_Mean, 'time_mean', (2,)),  # bench_function_base.Mean.time_mean(100000)
        _select_case_params(_Official_bench_function_base_Mean, 'time_mean_axis', (0,)),  # bench_function_base.Mean.time_mean_axis(1)
        _select_case_params(_Official_bench_function_base_Mean, 'time_mean_axis', (1,)),  # bench_function_base.Mean.time_mean_axis(10)
        _select_case_params(_Official_bench_function_base_Mean, 'time_mean_axis', (2,)),  # bench_function_base.Mean.time_mean_axis(100000)
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_mean', (5,)),  # bench_reduce.StatsReductions.time_mean('bool_')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_mean', (2,)),  # bench_reduce.StatsReductions.time_mean('float32')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_mean', (3,)),  # bench_reduce.StatsReductions.time_mean('float64')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_mean', (0,)),  # bench_reduce.StatsReductions.time_mean('int64')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_mean', (1,)),  # bench_reduce.StatsReductions.time_mean('uint64')
    )
    run_repeat = (
        170,  # bench_core.StatsMethods.time_mean('bool_', 100)
        92,  # bench_core.StatsMethods.time_mean('bool_', 10000)
        121,  # bench_core.StatsMethods.time_mean('float32', 100)
        91,  # bench_core.StatsMethods.time_mean('float32', 10000)
        193,  # bench_core.StatsMethods.time_mean('float64', 100)
        125,  # bench_core.StatsMethods.time_mean('float64', 10000)
        176,  # bench_core.StatsMethods.time_mean('int64', 100)
        91,  # bench_core.StatsMethods.time_mean('int64', 10000)
        173,  # bench_core.StatsMethods.time_mean('uint64', 100)
        91,  # bench_core.StatsMethods.time_mean('uint64', 10000)
        143,  # bench_function_base.Mean.time_mean(1)
        141,  # bench_function_base.Mean.time_mean(10)
        9,  # bench_function_base.Mean.time_mean(100000)
        129,  # bench_function_base.Mean.time_mean_axis(1)
        125,  # bench_function_base.Mean.time_mean_axis(10)
        8,  # bench_function_base.Mean.time_mean_axis(100000)
        148,  # bench_reduce.StatsReductions.time_mean('bool_')
        112,  # bench_reduce.StatsReductions.time_mean('float32')
        164,  # bench_reduce.StatsReductions.time_mean('float64')
        148,  # bench_reduce.StatsReductions.time_mean('int64')
        150,  # bench_reduce.StatsReductions.time_mean('uint64')
    )
    case_methods = (
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean_axis',
        'time_mean_axis',
        'time_mean_axis',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
        'time_mean',
    )
    case_types = (
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_function_base_Mean,
        _Official_bench_function_base_Mean,
        _Official_bench_function_base_Mean,
        _Official_bench_function_base_Mean,
        _Official_bench_function_base_Mean,
        _Official_bench_function_base_Mean,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
    )


class Std(_AggregateBenchmark):
    """Aggregate std with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_core_StatsMethods, 'time_std', (5, 0)),  # bench_core.StatsMethods.time_std('bool_', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_std', (5, 1)),  # bench_core.StatsMethods.time_std('bool_', 10000)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_std', (2, 0)),  # bench_core.StatsMethods.time_std('float32', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_std', (2, 1)),  # bench_core.StatsMethods.time_std('float32', 10000)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_std', (3, 0)),  # bench_core.StatsMethods.time_std('float64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_std', (3, 1)),  # bench_core.StatsMethods.time_std('float64', 10000)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_std', (0, 0)),  # bench_core.StatsMethods.time_std('int64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_std', (0, 1)),  # bench_core.StatsMethods.time_std('int64', 10000)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_std', (1, 0)),  # bench_core.StatsMethods.time_std('uint64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_std', (1, 1)),  # bench_core.StatsMethods.time_std('uint64', 10000)
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_std', (5,)),  # bench_reduce.StatsReductions.time_std('bool_')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_std', (2,)),  # bench_reduce.StatsReductions.time_std('float32')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_std', (3,)),  # bench_reduce.StatsReductions.time_std('float64')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_std', (0,)),  # bench_reduce.StatsReductions.time_std('int64')
        _select_case_params(_Official_bench_reduce_StatsReductions, 'time_std', (1,)),  # bench_reduce.StatsReductions.time_std('uint64')
    )
    run_repeat = (
        60,  # bench_core.StatsMethods.time_std('bool_', 100)
        28,  # bench_core.StatsMethods.time_std('bool_', 10000)
        57,  # bench_core.StatsMethods.time_std('float32', 100)
        40,  # bench_core.StatsMethods.time_std('float32', 10000)
        70,  # bench_core.StatsMethods.time_std('float64', 100)
        43,  # bench_core.StatsMethods.time_std('float64', 10000)
        63,  # bench_core.StatsMethods.time_std('int64', 100)
        35,  # bench_core.StatsMethods.time_std('int64', 10000)
        62,  # bench_core.StatsMethods.time_std('uint64', 100)
        34,  # bench_core.StatsMethods.time_std('uint64', 10000)
        54,  # bench_reduce.StatsReductions.time_std('bool_')
        54,  # bench_reduce.StatsReductions.time_std('float32')
        63,  # bench_reduce.StatsReductions.time_std('float64')
        58,  # bench_reduce.StatsReductions.time_std('int64')
        57,  # bench_reduce.StatsReductions.time_std('uint64')
    )
    case_methods = (
        'time_std',
        'time_std',
        'time_std',
        'time_std',
        'time_std',
        'time_std',
        'time_std',
        'time_std',
        'time_std',
        'time_std',
        'time_std',
        'time_std',
        'time_std',
        'time_std',
        'time_std',
    )
    case_types = (
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
        _Official_bench_reduce_StatsReductions,
    )


class Sum(_AggregateBenchmark):
    """Aggregate sum with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_core_StatsMethods, 'time_sum', (3, 0)),  # bench_core.StatsMethods.time_sum('float64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_sum', (3, 1)),  # bench_core.StatsMethods.time_sum('float64', 10000)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_sum', (0, 0)),  # bench_core.StatsMethods.time_sum('int64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_sum', (0, 1)),  # bench_core.StatsMethods.time_sum('int64', 10000)
    )
    run_repeat = (
        385,  # bench_core.StatsMethods.time_sum('float64', 100)
        189,  # bench_core.StatsMethods.time_sum('float64', 10000)
        385,  # bench_core.StatsMethods.time_sum('int64', 100)
        189,  # bench_core.StatsMethods.time_sum('int64', 10000)
    )
    case_methods = (
        'time_sum',
        'time_sum',
        'time_sum',
        'time_sum',
    )
    case_types = (
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
    )


class Var(_AggregateBenchmark):
    """Aggregate var with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_core_StatsMethods, 'time_var', (3, 0)),  # bench_core.StatsMethods.time_var('float64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_var', (3, 1)),  # bench_core.StatsMethods.time_var('float64', 10000)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_var', (0, 0)),  # bench_core.StatsMethods.time_var('int64', 100)
        _select_case_params(_Official_bench_core_StatsMethods, 'time_var', (0, 1)),  # bench_core.StatsMethods.time_var('int64', 10000)
    )
    run_repeat = (
        75,  # bench_core.StatsMethods.time_var('float64', 100)
        44,  # bench_core.StatsMethods.time_var('float64', 10000)
        66,  # bench_core.StatsMethods.time_var('int64', 100)
        35,  # bench_core.StatsMethods.time_var('int64', 10000)
    )
    case_methods = (
        'time_var',
        'time_var',
        'time_var',
        'time_var',
    )
    case_types = (
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
        _Official_bench_core_StatsMethods,
    )
