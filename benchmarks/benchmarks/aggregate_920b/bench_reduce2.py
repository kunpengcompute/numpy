"""920b platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_reduce import ArgMax as _Official_bench_reduce_ArgMax


class Argmax(_AggregateBenchmark):
    """Aggregate argmax with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_reduce_ArgMax, 'time_argmax', (10,)),  # bench_reduce.ArgMax.time_argmax(<class 'bool'>)
        _select_case_params(_Official_bench_reduce_ArgMax, 'time_argmax', (8,)),  # bench_reduce.ArgMax.time_argmax(<class 'numpy.float32'>)
        _select_case_params(_Official_bench_reduce_ArgMax, 'time_argmax', (9,)),  # bench_reduce.ArgMax.time_argmax(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_reduce_ArgMax, 'time_argmax', (2,)),  # bench_reduce.ArgMax.time_argmax(<class 'numpy.int16'>)
        _select_case_params(_Official_bench_reduce_ArgMax, 'time_argmax', (4,)),  # bench_reduce.ArgMax.time_argmax(<class 'numpy.int32'>)
        _select_case_params(_Official_bench_reduce_ArgMax, 'time_argmax', (6,)),  # bench_reduce.ArgMax.time_argmax(<class 'numpy.int64'>)
        _select_case_params(_Official_bench_reduce_ArgMax, 'time_argmax', (0,)),  # bench_reduce.ArgMax.time_argmax(<class 'numpy.int8'>)
        _select_case_params(_Official_bench_reduce_ArgMax, 'time_argmax', (3,)),  # bench_reduce.ArgMax.time_argmax(<class 'numpy.uint16'>)
        _select_case_params(_Official_bench_reduce_ArgMax, 'time_argmax', (5,)),  # bench_reduce.ArgMax.time_argmax(<class 'numpy.uint32'>)
        _select_case_params(_Official_bench_reduce_ArgMax, 'time_argmax', (7,)),  # bench_reduce.ArgMax.time_argmax(<class 'numpy.uint64'>)
        _select_case_params(_Official_bench_reduce_ArgMax, 'time_argmax', (1,)),  # bench_reduce.ArgMax.time_argmax(<class 'numpy.uint8'>)
    )
    run_repeat = (
        130,  # bench_reduce.ArgMax.time_argmax(<class 'bool'>)
        21,  # bench_reduce.ArgMax.time_argmax(<class 'numpy.float32'>)
        10,  # bench_reduce.ArgMax.time_argmax(<class 'numpy.float64'>)
        65,  # bench_reduce.ArgMax.time_argmax(<class 'numpy.int16'>)
        36,  # bench_reduce.ArgMax.time_argmax(<class 'numpy.int32'>)
        18,  # bench_reduce.ArgMax.time_argmax(<class 'numpy.int64'>)
        110,  # bench_reduce.ArgMax.time_argmax(<class 'numpy.int8'>)
        64,  # bench_reduce.ArgMax.time_argmax(<class 'numpy.uint16'>)
        36,  # bench_reduce.ArgMax.time_argmax(<class 'numpy.uint32'>)
        18,  # bench_reduce.ArgMax.time_argmax(<class 'numpy.uint64'>)
        110,  # bench_reduce.ArgMax.time_argmax(<class 'numpy.uint8'>)
    )
    case_methods = (
        'time_argmax',
        'time_argmax',
        'time_argmax',
        'time_argmax',
        'time_argmax',
        'time_argmax',
        'time_argmax',
        'time_argmax',
        'time_argmax',
        'time_argmax',
        'time_argmax',
    )
    case_types = (
        _Official_bench_reduce_ArgMax,
        _Official_bench_reduce_ArgMax,
        _Official_bench_reduce_ArgMax,
        _Official_bench_reduce_ArgMax,
        _Official_bench_reduce_ArgMax,
        _Official_bench_reduce_ArgMax,
        _Official_bench_reduce_ArgMax,
        _Official_bench_reduce_ArgMax,
        _Official_bench_reduce_ArgMax,
        _Official_bench_reduce_ArgMax,
        _Official_bench_reduce_ArgMax,
    )
