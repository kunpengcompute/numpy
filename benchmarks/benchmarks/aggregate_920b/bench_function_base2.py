"""920b platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_function_base import (
    Histogram1D as _Official_bench_function_base_Histogram1D,
    Histogram2D as _Official_bench_function_base_Histogram2D,
    Linspace as _Official_bench_function_base_Linspace,
    Partition as _Official_bench_function_base_Partition,
    Percentile as _Official_bench_function_base_Percentile,
    Sort as _Official_bench_function_base_Sort,
    SortWorst as _Official_bench_function_base_SortWorst,
)


class Argpartition(_AggregateBenchmark):
    """Aggregate argpartition with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 1, 0)),  # bench_function_base.Partition.time_argpartition('float64', ('ordered',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 1, 1)),  # bench_function_base.Partition.time_argpartition('float64', ('ordered',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 1, 2)),  # bench_function_base.Partition.time_argpartition('float64', ('ordered',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 0, 0)),  # bench_function_base.Partition.time_argpartition('float64', ('random',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 0, 1)),  # bench_function_base.Partition.time_argpartition('float64', ('random',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 0, 2)),  # bench_function_base.Partition.time_argpartition('float64', ('random',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 2, 0)),  # bench_function_base.Partition.time_argpartition('float64', ('reversed',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 2, 1)),  # bench_function_base.Partition.time_argpartition('float64', ('reversed',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 2, 2)),  # bench_function_base.Partition.time_argpartition('float64', ('reversed',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 4, 0)),  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 10), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 4, 1)),  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 10), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 4, 2)),  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 10), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 5, 0)),  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 100), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 5, 1)),  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 100), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 5, 2)),  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 100), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 6, 0)),  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 1000), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 6, 1)),  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 1000), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 6, 2)),  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 1000), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 3, 0)),  # bench_function_base.Partition.time_argpartition('float64', ('uniform',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 3, 1)),  # bench_function_base.Partition.time_argpartition('float64', ('uniform',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (0, 3, 2)),  # bench_function_base.Partition.time_argpartition('float64', ('uniform',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 1, 0)),  # bench_function_base.Partition.time_argpartition('int64', ('ordered',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 1, 1)),  # bench_function_base.Partition.time_argpartition('int64', ('ordered',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 1, 2)),  # bench_function_base.Partition.time_argpartition('int64', ('ordered',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 0, 0)),  # bench_function_base.Partition.time_argpartition('int64', ('random',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 0, 1)),  # bench_function_base.Partition.time_argpartition('int64', ('random',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 0, 2)),  # bench_function_base.Partition.time_argpartition('int64', ('random',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 2, 0)),  # bench_function_base.Partition.time_argpartition('int64', ('reversed',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 2, 1)),  # bench_function_base.Partition.time_argpartition('int64', ('reversed',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 2, 2)),  # bench_function_base.Partition.time_argpartition('int64', ('reversed',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 4, 0)),  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 10), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 4, 1)),  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 10), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 4, 2)),  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 10), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 5, 0)),  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 100), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 5, 1)),  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 100), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 5, 2)),  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 100), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 6, 0)),  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 1000), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 6, 1)),  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 1000), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 6, 2)),  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 1000), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 3, 0)),  # bench_function_base.Partition.time_argpartition('int64', ('uniform',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 3, 1)),  # bench_function_base.Partition.time_argpartition('int64', ('uniform',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_argpartition', (1, 3, 2)),  # bench_function_base.Partition.time_argpartition('int64', ('uniform',), 1000)
    )
    run_repeat = (
        6,  # bench_function_base.Partition.time_argpartition('float64', ('ordered',), 10)
        6,  # bench_function_base.Partition.time_argpartition('float64', ('ordered',), 100)
        6,  # bench_function_base.Partition.time_argpartition('float64', ('ordered',), 1000)
        1,  # bench_function_base.Partition.time_argpartition('float64', ('random',), 10)
        1,  # bench_function_base.Partition.time_argpartition('float64', ('random',), 100)
        1,  # bench_function_base.Partition.time_argpartition('float64', ('random',), 1000)
        1,  # bench_function_base.Partition.time_argpartition('float64', ('reversed',), 10)
        1,  # bench_function_base.Partition.time_argpartition('float64', ('reversed',), 100)
        1,  # bench_function_base.Partition.time_argpartition('float64', ('reversed',), 1000)
        3,  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 10), 10)
        3,  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 10), 100)
        2,  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 10), 1000)
        4,  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 100), 10)
        4,  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 100), 100)
        4,  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 100), 1000)
        4,  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 1000), 10)
        4,  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 1000), 100)
        3,  # bench_function_base.Partition.time_argpartition('float64', ('sorted_block', 1000), 1000)
        5,  # bench_function_base.Partition.time_argpartition('float64', ('uniform',), 10)
        5,  # bench_function_base.Partition.time_argpartition('float64', ('uniform',), 100)
        5,  # bench_function_base.Partition.time_argpartition('float64', ('uniform',), 1000)
        8,  # bench_function_base.Partition.time_argpartition('int64', ('ordered',), 10)
        8,  # bench_function_base.Partition.time_argpartition('int64', ('ordered',), 100)
        8,  # bench_function_base.Partition.time_argpartition('int64', ('ordered',), 1000)
        2,  # bench_function_base.Partition.time_argpartition('int64', ('random',), 10)
        2,  # bench_function_base.Partition.time_argpartition('int64', ('random',), 100)
        2,  # bench_function_base.Partition.time_argpartition('int64', ('random',), 1000)
        1,  # bench_function_base.Partition.time_argpartition('int64', ('reversed',), 10)
        1,  # bench_function_base.Partition.time_argpartition('int64', ('reversed',), 100)
        1,  # bench_function_base.Partition.time_argpartition('int64', ('reversed',), 1000)
        3,  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 10), 10)
        3,  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 10), 100)
        3,  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 10), 1000)
        5,  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 100), 10)
        5,  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 100), 100)
        5,  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 100), 1000)
        5,  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 1000), 10)
        5,  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 1000), 100)
        3,  # bench_function_base.Partition.time_argpartition('int64', ('sorted_block', 1000), 1000)
        6,  # bench_function_base.Partition.time_argpartition('int64', ('uniform',), 10)
        6,  # bench_function_base.Partition.time_argpartition('int64', ('uniform',), 100)
        6,  # bench_function_base.Partition.time_argpartition('int64', ('uniform',), 1000)
    )
    case_methods = (
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
        'time_argpartition',
    )
    case_types = (
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
    )


class ArgsortHeap(_AggregateBenchmark):
    """Aggregate argsort-heap with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 6, 2)),  # bench_function_base.Sort.time_argsort('heap', 'float16', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 2, 1)),  # bench_function_base.Sort.time_argsort('heap', 'float32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 2, 0)),  # bench_function_base.Sort.time_argsort('heap', 'float32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 2, 2)),  # bench_function_base.Sort.time_argsort('heap', 'float32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 2, 4)),  # bench_function_base.Sort.time_argsort('heap', 'float32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 2, 5)),  # bench_function_base.Sort.time_argsort('heap', 'float32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 2, 6)),  # bench_function_base.Sort.time_argsort('heap', 'float32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 2, 3)),  # bench_function_base.Sort.time_argsort('heap', 'float32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 0, 1)),  # bench_function_base.Sort.time_argsort('heap', 'float64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 0, 0)),  # bench_function_base.Sort.time_argsort('heap', 'float64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 0, 2)),  # bench_function_base.Sort.time_argsort('heap', 'float64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 0, 4)),  # bench_function_base.Sort.time_argsort('heap', 'float64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 0, 5)),  # bench_function_base.Sort.time_argsort('heap', 'float64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 0, 6)),  # bench_function_base.Sort.time_argsort('heap', 'float64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 0, 3)),  # bench_function_base.Sort.time_argsort('heap', 'float64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 5, 1)),  # bench_function_base.Sort.time_argsort('heap', 'int16', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 5, 0)),  # bench_function_base.Sort.time_argsort('heap', 'int16', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 5, 2)),  # bench_function_base.Sort.time_argsort('heap', 'int16', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 5, 4)),  # bench_function_base.Sort.time_argsort('heap', 'int16', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 5, 5)),  # bench_function_base.Sort.time_argsort('heap', 'int16', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 5, 6)),  # bench_function_base.Sort.time_argsort('heap', 'int16', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 5, 3)),  # bench_function_base.Sort.time_argsort('heap', 'int16', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 4, 1)),  # bench_function_base.Sort.time_argsort('heap', 'int32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 4, 0)),  # bench_function_base.Sort.time_argsort('heap', 'int32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 4, 2)),  # bench_function_base.Sort.time_argsort('heap', 'int32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 4, 4)),  # bench_function_base.Sort.time_argsort('heap', 'int32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 4, 5)),  # bench_function_base.Sort.time_argsort('heap', 'int32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 4, 6)),  # bench_function_base.Sort.time_argsort('heap', 'int32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 4, 3)),  # bench_function_base.Sort.time_argsort('heap', 'int32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 1, 1)),  # bench_function_base.Sort.time_argsort('heap', 'int64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 1, 0)),  # bench_function_base.Sort.time_argsort('heap', 'int64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 1, 2)),  # bench_function_base.Sort.time_argsort('heap', 'int64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 1, 4)),  # bench_function_base.Sort.time_argsort('heap', 'int64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 1, 5)),  # bench_function_base.Sort.time_argsort('heap', 'int64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 1, 6)),  # bench_function_base.Sort.time_argsort('heap', 'int64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 1, 3)),  # bench_function_base.Sort.time_argsort('heap', 'int64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 3, 1)),  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 3, 0)),  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 3, 2)),  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 3, 4)),  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 3, 5)),  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 3, 6)),  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (2, 3, 3)),  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('uniform',))
    )
    run_repeat = (
        9,  # bench_function_base.Sort.time_argsort('heap', 'float16', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float32', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float32', ('random',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float32', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float32', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float32', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float64', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float64', ('random',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float64', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float64', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('heap', 'float64', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int16', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int16', ('random',))
        11,  # bench_function_base.Sort.time_argsort('heap', 'int16', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int16', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int16', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int16', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int16', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int32', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int32', ('random',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int32', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int32', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int32', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int64', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int64', ('random',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int64', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int64', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('heap', 'int64', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('random',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('heap', 'uint32', ('uniform',))
    )
    case_methods = (
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
    )
    case_types = (
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
    )


class ArgsortMerge(_AggregateBenchmark):
    """Aggregate argsort-merge with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 6, 2)),  # bench_function_base.Sort.time_argsort('merge', 'float16', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 2, 1)),  # bench_function_base.Sort.time_argsort('merge', 'float32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 2, 0)),  # bench_function_base.Sort.time_argsort('merge', 'float32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 2, 2)),  # bench_function_base.Sort.time_argsort('merge', 'float32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 2, 4)),  # bench_function_base.Sort.time_argsort('merge', 'float32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 2, 5)),  # bench_function_base.Sort.time_argsort('merge', 'float32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 2, 6)),  # bench_function_base.Sort.time_argsort('merge', 'float32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 2, 3)),  # bench_function_base.Sort.time_argsort('merge', 'float32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 0, 1)),  # bench_function_base.Sort.time_argsort('merge', 'float64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 0, 0)),  # bench_function_base.Sort.time_argsort('merge', 'float64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 0, 2)),  # bench_function_base.Sort.time_argsort('merge', 'float64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 0, 4)),  # bench_function_base.Sort.time_argsort('merge', 'float64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 0, 5)),  # bench_function_base.Sort.time_argsort('merge', 'float64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 0, 6)),  # bench_function_base.Sort.time_argsort('merge', 'float64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 0, 3)),  # bench_function_base.Sort.time_argsort('merge', 'float64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 5, 1)),  # bench_function_base.Sort.time_argsort('merge', 'int16', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 5, 0)),  # bench_function_base.Sort.time_argsort('merge', 'int16', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 5, 2)),  # bench_function_base.Sort.time_argsort('merge', 'int16', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 5, 4)),  # bench_function_base.Sort.time_argsort('merge', 'int16', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 5, 5)),  # bench_function_base.Sort.time_argsort('merge', 'int16', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 5, 6)),  # bench_function_base.Sort.time_argsort('merge', 'int16', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 5, 3)),  # bench_function_base.Sort.time_argsort('merge', 'int16', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 4, 1)),  # bench_function_base.Sort.time_argsort('merge', 'int32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 4, 0)),  # bench_function_base.Sort.time_argsort('merge', 'int32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 4, 2)),  # bench_function_base.Sort.time_argsort('merge', 'int32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 4, 4)),  # bench_function_base.Sort.time_argsort('merge', 'int32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 4, 5)),  # bench_function_base.Sort.time_argsort('merge', 'int32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 4, 6)),  # bench_function_base.Sort.time_argsort('merge', 'int32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 4, 3)),  # bench_function_base.Sort.time_argsort('merge', 'int32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 1, 1)),  # bench_function_base.Sort.time_argsort('merge', 'int64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 1, 0)),  # bench_function_base.Sort.time_argsort('merge', 'int64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 1, 2)),  # bench_function_base.Sort.time_argsort('merge', 'int64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 1, 4)),  # bench_function_base.Sort.time_argsort('merge', 'int64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 1, 5)),  # bench_function_base.Sort.time_argsort('merge', 'int64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 1, 6)),  # bench_function_base.Sort.time_argsort('merge', 'int64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 1, 3)),  # bench_function_base.Sort.time_argsort('merge', 'int64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 3, 1)),  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 3, 0)),  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 3, 2)),  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 3, 4)),  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 3, 5)),  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 3, 6)),  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (1, 3, 3)),  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('uniform',))
    )
    run_repeat = (
        53,  # bench_function_base.Sort.time_argsort('merge', 'float16', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float32', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float32', ('random',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float32', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float32', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float32', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float64', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float64', ('random',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float64', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float64', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('merge', 'float64', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int16', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int16', ('random',))
        19,  # bench_function_base.Sort.time_argsort('merge', 'int16', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int16', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int16', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int16', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int16', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int32', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int32', ('random',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int32', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int32', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int32', ('uniform',))
        2,  # bench_function_base.Sort.time_argsort('merge', 'int64', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int64', ('random',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int64', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('merge', 'int64', ('sorted_block', 1000))
        2,  # bench_function_base.Sort.time_argsort('merge', 'int64', ('uniform',))
        2,  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('random',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('sorted_block', 1000))
        2,  # bench_function_base.Sort.time_argsort('merge', 'uint32', ('uniform',))
    )
    case_methods = (
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
    )
    case_types = (
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
    )


class ArgsortQuick(_AggregateBenchmark):
    """Aggregate argsort-quick with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 6, 2)),  # bench_function_base.Sort.time_argsort('quick', 'float16', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 2, 1)),  # bench_function_base.Sort.time_argsort('quick', 'float32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 2, 0)),  # bench_function_base.Sort.time_argsort('quick', 'float32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 2, 2)),  # bench_function_base.Sort.time_argsort('quick', 'float32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 2, 4)),  # bench_function_base.Sort.time_argsort('quick', 'float32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 2, 5)),  # bench_function_base.Sort.time_argsort('quick', 'float32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 2, 6)),  # bench_function_base.Sort.time_argsort('quick', 'float32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 2, 3)),  # bench_function_base.Sort.time_argsort('quick', 'float32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 0, 1)),  # bench_function_base.Sort.time_argsort('quick', 'float64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 0, 0)),  # bench_function_base.Sort.time_argsort('quick', 'float64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 0, 2)),  # bench_function_base.Sort.time_argsort('quick', 'float64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 0, 4)),  # bench_function_base.Sort.time_argsort('quick', 'float64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 0, 5)),  # bench_function_base.Sort.time_argsort('quick', 'float64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 0, 6)),  # bench_function_base.Sort.time_argsort('quick', 'float64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 0, 3)),  # bench_function_base.Sort.time_argsort('quick', 'float64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 5, 1)),  # bench_function_base.Sort.time_argsort('quick', 'int16', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 5, 0)),  # bench_function_base.Sort.time_argsort('quick', 'int16', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 5, 2)),  # bench_function_base.Sort.time_argsort('quick', 'int16', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 5, 4)),  # bench_function_base.Sort.time_argsort('quick', 'int16', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 5, 5)),  # bench_function_base.Sort.time_argsort('quick', 'int16', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 5, 6)),  # bench_function_base.Sort.time_argsort('quick', 'int16', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 5, 3)),  # bench_function_base.Sort.time_argsort('quick', 'int16', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 4, 1)),  # bench_function_base.Sort.time_argsort('quick', 'int32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 4, 0)),  # bench_function_base.Sort.time_argsort('quick', 'int32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 4, 2)),  # bench_function_base.Sort.time_argsort('quick', 'int32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 4, 4)),  # bench_function_base.Sort.time_argsort('quick', 'int32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 4, 5)),  # bench_function_base.Sort.time_argsort('quick', 'int32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 4, 6)),  # bench_function_base.Sort.time_argsort('quick', 'int32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 4, 3)),  # bench_function_base.Sort.time_argsort('quick', 'int32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 1, 1)),  # bench_function_base.Sort.time_argsort('quick', 'int64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 1, 0)),  # bench_function_base.Sort.time_argsort('quick', 'int64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 1, 2)),  # bench_function_base.Sort.time_argsort('quick', 'int64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 1, 4)),  # bench_function_base.Sort.time_argsort('quick', 'int64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 1, 5)),  # bench_function_base.Sort.time_argsort('quick', 'int64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 1, 6)),  # bench_function_base.Sort.time_argsort('quick', 'int64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 1, 3)),  # bench_function_base.Sort.time_argsort('quick', 'int64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 3, 1)),  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 3, 0)),  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 3, 2)),  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 3, 4)),  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 3, 5)),  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 3, 6)),  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_argsort', (0, 3, 3)),  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('uniform',))
    )
    run_repeat = (
        9,  # bench_function_base.Sort.time_argsort('quick', 'float16', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float32', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float32', ('random',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float32', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float32', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float32', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float64', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float64', ('random',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float64', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float64', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('quick', 'float64', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int16', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int16', ('random',))
        11,  # bench_function_base.Sort.time_argsort('quick', 'int16', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int16', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int16', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int16', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int16', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int32', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int32', ('random',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int32', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int32', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int32', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int64', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int64', ('random',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int64', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int64', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('quick', 'int64', ('uniform',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('ordered',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('random',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('reversed',))
        1,  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_argsort('quick', 'uint32', ('uniform',))
    )
    case_methods = (
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
        'time_argsort',
    )
    case_types = (
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
    )


class Histogram1d(_AggregateBenchmark):
    """Aggregate histogram1d with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Histogram1D, 'time_fine_binning', ()),  # bench_function_base.Histogram1D.time_fine_binning
        _select_case_params(_Official_bench_function_base_Histogram1D, 'time_full_coverage', ()),  # bench_function_base.Histogram1D.time_full_coverage
        _select_case_params(_Official_bench_function_base_Histogram1D, 'time_small_coverage', ()),  # bench_function_base.Histogram1D.time_small_coverage
    )
    run_repeat = (
        1,  # bench_function_base.Histogram1D.time_fine_binning
        1,  # bench_function_base.Histogram1D.time_full_coverage
        6,  # bench_function_base.Histogram1D.time_small_coverage
    )
    case_methods = (
        'time_fine_binning',
        'time_full_coverage',
        'time_small_coverage',
    )
    case_types = (
        _Official_bench_function_base_Histogram1D,
        _Official_bench_function_base_Histogram1D,
        _Official_bench_function_base_Histogram1D,
    )


class Histogram2d(_AggregateBenchmark):
    """Aggregate histogram2d with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Histogram2D, 'time_fine_binning', ()),  # bench_function_base.Histogram2D.time_fine_binning
        _select_case_params(_Official_bench_function_base_Histogram2D, 'time_full_coverage', ()),  # bench_function_base.Histogram2D.time_full_coverage
        _select_case_params(_Official_bench_function_base_Histogram2D, 'time_small_coverage', ()),  # bench_function_base.Histogram2D.time_small_coverage
    )
    run_repeat = (
        1,  # bench_function_base.Histogram2D.time_fine_binning
        1,  # bench_function_base.Histogram2D.time_full_coverage
        1,  # bench_function_base.Histogram2D.time_small_coverage
    )
    case_methods = (
        'time_fine_binning',
        'time_full_coverage',
        'time_small_coverage',
    )
    case_types = (
        _Official_bench_function_base_Histogram2D,
        _Official_bench_function_base_Histogram2D,
        _Official_bench_function_base_Histogram2D,
    )


class Linspace(_AggregateBenchmark):
    """Aggregate linspace with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Linspace, 'time_linspace_array', ()),  # bench_function_base.Linspace.time_linspace_array
        _select_case_params(_Official_bench_function_base_Linspace, 'time_linspace_scalar', ()),  # bench_function_base.Linspace.time_linspace_scalar
    )
    run_repeat = (
        56,  # bench_function_base.Linspace.time_linspace_array
        109,  # bench_function_base.Linspace.time_linspace_scalar
    )
    case_methods = (
        'time_linspace_array',
        'time_linspace_scalar',
    )
    case_types = (
        _Official_bench_function_base_Linspace,
        _Official_bench_function_base_Linspace,
    )


class Partition(_AggregateBenchmark):
    """Aggregate partition with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 1, 0)),  # bench_function_base.Partition.time_partition('float64', ('ordered',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 1, 1)),  # bench_function_base.Partition.time_partition('float64', ('ordered',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 1, 2)),  # bench_function_base.Partition.time_partition('float64', ('ordered',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 0, 0)),  # bench_function_base.Partition.time_partition('float64', ('random',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 0, 1)),  # bench_function_base.Partition.time_partition('float64', ('random',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 0, 2)),  # bench_function_base.Partition.time_partition('float64', ('random',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 2, 0)),  # bench_function_base.Partition.time_partition('float64', ('reversed',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 2, 1)),  # bench_function_base.Partition.time_partition('float64', ('reversed',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 2, 2)),  # bench_function_base.Partition.time_partition('float64', ('reversed',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 4, 0)),  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 10), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 4, 1)),  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 10), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 4, 2)),  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 10), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 5, 0)),  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 100), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 5, 1)),  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 100), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 5, 2)),  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 100), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 6, 0)),  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 1000), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 6, 1)),  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 1000), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 6, 2)),  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 1000), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 3, 0)),  # bench_function_base.Partition.time_partition('float64', ('uniform',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 3, 1)),  # bench_function_base.Partition.time_partition('float64', ('uniform',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (0, 3, 2)),  # bench_function_base.Partition.time_partition('float64', ('uniform',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 1, 0)),  # bench_function_base.Partition.time_partition('int64', ('ordered',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 1, 1)),  # bench_function_base.Partition.time_partition('int64', ('ordered',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 1, 2)),  # bench_function_base.Partition.time_partition('int64', ('ordered',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 0, 0)),  # bench_function_base.Partition.time_partition('int64', ('random',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 0, 1)),  # bench_function_base.Partition.time_partition('int64', ('random',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 0, 2)),  # bench_function_base.Partition.time_partition('int64', ('random',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 2, 0)),  # bench_function_base.Partition.time_partition('int64', ('reversed',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 2, 1)),  # bench_function_base.Partition.time_partition('int64', ('reversed',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 2, 2)),  # bench_function_base.Partition.time_partition('int64', ('reversed',), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 4, 0)),  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 10), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 4, 1)),  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 10), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 4, 2)),  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 10), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 5, 0)),  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 100), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 5, 1)),  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 100), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 5, 2)),  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 100), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 6, 0)),  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 1000), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 6, 1)),  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 1000), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 6, 2)),  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 1000), 1000)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 3, 0)),  # bench_function_base.Partition.time_partition('int64', ('uniform',), 10)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 3, 1)),  # bench_function_base.Partition.time_partition('int64', ('uniform',), 100)
        _select_case_params(_Official_bench_function_base_Partition, 'time_partition', (1, 3, 2)),  # bench_function_base.Partition.time_partition('int64', ('uniform',), 1000)
    )
    run_repeat = (
        7,  # bench_function_base.Partition.time_partition('float64', ('ordered',), 10)
        7,  # bench_function_base.Partition.time_partition('float64', ('ordered',), 100)
        7,  # bench_function_base.Partition.time_partition('float64', ('ordered',), 1000)
        2,  # bench_function_base.Partition.time_partition('float64', ('random',), 10)
        2,  # bench_function_base.Partition.time_partition('float64', ('random',), 100)
        2,  # bench_function_base.Partition.time_partition('float64', ('random',), 1000)
        1,  # bench_function_base.Partition.time_partition('float64', ('reversed',), 10)
        1,  # bench_function_base.Partition.time_partition('float64', ('reversed',), 100)
        1,  # bench_function_base.Partition.time_partition('float64', ('reversed',), 1000)
        3,  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 10), 10)
        3,  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 10), 100)
        3,  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 10), 1000)
        4,  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 100), 10)
        4,  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 100), 100)
        4,  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 100), 1000)
        4,  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 1000), 10)
        4,  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 1000), 100)
        3,  # bench_function_base.Partition.time_partition('float64', ('sorted_block', 1000), 1000)
        5,  # bench_function_base.Partition.time_partition('float64', ('uniform',), 10)
        5,  # bench_function_base.Partition.time_partition('float64', ('uniform',), 100)
        5,  # bench_function_base.Partition.time_partition('float64', ('uniform',), 1000)
        10,  # bench_function_base.Partition.time_partition('int64', ('ordered',), 10)
        10,  # bench_function_base.Partition.time_partition('int64', ('ordered',), 100)
        10,  # bench_function_base.Partition.time_partition('int64', ('ordered',), 1000)
        2,  # bench_function_base.Partition.time_partition('int64', ('random',), 10)
        2,  # bench_function_base.Partition.time_partition('int64', ('random',), 100)
        2,  # bench_function_base.Partition.time_partition('int64', ('random',), 1000)
        2,  # bench_function_base.Partition.time_partition('int64', ('reversed',), 10)
        2,  # bench_function_base.Partition.time_partition('int64', ('reversed',), 100)
        2,  # bench_function_base.Partition.time_partition('int64', ('reversed',), 1000)
        4,  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 10), 10)
        4,  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 10), 100)
        4,  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 10), 1000)
        6,  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 100), 10)
        7,  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 100), 100)
        7,  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 100), 1000)
        6,  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 1000), 10)
        6,  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 1000), 100)
        4,  # bench_function_base.Partition.time_partition('int64', ('sorted_block', 1000), 1000)
        7,  # bench_function_base.Partition.time_partition('int64', ('uniform',), 10)
        7,  # bench_function_base.Partition.time_partition('int64', ('uniform',), 100)
        7,  # bench_function_base.Partition.time_partition('int64', ('uniform',), 1000)
    )
    case_methods = (
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
        'time_partition',
    )
    case_types = (
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
        _Official_bench_function_base_Partition,
    )


class Percentile(_AggregateBenchmark):
    """Aggregate percentile with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Percentile, 'time_percentile', ()),  # bench_function_base.Percentile.time_percentile
        _select_case_params(_Official_bench_function_base_Percentile, 'time_percentile_small', ()),  # bench_function_base.Percentile.time_percentile_small
        _select_case_params(_Official_bench_function_base_Percentile, 'time_quartile', ()),  # bench_function_base.Percentile.time_quartile
    )
    run_repeat = (
        8,  # bench_function_base.Percentile.time_percentile
        18,  # bench_function_base.Percentile.time_percentile_small
        8,  # bench_function_base.Percentile.time_quartile
    )
    case_methods = (
        'time_percentile',
        'time_percentile_small',
        'time_quartile',
    )
    case_types = (
        _Official_bench_function_base_Percentile,
        _Official_bench_function_base_Percentile,
        _Official_bench_function_base_Percentile,
    )


class SortHeap(_AggregateBenchmark):
    """Aggregate sort-heap with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 2, 1)),  # bench_function_base.Sort.time_sort('heap', 'float32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 2, 0)),  # bench_function_base.Sort.time_sort('heap', 'float32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 2, 2)),  # bench_function_base.Sort.time_sort('heap', 'float32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 2, 4)),  # bench_function_base.Sort.time_sort('heap', 'float32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 2, 5)),  # bench_function_base.Sort.time_sort('heap', 'float32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 2, 6)),  # bench_function_base.Sort.time_sort('heap', 'float32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 2, 3)),  # bench_function_base.Sort.time_sort('heap', 'float32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 0, 1)),  # bench_function_base.Sort.time_sort('heap', 'float64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 0, 0)),  # bench_function_base.Sort.time_sort('heap', 'float64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 0, 2)),  # bench_function_base.Sort.time_sort('heap', 'float64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 0, 4)),  # bench_function_base.Sort.time_sort('heap', 'float64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 0, 5)),  # bench_function_base.Sort.time_sort('heap', 'float64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 0, 6)),  # bench_function_base.Sort.time_sort('heap', 'float64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 0, 3)),  # bench_function_base.Sort.time_sort('heap', 'float64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 5, 1)),  # bench_function_base.Sort.time_sort('heap', 'int16', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 5, 0)),  # bench_function_base.Sort.time_sort('heap', 'int16', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 5, 4)),  # bench_function_base.Sort.time_sort('heap', 'int16', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 5, 5)),  # bench_function_base.Sort.time_sort('heap', 'int16', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 5, 6)),  # bench_function_base.Sort.time_sort('heap', 'int16', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 5, 3)),  # bench_function_base.Sort.time_sort('heap', 'int16', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 4, 1)),  # bench_function_base.Sort.time_sort('heap', 'int32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 4, 0)),  # bench_function_base.Sort.time_sort('heap', 'int32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 4, 2)),  # bench_function_base.Sort.time_sort('heap', 'int32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 4, 4)),  # bench_function_base.Sort.time_sort('heap', 'int32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 4, 5)),  # bench_function_base.Sort.time_sort('heap', 'int32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 4, 6)),  # bench_function_base.Sort.time_sort('heap', 'int32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 4, 3)),  # bench_function_base.Sort.time_sort('heap', 'int32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 1, 1)),  # bench_function_base.Sort.time_sort('heap', 'int64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 1, 0)),  # bench_function_base.Sort.time_sort('heap', 'int64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 1, 2)),  # bench_function_base.Sort.time_sort('heap', 'int64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 1, 4)),  # bench_function_base.Sort.time_sort('heap', 'int64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 1, 5)),  # bench_function_base.Sort.time_sort('heap', 'int64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 1, 6)),  # bench_function_base.Sort.time_sort('heap', 'int64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 1, 3)),  # bench_function_base.Sort.time_sort('heap', 'int64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 3, 1)),  # bench_function_base.Sort.time_sort('heap', 'uint32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 3, 0)),  # bench_function_base.Sort.time_sort('heap', 'uint32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 3, 2)),  # bench_function_base.Sort.time_sort('heap', 'uint32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 3, 4)),  # bench_function_base.Sort.time_sort('heap', 'uint32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 3, 5)),  # bench_function_base.Sort.time_sort('heap', 'uint32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 3, 6)),  # bench_function_base.Sort.time_sort('heap', 'uint32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (2, 3, 3)),  # bench_function_base.Sort.time_sort('heap', 'uint32', ('uniform',))
    )
    run_repeat = (
        1,  # bench_function_base.Sort.time_sort('heap', 'float32', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('heap', 'float32', ('random',))
        1,  # bench_function_base.Sort.time_sort('heap', 'float32', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('heap', 'float32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('heap', 'float32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('heap', 'float32', ('sorted_block', 1000))
        2,  # bench_function_base.Sort.time_sort('heap', 'float32', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('heap', 'float64', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('heap', 'float64', ('random',))
        1,  # bench_function_base.Sort.time_sort('heap', 'float64', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('heap', 'float64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('heap', 'float64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('heap', 'float64', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_sort('heap', 'float64', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('heap', 'int16', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('heap', 'int16', ('random',))
        1,  # bench_function_base.Sort.time_sort('heap', 'int16', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('heap', 'int16', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('heap', 'int16', ('sorted_block', 1000))
        9,  # bench_function_base.Sort.time_sort('heap', 'int16', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('heap', 'int32', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('heap', 'int32', ('random',))
        1,  # bench_function_base.Sort.time_sort('heap', 'int32', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('heap', 'int32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('heap', 'int32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('heap', 'int32', ('sorted_block', 1000))
        5,  # bench_function_base.Sort.time_sort('heap', 'int32', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('heap', 'int64', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('heap', 'int64', ('random',))
        1,  # bench_function_base.Sort.time_sort('heap', 'int64', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('heap', 'int64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('heap', 'int64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('heap', 'int64', ('sorted_block', 1000))
        3,  # bench_function_base.Sort.time_sort('heap', 'int64', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('heap', 'uint32', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('heap', 'uint32', ('random',))
        1,  # bench_function_base.Sort.time_sort('heap', 'uint32', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('heap', 'uint32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('heap', 'uint32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('heap', 'uint32', ('sorted_block', 1000))
        5,  # bench_function_base.Sort.time_sort('heap', 'uint32', ('uniform',))
    )
    case_methods = (
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
    )
    case_types = (
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
    )


class SortMerge(_AggregateBenchmark):
    """Aggregate sort-merge with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 2, 1)),  # bench_function_base.Sort.time_sort('merge', 'float32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 2, 0)),  # bench_function_base.Sort.time_sort('merge', 'float32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 2, 2)),  # bench_function_base.Sort.time_sort('merge', 'float32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 2, 4)),  # bench_function_base.Sort.time_sort('merge', 'float32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 2, 5)),  # bench_function_base.Sort.time_sort('merge', 'float32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 2, 6)),  # bench_function_base.Sort.time_sort('merge', 'float32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 2, 3)),  # bench_function_base.Sort.time_sort('merge', 'float32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 0, 1)),  # bench_function_base.Sort.time_sort('merge', 'float64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 0, 0)),  # bench_function_base.Sort.time_sort('merge', 'float64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 0, 2)),  # bench_function_base.Sort.time_sort('merge', 'float64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 0, 4)),  # bench_function_base.Sort.time_sort('merge', 'float64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 0, 5)),  # bench_function_base.Sort.time_sort('merge', 'float64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 0, 6)),  # bench_function_base.Sort.time_sort('merge', 'float64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 0, 3)),  # bench_function_base.Sort.time_sort('merge', 'float64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 5, 1)),  # bench_function_base.Sort.time_sort('merge', 'int16', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 5, 0)),  # bench_function_base.Sort.time_sort('merge', 'int16', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 5, 4)),  # bench_function_base.Sort.time_sort('merge', 'int16', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 5, 5)),  # bench_function_base.Sort.time_sort('merge', 'int16', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 5, 6)),  # bench_function_base.Sort.time_sort('merge', 'int16', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 5, 3)),  # bench_function_base.Sort.time_sort('merge', 'int16', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 4, 1)),  # bench_function_base.Sort.time_sort('merge', 'int32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 4, 0)),  # bench_function_base.Sort.time_sort('merge', 'int32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 4, 2)),  # bench_function_base.Sort.time_sort('merge', 'int32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 4, 4)),  # bench_function_base.Sort.time_sort('merge', 'int32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 4, 5)),  # bench_function_base.Sort.time_sort('merge', 'int32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 4, 6)),  # bench_function_base.Sort.time_sort('merge', 'int32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 4, 3)),  # bench_function_base.Sort.time_sort('merge', 'int32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 1, 1)),  # bench_function_base.Sort.time_sort('merge', 'int64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 1, 0)),  # bench_function_base.Sort.time_sort('merge', 'int64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 1, 2)),  # bench_function_base.Sort.time_sort('merge', 'int64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 1, 4)),  # bench_function_base.Sort.time_sort('merge', 'int64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 1, 5)),  # bench_function_base.Sort.time_sort('merge', 'int64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 1, 6)),  # bench_function_base.Sort.time_sort('merge', 'int64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 1, 3)),  # bench_function_base.Sort.time_sort('merge', 'int64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 3, 1)),  # bench_function_base.Sort.time_sort('merge', 'uint32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 3, 0)),  # bench_function_base.Sort.time_sort('merge', 'uint32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 3, 2)),  # bench_function_base.Sort.time_sort('merge', 'uint32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 3, 4)),  # bench_function_base.Sort.time_sort('merge', 'uint32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 3, 5)),  # bench_function_base.Sort.time_sort('merge', 'uint32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 3, 6)),  # bench_function_base.Sort.time_sort('merge', 'uint32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (1, 3, 3)),  # bench_function_base.Sort.time_sort('merge', 'uint32', ('uniform',))
    )
    run_repeat = (
        2,  # bench_function_base.Sort.time_sort('merge', 'float32', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('merge', 'float32', ('random',))
        1,  # bench_function_base.Sort.time_sort('merge', 'float32', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('merge', 'float32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('merge', 'float32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('merge', 'float32', ('sorted_block', 1000))
        2,  # bench_function_base.Sort.time_sort('merge', 'float32', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('merge', 'float64', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('merge', 'float64', ('random',))
        1,  # bench_function_base.Sort.time_sort('merge', 'float64', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('merge', 'float64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('merge', 'float64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('merge', 'float64', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_sort('merge', 'float64', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('merge', 'int16', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('merge', 'int16', ('random',))
        1,  # bench_function_base.Sort.time_sort('merge', 'int16', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('merge', 'int16', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('merge', 'int16', ('sorted_block', 1000))
        2,  # bench_function_base.Sort.time_sort('merge', 'int16', ('uniform',))
        2,  # bench_function_base.Sort.time_sort('merge', 'int32', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('merge', 'int32', ('random',))
        2,  # bench_function_base.Sort.time_sort('merge', 'int32', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('merge', 'int32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('merge', 'int32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('merge', 'int32', ('sorted_block', 1000))
        2,  # bench_function_base.Sort.time_sort('merge', 'int32', ('uniform',))
        2,  # bench_function_base.Sort.time_sort('merge', 'int64', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('merge', 'int64', ('random',))
        1,  # bench_function_base.Sort.time_sort('merge', 'int64', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('merge', 'int64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('merge', 'int64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('merge', 'int64', ('sorted_block', 1000))
        2,  # bench_function_base.Sort.time_sort('merge', 'int64', ('uniform',))
        2,  # bench_function_base.Sort.time_sort('merge', 'uint32', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('merge', 'uint32', ('random',))
        2,  # bench_function_base.Sort.time_sort('merge', 'uint32', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('merge', 'uint32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('merge', 'uint32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('merge', 'uint32', ('sorted_block', 1000))
        2,  # bench_function_base.Sort.time_sort('merge', 'uint32', ('uniform',))
    )
    case_methods = (
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
    )
    case_types = (
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
    )


class SortQuick(_AggregateBenchmark):
    """Aggregate sort-quick with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 2, 1)),  # bench_function_base.Sort.time_sort('quick', 'float32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 2, 0)),  # bench_function_base.Sort.time_sort('quick', 'float32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 2, 2)),  # bench_function_base.Sort.time_sort('quick', 'float32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 2, 4)),  # bench_function_base.Sort.time_sort('quick', 'float32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 2, 5)),  # bench_function_base.Sort.time_sort('quick', 'float32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 2, 6)),  # bench_function_base.Sort.time_sort('quick', 'float32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 2, 3)),  # bench_function_base.Sort.time_sort('quick', 'float32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 0, 1)),  # bench_function_base.Sort.time_sort('quick', 'float64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 0, 0)),  # bench_function_base.Sort.time_sort('quick', 'float64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 0, 2)),  # bench_function_base.Sort.time_sort('quick', 'float64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 0, 4)),  # bench_function_base.Sort.time_sort('quick', 'float64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 0, 5)),  # bench_function_base.Sort.time_sort('quick', 'float64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 0, 6)),  # bench_function_base.Sort.time_sort('quick', 'float64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 0, 3)),  # bench_function_base.Sort.time_sort('quick', 'float64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 5, 1)),  # bench_function_base.Sort.time_sort('quick', 'int16', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 5, 0)),  # bench_function_base.Sort.time_sort('quick', 'int16', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 5, 4)),  # bench_function_base.Sort.time_sort('quick', 'int16', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 5, 5)),  # bench_function_base.Sort.time_sort('quick', 'int16', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 5, 6)),  # bench_function_base.Sort.time_sort('quick', 'int16', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 5, 3)),  # bench_function_base.Sort.time_sort('quick', 'int16', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 4, 1)),  # bench_function_base.Sort.time_sort('quick', 'int32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 4, 0)),  # bench_function_base.Sort.time_sort('quick', 'int32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 4, 2)),  # bench_function_base.Sort.time_sort('quick', 'int32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 4, 4)),  # bench_function_base.Sort.time_sort('quick', 'int32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 4, 5)),  # bench_function_base.Sort.time_sort('quick', 'int32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 4, 6)),  # bench_function_base.Sort.time_sort('quick', 'int32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 4, 3)),  # bench_function_base.Sort.time_sort('quick', 'int32', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 1, 1)),  # bench_function_base.Sort.time_sort('quick', 'int64', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 1, 0)),  # bench_function_base.Sort.time_sort('quick', 'int64', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 1, 2)),  # bench_function_base.Sort.time_sort('quick', 'int64', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 1, 4)),  # bench_function_base.Sort.time_sort('quick', 'int64', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 1, 5)),  # bench_function_base.Sort.time_sort('quick', 'int64', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 1, 6)),  # bench_function_base.Sort.time_sort('quick', 'int64', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 1, 3)),  # bench_function_base.Sort.time_sort('quick', 'int64', ('uniform',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 3, 1)),  # bench_function_base.Sort.time_sort('quick', 'uint32', ('ordered',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 3, 0)),  # bench_function_base.Sort.time_sort('quick', 'uint32', ('random',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 3, 2)),  # bench_function_base.Sort.time_sort('quick', 'uint32', ('reversed',))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 3, 4)),  # bench_function_base.Sort.time_sort('quick', 'uint32', ('sorted_block', 10))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 3, 5)),  # bench_function_base.Sort.time_sort('quick', 'uint32', ('sorted_block', 100))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 3, 6)),  # bench_function_base.Sort.time_sort('quick', 'uint32', ('sorted_block', 1000))
        _select_case_params(_Official_bench_function_base_Sort, 'time_sort', (0, 3, 3)),  # bench_function_base.Sort.time_sort('quick', 'uint32', ('uniform',))
        _select_case_params(_Official_bench_function_base_SortWorst, 'time_sort_worst', ()),  # bench_function_base.Sort.time_sort_worst
    )
    run_repeat = (
        1,  # bench_function_base.Sort.time_sort('quick', 'float32', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('quick', 'float32', ('random',))
        1,  # bench_function_base.Sort.time_sort('quick', 'float32', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('quick', 'float32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('quick', 'float32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('quick', 'float32', ('sorted_block', 1000))
        2,  # bench_function_base.Sort.time_sort('quick', 'float32', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('quick', 'float64', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('quick', 'float64', ('random',))
        1,  # bench_function_base.Sort.time_sort('quick', 'float64', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('quick', 'float64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('quick', 'float64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('quick', 'float64', ('sorted_block', 1000))
        1,  # bench_function_base.Sort.time_sort('quick', 'float64', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('quick', 'int16', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('quick', 'int16', ('random',))
        1,  # bench_function_base.Sort.time_sort('quick', 'int16', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('quick', 'int16', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('quick', 'int16', ('sorted_block', 1000))
        9,  # bench_function_base.Sort.time_sort('quick', 'int16', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('quick', 'int32', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('quick', 'int32', ('random',))
        1,  # bench_function_base.Sort.time_sort('quick', 'int32', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('quick', 'int32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('quick', 'int32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('quick', 'int32', ('sorted_block', 1000))
        5,  # bench_function_base.Sort.time_sort('quick', 'int32', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('quick', 'int64', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('quick', 'int64', ('random',))
        1,  # bench_function_base.Sort.time_sort('quick', 'int64', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('quick', 'int64', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('quick', 'int64', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('quick', 'int64', ('sorted_block', 1000))
        3,  # bench_function_base.Sort.time_sort('quick', 'int64', ('uniform',))
        1,  # bench_function_base.Sort.time_sort('quick', 'uint32', ('ordered',))
        1,  # bench_function_base.Sort.time_sort('quick', 'uint32', ('random',))
        1,  # bench_function_base.Sort.time_sort('quick', 'uint32', ('reversed',))
        1,  # bench_function_base.Sort.time_sort('quick', 'uint32', ('sorted_block', 10))
        1,  # bench_function_base.Sort.time_sort('quick', 'uint32', ('sorted_block', 100))
        1,  # bench_function_base.Sort.time_sort('quick', 'uint32', ('sorted_block', 1000))
        5,  # bench_function_base.Sort.time_sort('quick', 'uint32', ('uniform',))
        1,  # bench_function_base.Sort.time_sort_worst
    )
    case_methods = (
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort',
        'time_sort_worst',
    )
    case_types = (
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_Sort,
        _Official_bench_function_base_SortWorst,
    )
