"""920b platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_indexing import (
    Indexing as _Official_bench_indexing_Indexing,
    IndexingSeparate as _Official_bench_indexing_IndexingSeparate,
    IndexingStructured0D as _Official_bench_indexing_IndexingStructured0D,
    IndexingWith1DArr as _Official_bench_indexing_IndexingWith1DArr,
)


class Getitem(_AggregateBenchmark):
    """Aggregate getitem with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (1, 8)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (1, 3)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'float32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (1, 5)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'float64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (1, 9)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'i,O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (1, 0)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'int16')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (1, 2)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'int32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (1, 4)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'int64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (2, 8)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (2, 3)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'float32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (2, 5)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'float64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (2, 9)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'i,O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (2, 0)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'int16')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (2, 2)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'int32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (2, 4)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'int64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (4, 8)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (4, 3)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'float32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (4, 5)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'float64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (4, 9)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'i,O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (4, 0)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'int16')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (4, 2)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'int32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (4, 4)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'int64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (0, 8)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (0, 3)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'float32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (0, 5)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'float64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (0, 9)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'i,O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (0, 0)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'int16')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (0, 2)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'int32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (0, 4)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'int64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (3, 8)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (3, 3)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'float32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (3, 5)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'float64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (3, 9)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'i,O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (3, 0)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'int16')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (3, 2)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'int32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_getitem_ordered', (3, 4)),  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'int64')
    )
    run_repeat = (
        80,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'O')
        197,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'float32')
        197,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'float64')
        40,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'i,O')
        197,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'int16')
        197,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'int32')
        197,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 1), 'int64')
        53,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'O')
        65,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'float32')
        64,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'float64')
        29,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'i,O')
        65,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'int16')
        65,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'int32')
        64,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 2), 'int64')
        48,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'O')
        64,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'float32')
        62,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'float64')
        27,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'i,O')
        65,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'int16')
        64,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'int32')
        62,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000, 3), 'int64')
        162,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'O')
        477,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'float32')
        500,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'float64')
        46,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'i,O')
        477,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'int16')
        500,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'int32')
        500,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((1000,), 'int64')
        52,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'O')
        66,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'float32')
        66,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'float64')
        29,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'i,O')
        65,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'int16')
        66,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'int32')
        66,  # bench_indexing.IndexingWith1DArr.time_getitem_ordered((2, 1000, 1), 'int64')
    )
    case_methods = (
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
        'time_getitem_ordered',
    )
    case_types = (
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
    )


class Indexing(_AggregateBenchmark):
    """Aggregate indexing with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 0, 1, 0)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 0, 1, 1)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 0, 0, 0)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 0, 0, 1)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 0, 2, 0)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 0, 2, 1)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 1, 1, 0)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 1, 1, 1)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 1, 0, 0)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 1, 0, 1)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 1, 2, 0)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (9, 1, 2, 1)),  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 0, 1, 0)),  # bench_indexing.Indexing.time_op('float32', 'indexes_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 0, 1, 1)),  # bench_indexing.Indexing.time_op('float32', 'indexes_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 0, 0, 0)),  # bench_indexing.Indexing.time_op('float32', 'indexes_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 0, 0, 1)),  # bench_indexing.Indexing.time_op('float32', 'indexes_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 0, 2, 0)),  # bench_indexing.Indexing.time_op('float32', 'indexes_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 0, 2, 1)),  # bench_indexing.Indexing.time_op('float32', 'indexes_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 1, 1, 0)),  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 1, 1, 1)),  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 1, 0, 0)),  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 1, 0, 1)),  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 1, 2, 0)),  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (3, 1, 2, 1)),  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 0, 1, 0)),  # bench_indexing.Indexing.time_op('float64', 'indexes_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 0, 1, 1)),  # bench_indexing.Indexing.time_op('float64', 'indexes_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 0, 0, 0)),  # bench_indexing.Indexing.time_op('float64', 'indexes_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 0, 0, 1)),  # bench_indexing.Indexing.time_op('float64', 'indexes_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 0, 2, 0)),  # bench_indexing.Indexing.time_op('float64', 'indexes_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 0, 2, 1)),  # bench_indexing.Indexing.time_op('float64', 'indexes_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 1, 1, 0)),  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 1, 1, 1)),  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 1, 0, 0)),  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 1, 0, 1)),  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 1, 2, 0)),  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (5, 1, 2, 1)),  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 0, 1, 0)),  # bench_indexing.Indexing.time_op('int16', 'indexes_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 0, 1, 1)),  # bench_indexing.Indexing.time_op('int16', 'indexes_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 0, 0, 0)),  # bench_indexing.Indexing.time_op('int16', 'indexes_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 0, 0, 1)),  # bench_indexing.Indexing.time_op('int16', 'indexes_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 0, 2, 0)),  # bench_indexing.Indexing.time_op('int16', 'indexes_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 0, 2, 1)),  # bench_indexing.Indexing.time_op('int16', 'indexes_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 1, 1, 0)),  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 1, 1, 1)),  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 1, 0, 0)),  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 1, 0, 1)),  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 1, 2, 0)),  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (0, 1, 2, 1)),  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 0, 1, 0)),  # bench_indexing.Indexing.time_op('int32', 'indexes_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 0, 1, 1)),  # bench_indexing.Indexing.time_op('int32', 'indexes_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 0, 0, 0)),  # bench_indexing.Indexing.time_op('int32', 'indexes_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 0, 0, 1)),  # bench_indexing.Indexing.time_op('int32', 'indexes_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 0, 2, 0)),  # bench_indexing.Indexing.time_op('int32', 'indexes_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 0, 2, 1)),  # bench_indexing.Indexing.time_op('int32', 'indexes_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 1, 1, 0)),  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 1, 1, 1)),  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 1, 0, 0)),  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 1, 0, 1)),  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 1, 2, 0)),  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (2, 1, 2, 1)),  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 0, 1, 0)),  # bench_indexing.Indexing.time_op('int64', 'indexes_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 0, 1, 1)),  # bench_indexing.Indexing.time_op('int64', 'indexes_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 0, 0, 0)),  # bench_indexing.Indexing.time_op('int64', 'indexes_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 0, 0, 1)),  # bench_indexing.Indexing.time_op('int64', 'indexes_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 0, 2, 0)),  # bench_indexing.Indexing.time_op('int64', 'indexes_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 0, 2, 1)),  # bench_indexing.Indexing.time_op('int64', 'indexes_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 1, 1, 0)),  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 1, 1, 1)),  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 1, 0, 0)),  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 1, 0, 1)),  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 1, 2, 0)),  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (4, 1, 2, 1)),  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 0, 1, 0)),  # bench_indexing.Indexing.time_op('object', 'indexes_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 0, 1, 1)),  # bench_indexing.Indexing.time_op('object', 'indexes_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 0, 0, 0)),  # bench_indexing.Indexing.time_op('object', 'indexes_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 0, 0, 1)),  # bench_indexing.Indexing.time_op('object', 'indexes_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 0, 2, 0)),  # bench_indexing.Indexing.time_op('object', 'indexes_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 0, 2, 1)),  # bench_indexing.Indexing.time_op('object', 'indexes_', 'np.ix_(I, I)', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 1, 1, 0)),  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', ':,I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 1, 1, 1)),  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', ':,I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 1, 0, 0)),  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', 'I', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 1, 0, 1)),  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', 'I', '=1')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 1, 2, 0)),  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', 'np.ix_(I, I)', '')
        _select_case_params(_Official_bench_indexing_Indexing, 'time_op', (8, 1, 2, 1)),  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', 'np.ix_(I, I)', '=1')
    )
    run_repeat = (
        15,  # bench_indexing.Indexing.time_op('O,i', 'indexes_', ':,I', '')
        23,  # bench_indexing.Indexing.time_op('O,i', 'indexes_', ':,I', '=1')
        16,  # bench_indexing.Indexing.time_op('O,i', 'indexes_', 'I', '')
        24,  # bench_indexing.Indexing.time_op('O,i', 'indexes_', 'I', '=1')
        4,  # bench_indexing.Indexing.time_op('O,i', 'indexes_', 'np.ix_(I, I)', '')
        5,  # bench_indexing.Indexing.time_op('O,i', 'indexes_', 'np.ix_(I, I)', '=1')
        14,  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', ':,I', '')
        23,  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', ':,I', '=1')
        16,  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', 'I', '')
        24,  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', 'I', '=1')
        4,  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', 'np.ix_(I, I)', '')
        5,  # bench_indexing.Indexing.time_op('O,i', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        85,  # bench_indexing.Indexing.time_op('float32', 'indexes_', ':,I', '')
        81,  # bench_indexing.Indexing.time_op('float32', 'indexes_', ':,I', '=1')
        152,  # bench_indexing.Indexing.time_op('float32', 'indexes_', 'I', '')
        154,  # bench_indexing.Indexing.time_op('float32', 'indexes_', 'I', '=1')
        24,  # bench_indexing.Indexing.time_op('float32', 'indexes_', 'np.ix_(I, I)', '')
        25,  # bench_indexing.Indexing.time_op('float32', 'indexes_', 'np.ix_(I, I)', '=1')
        86,  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', ':,I', '')
        82,  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', ':,I', '=1')
        145,  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', 'I', '')
        152,  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', 'I', '=1')
        24,  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', 'np.ix_(I, I)', '')
        25,  # bench_indexing.Indexing.time_op('float32', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        85,  # bench_indexing.Indexing.time_op('float64', 'indexes_', ':,I', '')
        80,  # bench_indexing.Indexing.time_op('float64', 'indexes_', ':,I', '=1')
        130,  # bench_indexing.Indexing.time_op('float64', 'indexes_', 'I', '')
        127,  # bench_indexing.Indexing.time_op('float64', 'indexes_', 'I', '=1')
        23,  # bench_indexing.Indexing.time_op('float64', 'indexes_', 'np.ix_(I, I)', '')
        25,  # bench_indexing.Indexing.time_op('float64', 'indexes_', 'np.ix_(I, I)', '=1')
        85,  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', ':,I', '')
        79,  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', ':,I', '=1')
        127,  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', 'I', '')
        129,  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', 'I', '=1')
        23,  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', 'np.ix_(I, I)', '')
        25,  # bench_indexing.Indexing.time_op('float64', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        95,  # bench_indexing.Indexing.time_op('int16', 'indexes_', ':,I', '')
        89,  # bench_indexing.Indexing.time_op('int16', 'indexes_', ':,I', '=1')
        176,  # bench_indexing.Indexing.time_op('int16', 'indexes_', 'I', '')
        167,  # bench_indexing.Indexing.time_op('int16', 'indexes_', 'I', '=1')
        25,  # bench_indexing.Indexing.time_op('int16', 'indexes_', 'np.ix_(I, I)', '')
        24,  # bench_indexing.Indexing.time_op('int16', 'indexes_', 'np.ix_(I, I)', '=1')
        97,  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', ':,I', '')
        91,  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', ':,I', '=1')
        167,  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', 'I', '')
        164,  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', 'I', '=1')
        25,  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', 'np.ix_(I, I)', '')
        24,  # bench_indexing.Indexing.time_op('int16', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        85,  # bench_indexing.Indexing.time_op('int32', 'indexes_', ':,I', '')
        82,  # bench_indexing.Indexing.time_op('int32', 'indexes_', ':,I', '=1')
        150,  # bench_indexing.Indexing.time_op('int32', 'indexes_', 'I', '')
        154,  # bench_indexing.Indexing.time_op('int32', 'indexes_', 'I', '=1')
        23,  # bench_indexing.Indexing.time_op('int32', 'indexes_', 'np.ix_(I, I)', '')
        25,  # bench_indexing.Indexing.time_op('int32', 'indexes_', 'np.ix_(I, I)', '=1')
        86,  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', ':,I', '')
        82,  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', ':,I', '=1')
        145,  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', 'I', '')
        157,  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', 'I', '=1')
        23,  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', 'np.ix_(I, I)', '')
        25,  # bench_indexing.Indexing.time_op('int32', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        84,  # bench_indexing.Indexing.time_op('int64', 'indexes_', ':,I', '')
        80,  # bench_indexing.Indexing.time_op('int64', 'indexes_', ':,I', '=1')
        130,  # bench_indexing.Indexing.time_op('int64', 'indexes_', 'I', '')
        127,  # bench_indexing.Indexing.time_op('int64', 'indexes_', 'I', '=1')
        23,  # bench_indexing.Indexing.time_op('int64', 'indexes_', 'np.ix_(I, I)', '')
        25,  # bench_indexing.Indexing.time_op('int64', 'indexes_', 'np.ix_(I, I)', '=1')
        85,  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', ':,I', '')
        79,  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', ':,I', '=1')
        132,  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', 'I', '')
        127,  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', 'I', '=1')
        22,  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', 'np.ix_(I, I)', '')
        25,  # bench_indexing.Indexing.time_op('int64', 'indexes_rand_', 'np.ix_(I, I)', '=1')
        26,  # bench_indexing.Indexing.time_op('object', 'indexes_', ':,I', '')
        52,  # bench_indexing.Indexing.time_op('object', 'indexes_', ':,I', '=1')
        30,  # bench_indexing.Indexing.time_op('object', 'indexes_', 'I', '')
        56,  # bench_indexing.Indexing.time_op('object', 'indexes_', 'I', '=1')
        8,  # bench_indexing.Indexing.time_op('object', 'indexes_', 'np.ix_(I, I)', '')
        13,  # bench_indexing.Indexing.time_op('object', 'indexes_', 'np.ix_(I, I)', '=1')
        24,  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', ':,I', '')
        51,  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', ':,I', '=1')
        29,  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', 'I', '')
        56,  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', 'I', '=1')
        7,  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', 'np.ix_(I, I)', '')
        13,  # bench_indexing.Indexing.time_op('object', 'indexes_rand_', 'np.ix_(I, I)', '=1')
    )
    case_methods = (
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
        'time_op',
    )
    case_types = (
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
        _Official_bench_indexing_Indexing,
    )


class IndexingSeparate(_AggregateBenchmark):
    """Aggregate IndexingSeparate with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_indexing_IndexingSeparate, 'time_mmap_fancy_indexing', ()),  # bench_indexing.IndexingSeparate.time_mmap_fancy_indexing
        _select_case_params(_Official_bench_indexing_IndexingSeparate, 'time_mmap_slicing', ()),  # bench_indexing.IndexingSeparate.time_mmap_slicing
    )
    run_repeat = (
        1,  # bench_indexing.IndexingSeparate.time_mmap_fancy_indexing
        2,  # bench_indexing.IndexingSeparate.time_mmap_slicing
    )
    case_methods = (
        'time_mmap_fancy_indexing',
        'time_mmap_slicing',
    )
    case_types = (
        _Official_bench_indexing_IndexingSeparate,
        _Official_bench_indexing_IndexingSeparate,
    )


class IndexingStructured(_AggregateBenchmark):
    """Aggregate IndexingStructured with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_indexing_IndexingStructured0D, 'time_array_all', ()),  # bench_indexing.IndexingStructured0D.time_array_all
        _select_case_params(_Official_bench_indexing_IndexingStructured0D, 'time_array_slice', ()),  # bench_indexing.IndexingStructured0D.time_array_slice
        _select_case_params(_Official_bench_indexing_IndexingStructured0D, 'time_scalar_all', ()),  # bench_indexing.IndexingStructured0D.time_scalar_all
        _select_case_params(_Official_bench_indexing_IndexingStructured0D, 'time_scalar_slice', ()),  # bench_indexing.IndexingStructured0D.time_scalar_slice
    )
    run_repeat = (
        1000,  # bench_indexing.IndexingStructured0D.time_array_all
        770,  # bench_indexing.IndexingStructured0D.time_array_slice
        500,  # bench_indexing.IndexingStructured0D.time_scalar_all
        556,  # bench_indexing.IndexingStructured0D.time_scalar_slice
    )
    case_methods = (
        'time_array_all',
        'time_array_slice',
        'time_scalar_all',
        'time_scalar_slice',
    )
    case_types = (
        _Official_bench_indexing_IndexingStructured0D,
        _Official_bench_indexing_IndexingStructured0D,
        _Official_bench_indexing_IndexingStructured0D,
        _Official_bench_indexing_IndexingStructured0D,
    )


class Setitem(_AggregateBenchmark):
    """Aggregate setitem with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (1, 8)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (1, 3)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'float32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (1, 5)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'float64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (1, 9)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'i,O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (1, 0)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'int16')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (1, 2)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'int32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (1, 4)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'int64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (2, 8)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (2, 3)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'float32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (2, 5)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'float64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (2, 9)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'i,O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (2, 0)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'int16')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (2, 2)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'int32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (2, 4)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'int64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (4, 8)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (4, 3)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'float32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (4, 5)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'float64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (4, 9)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'i,O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (4, 0)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'int16')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (4, 2)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'int32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (4, 4)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'int64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (0, 8)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (0, 3)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'float32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (0, 5)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'float64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (0, 9)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'i,O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (0, 0)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'int16')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (0, 2)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'int32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (0, 4)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'int64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (3, 8)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (3, 3)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'float32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (3, 5)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'float64')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (3, 9)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'i,O')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (3, 0)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'int16')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (3, 2)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'int32')
        _select_case_params(_Official_bench_indexing_IndexingWith1DArr, 'time_setitem_ordered', (3, 4)),  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'int64')
    )
    run_repeat = (
        129,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'O')
        182,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'float32')
        182,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'float64')
        42,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'i,O')
        182,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'int16')
        182,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'int32')
        182,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 1), 'int64')
        47,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'O')
        68,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'float32')
        68,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'float64')
        28,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'i,O')
        68,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'int16')
        68,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'int32')
        68,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 2), 'int64')
        45,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'O')
        69,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'float32')
        66,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'float64')
        26,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'i,O')
        68,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'int16')
        69,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'int32')
        67,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000, 3), 'int64')
        167,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'O')
        400,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'float32')
        400,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'float64')
        45,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'i,O')
        400,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'int16')
        400,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'int32')
        400,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((1000,), 'int64')
        46,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'O')
        64,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'float32')
        64,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'float64')
        28,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'i,O')
        64,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'int16')
        64,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'int32')
        64,  # bench_indexing.IndexingWith1DArr.time_setitem_ordered((2, 1000, 1), 'int64')
    )
    case_methods = (
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
        'time_setitem_ordered',
    )
    case_types = (
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
        _Official_bench_indexing_IndexingWith1DArr,
    )
