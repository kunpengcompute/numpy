"""920b platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_lib import Nan as _Official_bench_lib_Nan


class Nanargmax(_AggregateBenchmark):
    """Aggregate nanargmax with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmax', (0, 0)),  # bench_lib.Nan.time_nanargmax(200, 0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmax', (0, 1)),  # bench_lib.Nan.time_nanargmax(200, 0.1)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmax', (0, 2)),  # bench_lib.Nan.time_nanargmax(200, 2.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmax', (0, 3)),  # bench_lib.Nan.time_nanargmax(200, 50.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmax', (0, 4)),  # bench_lib.Nan.time_nanargmax(200, 90.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmax', (1, 0)),  # bench_lib.Nan.time_nanargmax(200000, 0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmax', (1, 1)),  # bench_lib.Nan.time_nanargmax(200000, 0.1)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmax', (1, 2)),  # bench_lib.Nan.time_nanargmax(200000, 2.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmax', (1, 3)),  # bench_lib.Nan.time_nanargmax(200000, 50.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmax', (1, 4)),  # bench_lib.Nan.time_nanargmax(200000, 90.0)
    )
    run_repeat = (
        59,  # bench_lib.Nan.time_nanargmax(200, 0)
        60,  # bench_lib.Nan.time_nanargmax(200, 0.1)
        59,  # bench_lib.Nan.time_nanargmax(200, 2.0)
        59,  # bench_lib.Nan.time_nanargmax(200, 50.0)
        59,  # bench_lib.Nan.time_nanargmax(200, 90.0)
        2,  # bench_lib.Nan.time_nanargmax(200000, 0)
        2,  # bench_lib.Nan.time_nanargmax(200000, 0.1)
        2,  # bench_lib.Nan.time_nanargmax(200000, 2.0)
        1,  # bench_lib.Nan.time_nanargmax(200000, 50.0)
        2,  # bench_lib.Nan.time_nanargmax(200000, 90.0)
    )
    case_methods = (
        'time_nanargmax',
        'time_nanargmax',
        'time_nanargmax',
        'time_nanargmax',
        'time_nanargmax',
        'time_nanargmax',
        'time_nanargmax',
        'time_nanargmax',
        'time_nanargmax',
        'time_nanargmax',
    )
    case_types = (
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
    )


class Nanargmin(_AggregateBenchmark):
    """Aggregate nanargmin with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmin', (0, 0)),  # bench_lib.Nan.time_nanargmin(200, 0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmin', (0, 1)),  # bench_lib.Nan.time_nanargmin(200, 0.1)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmin', (0, 2)),  # bench_lib.Nan.time_nanargmin(200, 2.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmin', (0, 3)),  # bench_lib.Nan.time_nanargmin(200, 50.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmin', (0, 4)),  # bench_lib.Nan.time_nanargmin(200, 90.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmin', (1, 0)),  # bench_lib.Nan.time_nanargmin(200000, 0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmin', (1, 1)),  # bench_lib.Nan.time_nanargmin(200000, 0.1)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmin', (1, 2)),  # bench_lib.Nan.time_nanargmin(200000, 2.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmin', (1, 3)),  # bench_lib.Nan.time_nanargmin(200000, 50.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanargmin', (1, 4)),  # bench_lib.Nan.time_nanargmin(200000, 90.0)
    )
    run_repeat = (
        60,  # bench_lib.Nan.time_nanargmin(200, 0)
        60,  # bench_lib.Nan.time_nanargmin(200, 0.1)
        60,  # bench_lib.Nan.time_nanargmin(200, 2.0)
        58,  # bench_lib.Nan.time_nanargmin(200, 50.0)
        60,  # bench_lib.Nan.time_nanargmin(200, 90.0)
        2,  # bench_lib.Nan.time_nanargmin(200000, 0)
        2,  # bench_lib.Nan.time_nanargmin(200000, 0.1)
        2,  # bench_lib.Nan.time_nanargmin(200000, 2.0)
        1,  # bench_lib.Nan.time_nanargmin(200000, 50.0)
        2,  # bench_lib.Nan.time_nanargmin(200000, 90.0)
    )
    case_methods = (
        'time_nanargmin',
        'time_nanargmin',
        'time_nanargmin',
        'time_nanargmin',
        'time_nanargmin',
        'time_nanargmin',
        'time_nanargmin',
        'time_nanargmin',
        'time_nanargmin',
        'time_nanargmin',
    )
    case_types = (
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
    )


class Nanpercentile(_AggregateBenchmark):
    """Aggregate nanpercentile with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_lib_Nan, 'time_nanpercentile', (0, 0)),  # bench_lib.Nan.time_nanpercentile(200, 0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanpercentile', (0, 1)),  # bench_lib.Nan.time_nanpercentile(200, 0.1)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanpercentile', (0, 2)),  # bench_lib.Nan.time_nanpercentile(200, 2.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanpercentile', (0, 3)),  # bench_lib.Nan.time_nanpercentile(200, 50.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanpercentile', (0, 4)),  # bench_lib.Nan.time_nanpercentile(200, 90.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanpercentile', (1, 0)),  # bench_lib.Nan.time_nanpercentile(200000, 0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanpercentile', (1, 1)),  # bench_lib.Nan.time_nanpercentile(200000, 0.1)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanpercentile', (1, 2)),  # bench_lib.Nan.time_nanpercentile(200000, 2.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanpercentile', (1, 3)),  # bench_lib.Nan.time_nanpercentile(200000, 50.0)
        _select_case_params(_Official_bench_lib_Nan, 'time_nanpercentile', (1, 4)),  # bench_lib.Nan.time_nanpercentile(200000, 90.0)
    )
    run_repeat = (
        16,  # bench_lib.Nan.time_nanpercentile(200, 0)
        16,  # bench_lib.Nan.time_nanpercentile(200, 0.1)
        14,  # bench_lib.Nan.time_nanpercentile(200, 2.0)
        14,  # bench_lib.Nan.time_nanpercentile(200, 50.0)
        15,  # bench_lib.Nan.time_nanpercentile(200, 90.0)
        1,  # bench_lib.Nan.time_nanpercentile(200000, 0)
        1,  # bench_lib.Nan.time_nanpercentile(200000, 0.1)
        1,  # bench_lib.Nan.time_nanpercentile(200000, 2.0)
        1,  # bench_lib.Nan.time_nanpercentile(200000, 50.0)
        1,  # bench_lib.Nan.time_nanpercentile(200000, 90.0)
    )
    case_methods = (
        'time_nanpercentile',
        'time_nanpercentile',
        'time_nanpercentile',
        'time_nanpercentile',
        'time_nanpercentile',
        'time_nanpercentile',
        'time_nanpercentile',
        'time_nanpercentile',
        'time_nanpercentile',
        'time_nanpercentile',
    )
    case_types = (
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
        _Official_bench_lib_Nan,
    )
