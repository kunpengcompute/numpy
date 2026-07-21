"""920b platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_custom import (
    ElementwiseBroadcast as _Official_bench_custom_ElementwiseBroadcast,
    RandomAndStats as _Official_bench_custom_RandomAndStats,
    SortSearch as _Official_bench_custom_SortSearch,
)


class Normalise(_AggregateBenchmark):
    """Aggregate normalise with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_custom_ElementwiseBroadcast, 'time_normalise_and_sigmoid', (0,)),  # bench_custom.ElementwiseBroadcast.time_normalise_and_sigmoid(4000)
    )
    run_repeat = (
        1,  # bench_custom.ElementwiseBroadcast.time_normalise_and_sigmoid(4000)
    )
    case_methods = (
        'time_normalise_and_sigmoid',
    )
    case_types = (
        _Official_bench_custom_ElementwiseBroadcast,
    )


class Randome(_AggregateBenchmark):
    """Aggregate randome with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_custom_RandomAndStats, 'time_rng', (0,)),  # bench_custom.RandomAndStats.time_rng(4000)
    )
    run_repeat = (
        1,  # bench_custom.RandomAndStats.time_rng(4000)
    )
    case_methods = (
        'time_rng',
    )
    case_types = (
        _Official_bench_custom_RandomAndStats,
    )


class SortSearch1(_AggregateBenchmark):
    """Aggregate SortSearch1 with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_custom_SortSearch, 'time_argsort', (0,)),  # bench_custom.SortSearch.time_argsort(4000)
    )
    run_repeat = (
        1,  # bench_custom.SortSearch.time_argsort(4000)
    )
    case_methods = (
        'time_argsort',
    )
    case_types = (
        _Official_bench_custom_SortSearch,
    )


class SortSearch2(_AggregateBenchmark):
    """Aggregate SortSearch2 with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_custom_SortSearch, 'time_searchsorted', (0,)),  # bench_custom.SortSearch.time_searchsorted(4000)
    )
    run_repeat = (
        1,  # bench_custom.SortSearch.time_searchsorted(4000)
    )
    case_methods = (
        'time_searchsorted',
    )
    case_types = (
        _Official_bench_custom_SortSearch,
    )


class SortSearch3(_AggregateBenchmark):
    """Aggregate SortSearch3 with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_custom_SortSearch, 'time_sort', (0,)),  # bench_custom.SortSearch.time_sort(4000)
    )
    run_repeat = (
        1,  # bench_custom.SortSearch.time_sort(4000)
    )
    case_methods = (
        'time_sort',
    )
    case_types = (
        _Official_bench_custom_SortSearch,
    )


class Statistics(_AggregateBenchmark):
    """Aggregate statistics with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_custom_RandomAndStats, 'time_statistics', (0,)),  # bench_custom.RandomAndStats.time_statistics(4000)
    )
    run_repeat = (
        1,  # bench_custom.RandomAndStats.time_statistics(4000)
    )
    case_methods = (
        'time_statistics',
    )
    case_types = (
        _Official_bench_custom_RandomAndStats,
    )
