"""920b platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_random import (
    RNG as _Official_bench_random_RNG,
    Randint as _Official_bench_random_Randint,
    Randint_dtype as _Official_bench_random_Randint_dtype,
)


class Randint(_AggregateBenchmark):
    """Aggregate randint with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_random_Randint, 'time_randint_fast', ()),  # bench_random.Randint.time_randint_fast
        _select_case_params(_Official_bench_random_Randint, 'time_randint_slow', ()),  # bench_random.Randint.time_randint_slow
        _select_case_params(_Official_bench_random_Randint_dtype, 'time_randint_fast', (4,)),  # bench_random.Randint_dtype.time_randint_fast('uint64')
        _select_case_params(_Official_bench_random_Randint_dtype, 'time_randint_slow', (4,)),  # bench_random.Randint_dtype.time_randint_slow('uint64')
    )
    run_repeat = (
        4,  # bench_random.Randint.time_randint_fast
        1,  # bench_random.Randint.time_randint_slow
        3,  # bench_random.Randint_dtype.time_randint_fast('uint64')
        1,  # bench_random.Randint_dtype.time_randint_slow('uint64')
    )
    case_methods = (
        'time_randint_fast',
        'time_randint_slow',
        'time_randint_fast',
        'time_randint_slow',
    )
    case_types = (
        _Official_bench_random_Randint,
        _Official_bench_random_Randint,
        _Official_bench_random_Randint_dtype,
        _Official_bench_random_Randint_dtype,
    )


class RngBit(_AggregateBenchmark):
    """Aggregate rng-bit with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_random_RNG, 'time_32bit', (1,)),  # bench_random.RNG.time_32bit('MT19937')
        _select_case_params(_Official_bench_random_RNG, 'time_32bit', (0,)),  # bench_random.RNG.time_32bit('PCG64')
        _select_case_params(_Official_bench_random_RNG, 'time_32bit', (2,)),  # bench_random.RNG.time_32bit('Philox')
        _select_case_params(_Official_bench_random_RNG, 'time_32bit', (3,)),  # bench_random.RNG.time_32bit('SFC64')
        _select_case_params(_Official_bench_random_RNG, 'time_32bit', (4,)),  # bench_random.RNG.time_32bit('numpy')
        _select_case_params(_Official_bench_random_RNG, 'time_64bit', (1,)),  # bench_random.RNG.time_64bit('MT19937')
        _select_case_params(_Official_bench_random_RNG, 'time_64bit', (0,)),  # bench_random.RNG.time_64bit('PCG64')
        _select_case_params(_Official_bench_random_RNG, 'time_64bit', (2,)),  # bench_random.RNG.time_64bit('Philox')
        _select_case_params(_Official_bench_random_RNG, 'time_64bit', (3,)),  # bench_random.RNG.time_64bit('SFC64')
        _select_case_params(_Official_bench_random_RNG, 'time_64bit', (4,)),  # bench_random.RNG.time_64bit('numpy')
    )
    run_repeat = (
        4,  # bench_random.RNG.time_32bit('MT19937')
        4,  # bench_random.RNG.time_32bit('PCG64')
        3,  # bench_random.RNG.time_32bit('Philox')
        6,  # bench_random.RNG.time_32bit('SFC64')
        4,  # bench_random.RNG.time_32bit('numpy')
        3,  # bench_random.RNG.time_64bit('MT19937')
        3,  # bench_random.RNG.time_64bit('PCG64')
        2,  # bench_random.RNG.time_64bit('Philox')
        3,  # bench_random.RNG.time_64bit('SFC64')
        3,  # bench_random.RNG.time_64bit('numpy')
    )
    case_methods = (
        'time_32bit',
        'time_32bit',
        'time_32bit',
        'time_32bit',
        'time_32bit',
        'time_64bit',
        'time_64bit',
        'time_64bit',
        'time_64bit',
        'time_64bit',
    )
    case_types = (
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
    )


class RngNormal(_AggregateBenchmark):
    """Aggregate rng-normal with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_random_RNG, 'time_normal_zig', (1,)),  # bench_random.RNG.time_normal_zig('MT19937')
        _select_case_params(_Official_bench_random_RNG, 'time_normal_zig', (0,)),  # bench_random.RNG.time_normal_zig('PCG64')
        _select_case_params(_Official_bench_random_RNG, 'time_normal_zig', (2,)),  # bench_random.RNG.time_normal_zig('Philox')
        _select_case_params(_Official_bench_random_RNG, 'time_normal_zig', (3,)),  # bench_random.RNG.time_normal_zig('SFC64')
        _select_case_params(_Official_bench_random_RNG, 'time_normal_zig', (4,)),  # bench_random.RNG.time_normal_zig('numpy')
    )
    run_repeat = (
        2,  # bench_random.RNG.time_normal_zig('MT19937')
        2,  # bench_random.RNG.time_normal_zig('PCG64')
        2,  # bench_random.RNG.time_normal_zig('Philox')
        2,  # bench_random.RNG.time_normal_zig('SFC64')
        1,  # bench_random.RNG.time_normal_zig('numpy')
    )
    case_methods = (
        'time_normal_zig',
        'time_normal_zig',
        'time_normal_zig',
        'time_normal_zig',
        'time_normal_zig',
    )
    case_types = (
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
    )


class RngRaw(_AggregateBenchmark):
    """Aggregate rng-raw with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_random_RNG, 'time_raw', (1,)),  # bench_random.RNG.time_raw('MT19937')
        _select_case_params(_Official_bench_random_RNG, 'time_raw', (0,)),  # bench_random.RNG.time_raw('PCG64')
        _select_case_params(_Official_bench_random_RNG, 'time_raw', (2,)),  # bench_random.RNG.time_raw('Philox')
        _select_case_params(_Official_bench_random_RNG, 'time_raw', (3,)),  # bench_random.RNG.time_raw('SFC64')
        _select_case_params(_Official_bench_random_RNG, 'time_raw', (4,)),  # bench_random.RNG.time_raw('numpy')
    )
    run_repeat = (
        2,  # bench_random.RNG.time_raw('MT19937')
        2,  # bench_random.RNG.time_raw('PCG64')
        2,  # bench_random.RNG.time_raw('Philox')
        2,  # bench_random.RNG.time_raw('SFC64')
        3,  # bench_random.RNG.time_raw('numpy')
    )
    case_methods = (
        'time_raw',
        'time_raw',
        'time_raw',
        'time_raw',
        'time_raw',
    )
    case_types = (
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
        _Official_bench_random_RNG,
    )
