"""950 platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_ufunc_strides import (
    BinaryFP as _Official_bench_ufunc_strides_BinaryFP,
    BinaryInt as _Official_bench_ufunc_strides_BinaryInt,
    UnaryFP as _Official_bench_ufunc_strides_UnaryFP,
    UnaryFPSpecial as _Official_bench_ufunc_strides_UnaryFPSpecial,
)


class Maximum(_AggregateBenchmark):
    """Aggregate maximum with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 0, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary', (0, 1, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 0, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 0, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 0, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 0, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 0, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 0, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 0, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 1, 0, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 1, 0, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 1, 1, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 1, 1, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 1, 2, 1)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 4, 'd')
        _select_case_params(_Official_bench_ufunc_strides_BinaryFP, 'time_binary_scalar_in0', (0, 1, 1, 2, 0)),  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 4, 'f')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 1, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 1, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 1, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 1, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 1, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 1, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 1, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 1, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 1, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 0, 1, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 0, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 0, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 0, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 0, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 0, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 0, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 0, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 0, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 0, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 0, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 1, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 1, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 1, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 1, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 1, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 1, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 1, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 1, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 1, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 0, 1, 1, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 0, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 0, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 0, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 0, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 0, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 0, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 0, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 0, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 0, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 0, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 1, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 1, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 1, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 1, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 1, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 1, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 1, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 1, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 1, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 0, 1, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 0, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 0, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 0, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 0, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 0, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 0, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 0, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 0, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 0, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 0, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 1, 1)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'B')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 1, 3)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'H')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 1, 5)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'I')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 1, 7)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'L')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 1, 9)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'Q')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 1, 0)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'b')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 1, 2)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'h')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 1, 4)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'i')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 1, 6)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'l')
        _select_case_params(_Official_bench_ufunc_strides_BinaryInt, 'time_binary', (0, 1, 1, 1, 8)),  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'q')
    )
    run_repeat = (
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 1, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 4, 'd')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 1, 4, 4, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 1, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 4, 'd')
        1,  # bench_ufunc_strides.BinaryFP.time_binary(<ufunc 'maximum'>, 2, 4, 4, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 2, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 4, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 1, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 1, 4, 4, 'f')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 1, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 1, 'f')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 2, 'd')
        4,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 4, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 1, 4, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 1, 'd')
        3,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 1, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 2, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 2, 'f')
        1,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 4, 'd')
        2,  # bench_ufunc_strides.BinaryFP.time_binary_scalar_in0(<ufunc 'maximum'>, 2, 4, 4, 'f')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'B')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'H')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'I')
        16,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'L')
        16,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'b')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'h')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'i')
        16,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'l')
        16,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 1, 2, 'q')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'B')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'H')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'I')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'L')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'b')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'h')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'i')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'l')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 1, 'q')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'B')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'H')
        23,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'I')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'L')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'b')
        20,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'h')
        23,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'i')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'l')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 1, 2, 2, 'q')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'B')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'H')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'I')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'L')
        16,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'b')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'h')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'i')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'l')
        17,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 1, 'q')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'B')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'H')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'I')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'L')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'b')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'h')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'i')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'l')
        12,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 1, 2, 'q')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'B')
        26,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'H')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'I')
        13,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'L')
        13,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'b')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'h')
        24,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'i')
        13,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'l')
        14,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 1, 'q')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'B')
        25,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'H')
        20,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'I')
        10,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'L')
        10,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'Q')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'b')
        21,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'h')
        20,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'i')
        10,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'l')
        10,  # bench_ufunc_strides.BinaryInt.time_binary(<ufunc 'maximum'>, 2, 2, 2, 'q')
    )
    case_methods = (
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
    )


class OnesLike(_AggregateBenchmark):
    """Aggregate ones_like with frozen 950 repeats."""

    case_params = (
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 0, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 0, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 0, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 0, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 0, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 0, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 1, 0, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 1, 0, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 1, 0, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 1, 1, 2)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 1, 1, 0)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFP, 'time_unary', (39, 1, 1, 1)),  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 0, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 0, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 0, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 0, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 0, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 0, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 2, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 1, 0, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 1, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 1, 0, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 1, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 1, 0, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 1, 'f')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 1, 1, 2)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 2, 'd')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 1, 1, 0)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 2, 'e')
        _select_case_params(_Official_bench_ufunc_strides_UnaryFPSpecial, 'time_unary', (39, 1, 1, 1)),  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 2, 'f')
    )
    run_repeat = (
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 1, 'd')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 1, 'f')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 2, 'd')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 2, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 1, 2, 'f')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 1, 'd')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 1, 'f')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 2, 'd')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 2, 'e')
        3,  # bench_ufunc_strides.UnaryFP.time_unary(<ufunc '_ones_like'>, 4, 2, 'f')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 1, 'd')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 1, 'f')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 2, 'd')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 2, 'e')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 1, 2, 'f')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 1, 'd')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 1, 'e')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 1, 'f')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 2, 'd')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 2, 'e')
        3,  # bench_ufunc_strides.UnaryFPSpecial.time_unary(<ufunc '_ones_like'>, 4, 2, 'f')
    )
    case_methods = (
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
        'time_unary',
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
