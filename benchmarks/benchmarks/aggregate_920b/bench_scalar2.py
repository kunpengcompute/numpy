"""920b platform aggregate benchmarks.

The timed cases are imported from the canonical ASV benchmark modules;
this module contains only platform-specific selection and repeat metadata.
"""

from .._aggregate_common import (
    _AggregateBenchmark,
    select_case_params as _select_case_params,
)
from ..bench_scalar import ScalarMath as _Official_bench_scalar_ScalarMath
from ..bench_ufunc import (
    BinaryBench as _Official_bench_ufunc_BinaryBench,
    BinaryBenchInteger as _Official_bench_ufunc_BinaryBenchInteger,
    MethodsV1 as _Official_bench_ufunc_MethodsV1,
)


class Power(_AggregateBenchmark):
    """Aggregate power with frozen 920b repeats."""

    case_params = (
        _select_case_params(_Official_bench_scalar_ScalarMath, 'time_power_of_two', (5,)),  # bench_scalar.ScalarMath.time_power_of_two('float64')
        _select_case_params(_Official_bench_scalar_ScalarMath, 'time_power_of_two', (4,)),  # bench_scalar.ScalarMath.time_power_of_two('int64')
        _select_case_params(_Official_bench_ufunc_BinaryBench, 'time_pow', (1,)),  # bench_ufunc.BinaryBench.time_pow(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_ufunc_BinaryBench, 'time_pow_2', (1,)),  # bench_ufunc.BinaryBench.time_pow_2(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_ufunc_BinaryBench, 'time_pow_2_op', (1,)),  # bench_ufunc.BinaryBench.time_pow_2_op(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_ufunc_BinaryBench, 'time_pow_half', (1,)),  # bench_ufunc.BinaryBench.time_pow_half(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_ufunc_BinaryBench, 'time_pow_half_op', (1,)),  # bench_ufunc.BinaryBench.time_pow_half_op(<class 'numpy.float64'>)
        _select_case_params(_Official_bench_ufunc_BinaryBenchInteger, 'time_pow', (1,)),  # bench_ufunc.BinaryBenchInteger.time_pow(<class 'numpy.int64'>)
        _select_case_params(_Official_bench_ufunc_BinaryBenchInteger, 'time_pow_five', (1,)),  # bench_ufunc.BinaryBenchInteger.time_pow_five(<class 'numpy.int64'>)
        _select_case_params(_Official_bench_ufunc_BinaryBenchInteger, 'time_pow_two', (1,)),  # bench_ufunc.BinaryBenchInteger.time_pow_two(<class 'numpy.int64'>)
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (9, 5)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__pow__', 'float64')
        _select_case_params(_Official_bench_ufunc_MethodsV1, 'time_ndarray_meth', (9, 4)),  # bench_ufunc.MethodsV1.time_ndarray_meth('__pow__', 'int64')
    )
    run_repeat = (
        834,  # bench_scalar.ScalarMath.time_power_of_two('float64')
        1000,  # bench_scalar.ScalarMath.time_power_of_two('int64')
        1,  # bench_ufunc.BinaryBench.time_pow(<class 'numpy.float64'>)
        2,  # bench_ufunc.BinaryBench.time_pow_2(<class 'numpy.float64'>)
        5,  # bench_ufunc.BinaryBench.time_pow_2_op(<class 'numpy.float64'>)
        1,  # bench_ufunc.BinaryBench.time_pow_half(<class 'numpy.float64'>)
        2,  # bench_ufunc.BinaryBench.time_pow_half_op(<class 'numpy.float64'>)
        1,  # bench_ufunc.BinaryBenchInteger.time_pow(<class 'numpy.int64'>)
        1,  # bench_ufunc.BinaryBenchInteger.time_pow_five(<class 'numpy.int64'>)
        1,  # bench_ufunc.BinaryBenchInteger.time_pow_two(<class 'numpy.int64'>)
        385,  # bench_ufunc.MethodsV1.time_ndarray_meth('__pow__', 'float64')
        589,  # bench_ufunc.MethodsV1.time_ndarray_meth('__pow__', 'int64')
    )
    case_methods = (
        'time_power_of_two',
        'time_power_of_two',
        'time_pow',
        'time_pow_2',
        'time_pow_2_op',
        'time_pow_half',
        'time_pow_half_op',
        'time_pow',
        'time_pow_five',
        'time_pow_two',
        'time_ndarray_meth',
        'time_ndarray_meth',
    )
    case_types = (
        _Official_bench_scalar_ScalarMath,
        _Official_bench_scalar_ScalarMath,
        _Official_bench_ufunc_BinaryBench,
        _Official_bench_ufunc_BinaryBench,
        _Official_bench_ufunc_BinaryBench,
        _Official_bench_ufunc_BinaryBench,
        _Official_bench_ufunc_BinaryBench,
        _Official_bench_ufunc_BinaryBenchInteger,
        _Official_bench_ufunc_BinaryBenchInteger,
        _Official_bench_ufunc_BinaryBenchInteger,
        _Official_bench_ufunc_MethodsV1,
        _Official_bench_ufunc_MethodsV1,
    )
