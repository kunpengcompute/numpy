"""
KML FFT backend -- Python wrapper for _kml_fft_umath.

This module mirrors the _pocketfft.py interface with KML-specific
normalization: KML does not apply 1/n internally for inverse transforms,
so normalization is handled entirely in Python.

The public symbols from this module are NOT exported directly; they
are called through the _BackendManager dispatch in _backend.py.
"""

from numpy._core import (
    asarray,
    empty_like,
    reciprocal,
    result_type,
    sqrt,
)
from numpy.lib.array_utils import normalize_axis_index

from . import _kml_fft_umath as kml


def _raw_fft(a, n, axis, is_real, is_forward, norm, out=None):
    """Core 1-D FFT dispatch for KML backend.

    Unlike pocketfft, KML does NOT apply 1/n in the inverse direction.
    All normalization is applied via the ``fct`` factor.
    """
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")

    # KML-specific normalization (no _swap_direction needed because
    # KML does not internally apply any scaling in either direction).
    real_dtype = result_type(a.real.dtype, 1.0)
    if norm is None or norm == "backward":
        fct = 1 if is_forward else reciprocal(n, dtype=real_dtype)
    elif norm == "ortho":
        fct = reciprocal(sqrt(n, dtype=real_dtype))
    elif norm == "forward":
        fct = reciprocal(n, dtype=real_dtype) if is_forward else 1
    else:
        raise ValueError(f'Invalid norm value {norm}; should be "backward",'
                         '"ortho" or "forward".')

    n_out = n
    if is_real:
        if is_forward:
            ufunc = kml.rfft_n_even if n % 2 == 0 else kml.rfft_n_odd
            n_out = n // 2 + 1
        else:
            ufunc = kml.irfft
    else:
        ufunc = kml.fft if is_forward else kml.ifft

    axis = normalize_axis_index(axis, a.ndim)

    if out is None:
        # Determine output dtype: irfft produces real; all others produce complex.
        out_dtype = (real_dtype if is_real and not is_forward
                     else result_type(a.dtype, 1j))
        new_shape = list(a.shape)
        new_shape[axis] = n_out
        out = empty_like(a, shape=new_shape, dtype=out_dtype)
    elif getattr(out, "ndim", 0) != a.ndim or out.shape[axis] != n_out:
        raise ValueError("output array has wrong shape.")

    return ufunc(a, fct, axes=[(axis,), (), (axis,)], out=out)


def _raw_fftnd(a, s=None, axes=None, function=None, norm=None, out=None):
    """Multi-dimensional FFT via 1-D transforms.

    Reuses ``_cook_nd_args`` from ``_pocketfft`` for argument parsing,
    then iterates over axes applying 1-D transforms.
    """
    from ._pocketfft import _cook_nd_args

    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes)
    itl = list(range(len(axes)))
    itl.reverse()
    for ii in itl:
        a = function(a, n=s[ii], axis=axes[ii], norm=norm, out=out)
    return a
