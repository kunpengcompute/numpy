"""
FFT backend manager for NumPy.

Provides runtime backend switching between PocketFFT (default) and
KML FFT (optional, Kunpeng Math Library).

Public API:
    get_backend()      -- return name of the current backend
    set_backend(name)  -- context manager for temporary backend switch
    set_global_backend(name) -- set global default backend
    reset_backend()    -- reset to system default
"""

import os
import threading
import warnings


class FFTBackend:
    """Abstract base class for FFT backends.

    Subclasses must implement ``_raw_fft`` and ``_raw_fftnd`` with
    signatures matching those in ``_pocketfft.py``.
    """

    @property
    def name(self):
        raise NotImplementedError

    def _raw_fft(self, a, n, axis, is_real, is_forward, norm, out=None):
        """Core 1-D FFT dispatch."""
        raise NotImplementedError

    def _raw_fftnd(self, a, s=None, axes=None, function=None, norm=None,
                   out=None):
        """Core N-D FFT dispatch."""
        raise NotImplementedError

    def supports_type(self, dtype):
        """Return True if this backend supports the given dtype."""
        return True

    def supports_norm(self, norm):
        return norm in (None, "backward", "ortho", "forward")


class PocketFFTBackend(FFTBackend):
    """PocketFFT backend -- the built-in default."""

    @property
    def name(self):
        return "pocketfft"

    def _raw_fft(self, a, n, axis, is_real, is_forward, norm, out=None):
        from ._pocketfft import _raw_fft as _pf_raw_fft
        return _pf_raw_fft(a, n, axis, is_real, is_forward, norm, out)

    def _raw_fftnd(self, a, s=None, axes=None, function=None, norm=None,
                   out=None):
        from ._pocketfft import _raw_fftnd as _pf_raw_fftnd
        return _pf_raw_fftnd(a, s, axes, function, norm, out)


class KMLFFTBackend(FFTBackend):
    """KML FFT backend -- Kunpeng Math Library optimized FFT."""

    def __init__(self):
        from . import _kml_fft_umath  # noqa: F401  # verify C module available

    @property
    def name(self):
        return "kmlfft"

    def _raw_fft(self, a, n, axis, is_real, is_forward, norm, out=None):
        from ._kml_fft import _raw_fft as _kml_raw_fft
        return _kml_raw_fft(a, n, axis, is_real, is_forward, norm, out)

    def _raw_fftnd(self, a, s=None, axes=None, function=None, norm=None,
                   out=None):
        from ._kml_fft import _raw_fftnd as _kml_raw_fftnd
        return _kml_raw_fftnd(a, s, axes, function, norm, out)

    def supports_type(self, dtype):
        import numpy as np
        return dtype in (np.float16, np.float32, np.complex64,
                         np.float64, np.complex128)


class _BackendContext:
    """Context manager for scoped backend switching.

    Usage::

        with set_backend("kmlfft"):
            result = np.fft.fft(data)
    """

    def __init__(self, manager, backend_name):
        self._manager = manager
        self._backend_name = backend_name
        self._previous = None

    def __enter__(self):
        tl = self._manager._thread_local
        self._previous = getattr(tl, "backend", None)
        backend = self._manager._backends[self._backend_name]
        tl.backend = backend
        return self

    def __exit__(self, *args):
        tl = self._manager._thread_local
        if self._previous is not None:
            tl.backend = self._previous
        else:
            try:
                del tl.backend
            except AttributeError:
                pass
        return False  # don't suppress exceptions


class _BackendManager:
    """Singleton manager for FFT backend selection.

    Backend priority (highest to lowest):
        1. Thread-local backend (set via ``set_backend()`` context)
        2. Global backend (set via ``set_global_backend()``)
        3. Environment variable ``NUMPY_FFT_BACKEND``
        4. System default (``pocketfft``)
    """

    _DEFAULT = "pocketfft"

    def __init__(self):
        self._lock = threading.Lock()
        self._backends = {}
        self._global_backend = None
        self._env_backend = None
        self._thread_local = threading.local()

        # Register pocketfft (always available)
        self._backends["pocketfft"] = PocketFFTBackend()

        # Try to register KML FFT
        try:
            self._backends["kmlfft"] = KMLFFTBackend()
        except ImportError:
            pass

        # Check environment variable
        env_name = os.environ.get("NUMPY_FFT_BACKEND", "").lower()
        if env_name:
            if env_name in self._backends:
                self._env_backend = self._backends[env_name]
            else:
                available = ", ".join(sorted(self._backends))
                warnings.warn(
                    f"NUMPY_FFT_BACKEND={env_name!r} is not a registered "
                    f"backend. Available: {available}. "
                    f"Falling back to default ({self._DEFAULT}).",
                    RuntimeWarning, stacklevel=2
                )

    def get_current_backend(self):
        """Return the current effective backend instance."""
        tl = self._thread_local
        if hasattr(tl, "backend") and tl.backend is not None:
            return tl.backend
        if self._global_backend is not None:
            return self._global_backend
        if self._env_backend is not None:
            return self._env_backend
        return self._backends[self._DEFAULT]

    def get_backend_name(self):
        return self.get_current_backend().name

    def get_backend_for_type(self, dtype):
        """Return a suitable backend for the given dtype.

        If the current backend does not support the dtype, fall back
        to pocketfft.

        The *dtype* is promoted via ``result_type(dtype, 1.0)`` before
        the check, because integer/boolean inputs are automatically
        cast to float by NumPy's ufunc machinery before FFT computation.
        """
        import numpy as np
        promoted = np.result_type(dtype, 1.0)
        backend = self.get_current_backend()
        if backend.supports_type(promoted):
            return backend
        return self._backends["pocketfft"]

    def set_backend(self, name):
        """Return a context manager that temporarily switches the backend.

        Parameters
        ----------
        name : str
            Backend name, e.g. ``"pocketfft"`` or ``"kmlfft"``
            (case-insensitive).

        Returns
        -------
        _BackendContext

        Raises
        ------
        ValueError
            If the backend name is unknown.
        """
        name = name.lower()
        if name not in self._backends:
            available = ", ".join(sorted(self._backends))
            raise ValueError(
                f"Unknown FFT backend: {name!r}. "
                f"Available backends: {available}"
            )
        return _BackendContext(self, name)

    def set_global_backend(self, name):
        """Set the global default backend.

        Parameters
        ----------
        name : str
            Backend name, e.g. ``"pocketfft"`` or ``"kmlfft"``
            (case-insensitive).

        Raises
        ------
        ValueError
            If the backend name is unknown.
        """
        name = name.lower()
        if name not in self._backends:
            available = ", ".join(sorted(self._backends))
            raise ValueError(
                f"Unknown FFT backend: {name!r}. "
                f"Available backends: {available}"
            )
        with self._lock:
            self._global_backend = self._backends[name]

    def reset_backend(self):
        """Reset to the default backend (``pocketfft``)."""
        with self._lock:
            self._global_backend = None


# Singleton instance
_BACKEND_MANAGER = _BackendManager()


# Public API
def get_backend():
    """Return the name of the currently active FFT backend.

    Returns
    -------
    out : str
        The name of the backend, e.g. ``"pocketfft"`` or ``"kmlfft"``.
    """
    return _BACKEND_MANAGER.get_backend_name()


def set_backend(backend):
    """Context manager to temporarily set the FFT backend.

    Parameters
    ----------
    backend : str
        Name of the backend to use. Must be one of the available
        backends (case-insensitive).

    Returns
    -------
    _BackendContext
        A context manager that restores the previous backend on exit.

    Examples
    --------
    >>> import numpy as np
    >>> with np.fft.set_backend("kmlfft"):
    ...     result = np.fft.fft(data)
    """
    return _BACKEND_MANAGER.set_backend(backend)


def set_global_backend(backend):
    """Set the global default FFT backend.

    Parameters
    ----------
    backend : str
        Name of the backend to use as the global default.

    Examples
    --------
    >>> import numpy as np
    >>> np.fft.set_global_backend("kmlfft")
    """
    _BACKEND_MANAGER.set_global_backend(backend)


def reset_backend():
    """Reset the global backend to the system default (``pocketfft``).

    This clears any previous call to ``set_global_backend()``.
    The ``NUMPY_FFT_BACKEND`` environment variable (if set) will
    take effect again after the reset.
    """
    _BACKEND_MANAGER.reset_backend()
