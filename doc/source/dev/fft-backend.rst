.. _fft-backend-architecture:

========================
FFT Backend Architecture
========================

NumPy's ``numpy.fft`` module supports runtime switching between multiple
FFT backend implementations. This document describes the architecture
and how to integrate a new backend.

Architecture overview
---------------------

The backend system follows a layered dispatch pattern:

.. code-block:: text

    numpy.fft.fft / ifft / rfft / ...
           │
           ▼
    _pocketfft.py._raw_fft()  ──  1-D dispatch entry
           │
           ├── backend is pocketfft  ──  existing PocketFFT path
           │
           └── backend is kmlfft  ──  _kml_fft.py._raw_fft()
                                             │
                                             ▼
                                    _kml_fft_umath GUFuncs (C extension via KML FFT library)

Backend priority
----------------

The effective backend is determined by the following priority chain
(highest to lowest):

1. **Thread-local backend** — set via ``np.fft.set_backend(name)`` context
   manager, isolated per thread via ``threading.local()``.
2. **Global backend** — set via ``np.fft.set_global_backend(name)``.
3. **Environment variable** — ``NUMPY_FFT_BACKEND``, read once at import time.
4. **System default** — ``pocketfft`` (always available).

Per-dtype fallback
------------------

The 1-D FFT path (``_raw_fft``) uses ``get_backend_for_type(dtype)``,
which checks whether the current backend supports the promoted dtype.
If not, it silently falls back to ``pocketfft``. This ensures that
unsupported dtypes (e.g., ``float16`` when KML doesn't handle it) still
work correctly.

The N-D FFT path (``_raw_fftnd``) uses ``get_current_backend()`` without
type filtering, because individual sub-transforms (1-D) will hit the
per-dtype fallback on their own.

Key components
--------------

``_backend.py``
    Contains ``_BackendManager`` (singleton), ``FFTBackend`` (abstract
    base class), ``PocketFFTBackend``, ``KMLFFTBackend``, and the four
    public API functions.

``_pocketfft.py``
    Modified ``_raw_fft`` and ``_raw_fftnd`` to dispatch to the current
    backend via ``_BACKEND_MANAGER``. The original PocketFFT logic is
    preserved unchanged after the dispatch check.

``_kml_fft.py``
    Python wrapper for the KML C extension. Provides ``_raw_fft`` and
    ``_raw_fftnd`` with the same signature as ``_pocketfft.py``.
    Handles KML-specific normalization (KML does not apply ``1/n``
    internally for inverse transforms).

``_kml_fft_umath.cpp``
    C++ extension implementing 5 GUFuncs (``fft``, ``ifft``,
    ``rfft_n_even``, ``rfft_n_odd``, ``irfft``) using KML FFT library
    calls. Uses template-based type traits to dispatch between
    single- and double-precision API variants.

``_fft_backend.h``
    API abstraction header that maps uniform ``FFT_*`` / ``FFTF_*``
    macros to the selected backend's native API. Currently supports
    KML FFT (compile with ``-DNUMPY_FFT_USE_KML``).

Adding a new backend
--------------------

To add a new FFT backend:

1. **Meson build options** — Add the new backend name to the
   ``fft-backend`` combo option in ``meson.options``, and add a
   build branch in ``numpy/fft/meson.build`` that passes
   ``-DNUMPY_FFT_USE_{BACKEND}``.

2. **Abstraction header** — Add a corresponding ``#ifdef`` block in
   ``_fft_backend.h`` mapping the new backend's API functions to the
   ``FFT_*`` / ``FFTF_*`` macros.

3. **Backend class** — Subclass ``FFTBackend`` and implement
   ``name``, ``_raw_fft``, ``_raw_fftnd``, and optionally
   ``supports_type`` and ``supports_norm``.

4. **Registration** — Register the backend in
   ``_BackendManager.__init__`` inside a ``try/except ImportError``
   block so it is only available when the C extension is importable.

5. **Tests** — Add backend-specific tests in
   ``numpy/fft/tests/test_fft_backend.py``.

GIL management
--------------

The NumPy GUFunc framework automatically releases the GIL before calling
inner loop functions when the iterator does not have the
``NPY_ITER_REFS_OK`` flag set. Backend loop functions should **not**
manually release the GIL, as this would cause a double-release and
segfault.

Environment variable
--------------------

The ``NUMPY_FFT_BACKEND`` environment variable is read once at module
import time. Invalid backend names issue a ``RuntimeWarning`` and fall
back to ``pocketfft``. The ``reset_backend()`` function only clears
explicit ``set_global_backend()`` calls — the environment variable
value is preserved as a separate layer and takes effect again after the
reset.

Error handling
--------------

- Calling ``set_backend()`` or ``set_global_backend()`` with an unknown
  or unregistered backend name raises ``ValueError``.
- Backend initialization checks C extension availability at registration
  time via an import attempt in ``__init__``. If the C module is not
  importable, the backend is silently excluded from the registry.
