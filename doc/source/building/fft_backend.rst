.. _building-fft-backend:

External FFT Backend (KML FFT)
===============================

NumPy supports building with an optional external FFT backend library to
accelerate FFT computations. Currently, the `KML FFT`_ (Kunpeng Math Library)
backend is supported.

.. _KML FFT: https://www.hikunpeng.com/developer/boostkit/kunpengaccel


Default behavior for FFT backend selection
------------------------------------------

By default, NumPy uses its built-in PocketFFT library for all FFT computations.
No external FFT library is required.

To enable the KML FFT backend, you must explicitly configure the build with
the ``fft-backend`` option and provide the include and library directories::

    $ python -m pip install . -Csetup-args=-Dfft-backend=kml \
                              -Csetup-args=-Dfft-include-dir=/path/to/kml/include \
                              -Csetup-args=-Dfft-lib-dir=/path/to/kml/lib

    $ # Or with spin:
    $ spin build -- -Dfft-backend=kml \
                     -Dfft-include-dir=/path/to/kml/include \
                     -Dfft-lib-dir=/path/to/kml/lib

The KML FFT library provides two shared libraries:

- ``libkfft.so`` (double precision)
- ``libkfftf.so`` (single precision)

Both must be available in the specified library directory. The build system
links against both and produces a ``_kml_fft_umath`` C extension module.

At runtime, the KML backend is available only when the C extension module
was successfully built and is importable. If the module cannot be imported,
the KML backend is silently excluded, and ``pocketfft`` remains the sole
backend.


Runtime backend selection
-------------------------

Once built with KML FFT support, the backend can be switched at runtime:

.. code-block:: python

    import numpy as np

    # Check current backend
    print(np.fft.get_backend())  # 'pocketfft'

    # Set KML FFT as global backend
    np.fft.set_global_backend('kmlfft')

    # Or use as a context manager for a temporary switch
    with np.fft.set_backend('kmlfft'):
        result = np.fft.fft(data)

    # Or set via environment variable
    export NUMPY_FFT_BACKEND=kmlfft

See :ref:`fft-backend-architecture` for details on the backend dispatch
architecture.


Full list of FFT build options
------------------------------

The following build options are defined in ``meson.options``:

- ``fft-backend``: External FFT backend library (default: ``kml``).
  Currently only ``kml`` is available. If left unset, only the built-in
  PocketFFT backend is built.
- ``fft-include-dir``: Path to the FFT library include directory
  containing ``kfft.h``.
- ``fft-lib-dir``: Path to the FFT library link directory containing
  ``libkfft.so`` and ``libkfftf.so``. Can be an absolute path or a path
  relative to the project source root.

When ``fft-include-dir`` or ``fft-lib-dir`` are left empty, the external
FFT extension is not built, regardless of the ``fft-backend`` option value.


Using pkg-config (future)
-------------------------

.. note::

    Currently, KML FFT library detection uses ``find_library`` with explicit
    directories. Future versions may support ``pkg-config``-based detection
    similar to the BLAS/LAPACK build system.
