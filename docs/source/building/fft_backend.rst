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
                              -Csetup-args=-Dfft-include-dir=/path/to/kml/gcc/include \
                              -Csetup-args=-Dfft-lib-dir=/path/to/kml/gcc/lib/<cpu-isa>

    $ # Or with spin:
    $ spin build -- -Dfft-backend=kml \
                     -Dfft-include-dir=/path/to/kml/gcc/include \
                     -Dfft-lib-dir=/path/to/kml/gcc/lib/<cpu-isa>

KML installations include multiple CPU-specific library directories, such as
``neon``, ``sve``, ``sve_no_f64mm``, and ``sve512``. The value of
``fft-lib-dir`` must be the directory selected for the current CPU, not the
parent ``lib`` directory. Source the KML environment script first; it adds
the appropriate directory to ``LD_LIBRARY_PATH``. For example, a standalone
GCC installation may use::

    $ source /opt/kml/gcc/env/setvars.sh

An HPCKit installation may instead use::

    $ source /opt/HPCKit/latest/setvars.sh --use-gcc

After sourcing the environment, select the KML-owned entry in
``LD_LIBRARY_PATH`` that contains both ``libkfft.so`` and
``libkfftf.so``, and pass that entry as ``fft-lib-dir``.

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

Alternatively, set the environment variable before starting Python::

    $ NUMPY_FFT_BACKEND=kmlfft python your_script.py

See :ref:`fft-backend-architecture` for details on the backend dispatch
architecture.


Full list of FFT build options
------------------------------

The following build options are defined in ``meson.options``:

- ``fft-backend``: External FFT backend library (default: ``kml``).
  Currently only ``kml`` is available. PocketFFT is always built, regardless
  of this option.
- ``fft-include-dir``: Path to the FFT library include directory
  containing ``kfft.h``.
- ``fft-lib-dir``: Path to the FFT library link directory containing
  ``libkfft.so`` and ``libkfftf.so``. Can be an absolute path or a path
  relative to the project source root.

When ``fft-include-dir`` or ``fft-lib-dir`` are left empty, the external
FFT extension is not built, regardless of the ``fft-backend`` option value.


pkg-config detection
--------------------

.. note::

    The KML package supported here does not currently ship a ``.pc`` file.
    KML FFT detection therefore uses ``find_library`` with explicit
    ``fft-include-dir`` and ``fft-lib-dir`` values.
