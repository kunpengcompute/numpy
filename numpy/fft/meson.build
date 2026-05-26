largefile_define = []
if host_machine.system() == 'aix' or host_machine.system() == 'AIX'
  largefile_define += '-D_LARGE_FILES'
endif

if not fs.exists('pocketfft/README.md')
  error('Missing the `pocketfft` git submodule! Run `git submodule update --init` to fix this.')
endif

py.extension_module('_pocketfft_umath',
  ['_pocketfft_umath.cpp'],
  c_args: largefile_define,
  dependencies: np_core_dep,
  install: true,
  subdir: 'numpy/fft',
)

# ── External FFT extension (e.g. KML FFT) ──
fft_backend_opt = get_option('fft-backend')
fft_inc_opt = get_option('fft-include-dir')
fft_lib_opt = get_option('fft-lib-dir')

if fft_backend_opt == 'kml' and fft_inc_opt != '' and fft_lib_opt != ''
  fft_lib_dir = fft_lib_opt.startswith('/') ? fft_lib_opt \
    : meson.project_source_root() / fft_lib_opt

  kfft_dep = cc.find_library('kfft', dirs: fft_lib_dir, required: true)
  kfftf_dep = cc.find_library('kfftf', dirs: fft_lib_dir, required: true)

  py.extension_module('_kml_fft_umath',
    ['_kml_fft_umath.cpp'],
    cpp_args: largefile_define + ['-DNUMPY_FFT_USE_KML',
                                   '-I' + meson.project_source_root() / fft_inc_opt],
    dependencies: [np_core_dep, kfft_dep, kfftf_dep],
    install: true,
    subdir: 'numpy/fft',
  )
endif

py.install_sources(
  [
    '__init__.py',
    '__init__.pyi',
    '_backend.py',
    '_pocketfft.py',
    '_pocketfft.pyi',
    '_kml_fft.py',
    '_helper.py',
    '_helper.pyi',
  ],
  subdir: 'numpy/fft'
)

py.install_sources(
  [
    'tests/__init__.py',
    'tests/test_helper.py',
    'tests/test_pocketfft.py',
    'tests/test_fft_backend.py',
  ],
  subdir: 'numpy/fft/tests',
  install_tag: 'tests'
)
