/*
 * C extension module for KML FFT backend (_kml_fft_umath).
 *
 * Implements 5 GUFuncs (fft, ifft, rfft_n_even, rfft_n_odd, irfft)
 * with the same signatures as _pocketfft_umath.
 *
 * Backend selection is controlled at compile time:
 *   -DNUMPY_FFT_USE_KML  →  KML FFT (Kunpeng Math Library)
 *
 * Supports float32/complex64 and float64/complex128.
 * longdouble is not supported by either backend.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <complex>
#include <vector>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "npy_config.h"

#include "_fft_backend.h"

/* ── C++ exception to Python conversion wrapper ── */
template<PyUFuncGenericFunction cpp_ufunc>
static void
wrap_legacy_cpp_ufunc(char **args, npy_intp const *dimensions,
                      npy_intp const *steps, void *func)
{
    NPY_ALLOW_C_API_DEF
    try {
        cpp_ufunc(args, dimensions, steps, func);
    }
    catch (std::bad_alloc& e) {
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
    catch (const std::exception& e) {
        NPY_ALLOW_C_API;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        NPY_DISABLE_C_API;
    }
}

/* ── Type traits to dispatch between double/float FFT APIs ── */
template <typename T> struct kml_traits;

template <> struct kml_traits<double> {
    using real_t = double;
    using complex_t = FFT_COMPLEX;
    using plan_t = FFT_PLAN;

    static plan_t plan_dft_1d(int n, complex_t *in, complex_t *out,
                              int sign, unsigned flags) {
        return FFT_PLAN_DFT_1D(n, in, out, sign, flags);
    }
    static plan_t plan_many_dft(int rank, const int *n, int howmany,
                                complex_t *in, const int *inembed,
                                int istride, int idist,
                                complex_t *out, const int *onembed,
                                int ostride, int odist,
                                int sign, unsigned flags) {
        return FFT_PLAN_MANY_DFT(rank, n, howmany, in, inembed,
                                 istride, idist, out, onembed,
                                 ostride, odist, sign, flags);
    }
    static plan_t plan_dft_r2c(int rank, const int *n,
                               real_t *in, complex_t *out, unsigned flags) {
        return FFT_PLAN_DFT_R2C(rank, n, in, out, flags);
    }
    static plan_t plan_dft_r2c_1d(int n, real_t *in, complex_t *out,
                                  unsigned flags) {
        return FFT_PLAN_DFT_R2C_1D(n, in, out, flags);
    }
    static plan_t plan_many_dft_r2c(int rank, const int *n, int howmany,
                                    real_t *in, const int *inembed,
                                    int istride, int idist,
                                    complex_t *out, const int *onembed,
                                    int ostride, int odist,
                                    unsigned flags) {
        return FFT_PLAN_MANY_DFT_R2C(rank, n, howmany, in, inembed,
                                     istride, idist, out, onembed,
                                     ostride, odist, flags);
    }
    static plan_t plan_dft_c2r(int rank, const int *n,
                               complex_t *in, real_t *out, unsigned flags) {
        return FFT_PLAN_DFT_C2R(rank, n, in, out, flags);
    }
    static plan_t plan_dft_c2r_1d(int n, complex_t *in, real_t *out,
                                  unsigned flags) {
        return FFT_PLAN_DFT_C2R_1D(n, in, out, flags);
    }
    static plan_t plan_many_dft_c2r(int rank, const int *n, int howmany,
                                    complex_t *in, const int *inembed,
                                    int istride, int idist,
                                    real_t *out, const int *onembed,
                                    int ostride, int odist,
                                    unsigned flags) {
        return FFT_PLAN_MANY_DFT_C2R(rank, n, howmany, in, inembed,
                                     istride, idist, out, onembed,
                                     ostride, odist, flags);
    }
    static void execute_dft(plan_t p, complex_t *in, complex_t *out) {
        return FFT_EXECUTE_DFT(p, in, out);
    }
    static void execute_dft_r2c(plan_t p, real_t *in, complex_t *out) {
        return FFT_EXECUTE_DFT_R2C(p, in, out);
    }
    static void execute_dft_c2r(plan_t p, complex_t *in, real_t *out) {
        return FFT_EXECUTE_DFT_C2R(p, in, out);
    }
    static void destroy_plan(plan_t p) {
        FFT_DESTROY_PLAN(p);
    }
    static void init_threads() {
        FFT_INIT_THREADS();
    }
};

template <> struct kml_traits<float> {
    using real_t = float;
    using complex_t = FFTF_COMPLEX;
    using plan_t = FFTF_PLAN;

    static plan_t plan_dft_1d(int n, complex_t *in, complex_t *out,
                              int sign, unsigned flags) {
        return FFTF_PLAN_DFT_1D(n, in, out, sign, flags);
    }
    static plan_t plan_many_dft(int rank, const int *n, int howmany,
                                complex_t *in, const int *inembed,
                                int istride, int idist,
                                complex_t *out, const int *onembed,
                                int ostride, int odist,
                                int sign, unsigned flags) {
        return FFTF_PLAN_MANY_DFT(rank, n, howmany, in, inembed,
                                  istride, idist, out, onembed,
                                  ostride, odist, sign, flags);
    }
    static plan_t plan_dft_r2c(int rank, const int *n,
                               real_t *in, complex_t *out, unsigned flags) {
        return FFTF_PLAN_DFT_R2C(rank, n, in, out, flags);
    }
    static plan_t plan_dft_r2c_1d(int n, real_t *in, complex_t *out,
                                  unsigned flags) {
        return FFTF_PLAN_DFT_R2C_1D(n, in, out, flags);
    }
    static plan_t plan_many_dft_r2c(int rank, const int *n, int howmany,
                                    real_t *in, const int *inembed,
                                    int istride, int idist,
                                    complex_t *out, const int *onembed,
                                    int ostride, int odist,
                                    unsigned flags) {
        return FFTF_PLAN_MANY_DFT_R2C(rank, n, howmany, in, inembed,
                                      istride, idist, out, onembed,
                                      ostride, odist, flags);
    }
    static plan_t plan_dft_c2r(int rank, const int *n,
                               complex_t *in, real_t *out, unsigned flags) {
        return FFTF_PLAN_DFT_C2R(rank, n, in, out, flags);
    }
    static plan_t plan_dft_c2r_1d(int n, complex_t *in, real_t *out,
                                  unsigned flags) {
        return FFTF_PLAN_DFT_C2R_1D(n, in, out, flags);
    }
    static plan_t plan_many_dft_c2r(int rank, const int *n, int howmany,
                                    complex_t *in, const int *inembed,
                                    int istride, int idist,
                                    real_t *out, const int *onembed,
                                    int ostride, int odist,
                                    unsigned flags) {
        return FFTF_PLAN_MANY_DFT_C2R(rank, n, howmany, in, inembed,
                                      istride, idist, out, onembed,
                                      ostride, odist, flags);
    }
    static void execute_dft(plan_t p, complex_t *in, complex_t *out) {
        return FFTF_EXECUTE_DFT(p, in, out);
    }
    static void execute_dft_r2c(plan_t p, real_t *in, complex_t *out) {
        return FFTF_EXECUTE_DFT_R2C(p, in, out);
    }
    static void execute_dft_c2r(plan_t p, complex_t *in, real_t *out) {
        return FFTF_EXECUTE_DFT_C2R(p, in, out);
    }
    static void destroy_plan(plan_t p) {
        FFTF_DESTROY_PLAN(p);
    }
    static void init_threads() {
        FFTF_INIT_THREADS();
    }
};

/* ── Helper: apply scale factor to output ── */
template <typename T>
static inline void apply_fct(T *op, size_t n_elements, T fct) {
    if (fct != T(1)) {
        for (size_t i = 0; i < n_elements; i++) {
            op[i] *= fct;
        }
    }
}

template <typename T>
static inline void apply_fct_complex(std::complex<T> *op, size_t n_elements,
                                     T fct) {
    if (fct != T(1)) {
        for (size_t i = 0; i < n_elements; i++) {
            op[i] *= static_cast<typename std::complex<T>::value_type>(fct);
        }
    }
}

/* ── Copy helpers for non-contiguous input/output ── */
template <typename T>
static inline void copy_input(const char *in, npy_intp step_in, size_t nin,
                              T buff[], size_t n) {
    size_t ncopy = nin <= n ? nin : n;
    for (size_t i = 0; i < ncopy; i++, in += step_in) {
        buff[i] = *(const T *)in;
    }
    for (size_t i = ncopy; i < n; i++) {
        buff[i] = T(0);
    }
}

template <typename T>
static inline void copy_output(const T buff[], char *out, npy_intp step_out,
                               size_t n) {
    for (size_t i = 0; i < n; i++, out += step_out) {
        *(T *)out = buff[i];
    }
}

/* ── GUFunc loop: C2C (fft / ifft) ── */
template <typename T>
static void
fft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps,
         void *func)
{
    using traits = kml_traits<T>;
    using complex_t = typename traits::complex_t;
    using plan_t = typename traits::plan_t;

    char *ip = args[0], *fp = args[1], *op = args[2];
    npy_intp n_outer = dimensions[0];
    npy_intp nin = dimensions[1], nout = dimensions[2];
    npy_intp si = steps[0], so = steps[2];
    npy_intp step_in = steps[3], step_out = steps[4];
    int direction = *((int *)func);
    T fct = *((T *)fp);

    bool contiguous_in = (step_in == sizeof(complex_t));
    bool contiguous_out = (step_out == sizeof(complex_t));

    for (npy_intp i = 0; i < n_outer; i++) {
        complex_t *pin = (complex_t *)(ip + i * si);
        complex_t *pout = (complex_t *)(op + i * so);
        plan_t p = traits::plan_dft_1d((int)nout, pin, pout,
            direction, FFT_ESTIMATE);
        if (!p) continue;

        if (contiguous_in && contiguous_out && pin != pout && nin >= nout) {
            traits::execute_dft(p, pin, pout);
        } else {
            std::vector<std::complex<T>> buf(nout);
            copy_input((char *)(ip + i * si), step_in, nin,
                       (std::complex<T> *)buf.data(), nout);
            traits::execute_dft(p,
                (complex_t *)buf.data(), (complex_t *)buf.data());
            apply_fct_complex(buf.data(), nout, fct);
            copy_output(buf.data(), (char *)(op + i * so), step_out, nout);
            traits::destroy_plan(p);
            continue;
        }
        apply_fct_complex((std::complex<T> *)pout, nout, fct);
        traits::destroy_plan(p);
    }
}

/* ── GUFunc loop: R2C (rfft_n_even / rfft_n_odd) ── */
template <typename T>
static void
rfft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void * /* unused */, size_t npts)
{
    using traits = kml_traits<T>;
    using real_t = typename traits::real_t;
    using complex_t = typename traits::complex_t;
    using plan_t = typename traits::plan_t;

    char *ip = args[0], *fp = args[1], *op = args[2];
    npy_intp n_outer = dimensions[0];
    npy_intp nin = dimensions[1], nout = dimensions[2];
    npy_intp si = steps[0], so = steps[2];
    npy_intp step_in = steps[3], step_out = steps[4];
    T fct = *((T *)fp);

    bool contiguous_in = (step_in == sizeof(real_t));
    bool contiguous_out = (step_out == sizeof(complex_t));
    size_t nin_used = (size_t)nin <= npts ? (size_t)nin : npts;

    for (npy_intp i = 0; i < n_outer; i++) {
        if (contiguous_in && contiguous_out && (size_t)nin >= npts) {
            real_t *pin = (real_t *)(ip + i * si);
            complex_t *pout = (complex_t *)(op + i * so);
            plan_t p = traits::plan_dft_r2c_1d((int)npts,
                pin, pout, FFT_ESTIMATE);
            if (!p) continue;
            traits::execute_dft_r2c(p, pin, pout);
            traits::destroy_plan(p);
            apply_fct_complex((std::complex<T> *)pout, nout, fct);
        } else {
            std::vector<real_t> rbuf(npts);
            std::vector<std::complex<T>> cbuf(nout);
            copy_input<real_t>(ip + i * si, step_in, nin_used,
                               rbuf.data(), npts);
            plan_t p = traits::plan_dft_r2c_1d((int)npts,
                rbuf.data(), (complex_t *)cbuf.data(),
                FFT_ESTIMATE);
            if (!p) continue;
            traits::execute_dft_r2c(p, rbuf.data(),
                (complex_t *)cbuf.data());
            traits::destroy_plan(p);
            apply_fct_complex(cbuf.data(), nout, fct);
            copy_output(cbuf.data(), op + i * so, step_out, nout);
        }
    }
}

template <typename T>
static void rfft_n_even_loop(char **args, npy_intp const *dimensions,
                             npy_intp const *steps, void *func) {
    size_t nout = (size_t)dimensions[2];
    size_t npts = 2 * nout - 2;
    rfft_loop<T>(args, dimensions, steps, func, npts);
}

template <typename T>
static void rfft_n_odd_loop(char **args, npy_intp const *dimensions,
                            npy_intp const *steps, void *func) {
    size_t nout = (size_t)dimensions[2];
    size_t npts = 2 * nout - 1;
    rfft_loop<T>(args, dimensions, steps, func, npts);
}

/* ── GUFunc loop: C2R (irfft) ── */
template <typename T>
static void
irfft_loop(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void * /* unused */)
{
    using traits = kml_traits<T>;
    using real_t = typename traits::real_t;
    using complex_t = typename traits::complex_t;
    using plan_t = typename traits::plan_t;

    char *ip = args[0], *fp = args[1], *op = args[2];
    npy_intp n_outer = dimensions[0];
    npy_intp nin = dimensions[1], nout = dimensions[2];
    npy_intp si = steps[0], so = steps[2];
    npy_intp step_in = steps[3], step_out = steps[4];
    T fct = *((T *)fp);

    bool contiguous_in = (step_in == sizeof(complex_t));
    bool contiguous_out = (step_out == sizeof(real_t));

    for (npy_intp i = 0; i < n_outer; i++) {
        if (contiguous_in && contiguous_out && (size_t)nin >= (size_t)(nout / 2 + 1)) {
            complex_t *pin = (complex_t *)(ip + i * si);
            real_t *pout = (real_t *)(op + i * so);
            plan_t p = traits::plan_dft_c2r_1d((int)nout,
                pin, pout, FFT_ESTIMATE);
            if (!p) continue;
            traits::execute_dft_c2r(p, pin, pout);
            traits::destroy_plan(p);
            apply_fct(pout, nout, fct);
        } else {
            size_t nin_expected = (size_t)(nout / 2 + 1);
            std::vector<std::complex<T>> cbuf(nin_expected);
            std::vector<real_t> rbuf(nout);
            copy_input(ip + i * si, step_in, nin,
                       (std::complex<T> *)cbuf.data(), nin_expected);
            plan_t p = traits::plan_dft_c2r_1d((int)nout,
                (complex_t *)cbuf.data(), rbuf.data(),
                FFT_ESTIMATE);
            if (!p) continue;
            traits::execute_dft_c2r(p,
                (complex_t *)cbuf.data(), rbuf.data());
            traits::destroy_plan(p);
            apply_fct(rbuf.data(), nout, fct);
            copy_output(rbuf.data(), op + i * so, step_out, nout);
        }
    }
}

/* ── GUFunc registration ── */

/* Direction data (passed as void* func) */
static int kml_forward_d = FFT_FORWARD;
static int kml_backward_d = FFT_BACKWARD;

/* C2C fft / ifft — 2 types: double, float */
static PyUFuncGenericFunction fft_functions[] = {
    wrap_legacy_cpp_ufunc<fft_loop<double>>,
    wrap_legacy_cpp_ufunc<fft_loop<float>>,
};
static const char fft_types[] = {
    NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_CFLOAT,  NPY_FLOAT,  NPY_CFLOAT,
};
static void *const fft_data[] = {
    &kml_forward_d,
    &kml_forward_d,
};
static void *const ifft_data[] = {
    &kml_backward_d,
    &kml_backward_d,
};

/* R2C rfft — 2 types: double, float */
static PyUFuncGenericFunction rfft_n_even_functions[] = {
    wrap_legacy_cpp_ufunc<rfft_n_even_loop<double>>,
    wrap_legacy_cpp_ufunc<rfft_n_even_loop<float>>,
};
static PyUFuncGenericFunction rfft_n_odd_functions[] = {
    wrap_legacy_cpp_ufunc<rfft_n_odd_loop<double>>,
    wrap_legacy_cpp_ufunc<rfft_n_odd_loop<float>>,
};
static const char rfft_types[] = {
    NPY_DOUBLE, NPY_DOUBLE, NPY_CDOUBLE,
    NPY_FLOAT,  NPY_FLOAT,  NPY_CFLOAT,
};

/* C2R irfft — 2 types: double, float */
static PyUFuncGenericFunction irfft_functions[] = {
    wrap_legacy_cpp_ufunc<irfft_loop<double>>,
    wrap_legacy_cpp_ufunc<irfft_loop<float>>,
};
static const char irfft_types[] = {
    NPY_CDOUBLE, NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT,  NPY_FLOAT,  NPY_FLOAT,
};

static int
add_gufuncs(PyObject *dictionary) {
    PyObject *f;
    int ntypes = 2; /* double + float, no longdouble */

    f = PyUFunc_FromFuncAndDataAndSignature(
        fft_functions, fft_data, fft_types, ntypes, 2, 1, PyUFunc_None,
        "fft", "FFT complex forward\n", 0, "(n),()->(m)");
    if (f == NULL) return -1;
    PyDict_SetItemString(dictionary, "fft", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        fft_functions, ifft_data, fft_types, ntypes, 2, 1, PyUFunc_None,
        "ifft", "FFT complex backward\n", 0, "(m),()->(n)");
    if (f == NULL) return -1;
    PyDict_SetItemString(dictionary, "ifft", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        rfft_n_even_functions, NULL, rfft_types, ntypes, 2, 1, PyUFunc_None,
        "rfft_n_even", "FFT real forward for even n\n", 0, "(n),()->(m)");
    if (f == NULL) return -1;
    PyDict_SetItemString(dictionary, "rfft_n_even", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        rfft_n_odd_functions, NULL, rfft_types, ntypes, 2, 1, PyUFunc_None,
        "rfft_n_odd", "FFT real forward for odd n\n", 0, "(n),()->(m)");
    if (f == NULL) return -1;
    PyDict_SetItemString(dictionary, "rfft_n_odd", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(
        irfft_functions, NULL, irfft_types, ntypes, 2, 1, PyUFunc_None,
        "irfft", "FFT real backward\n", 0, "(m),()->(n)");
    if (f == NULL) return -1;
    PyDict_SetItemString(dictionary, "irfft", f);
    Py_DECREF(f);

    return 0;
}

/* ── Module init ── */
static int module_loaded = 0;

static int
_kml_fft_umath_exec(PyObject *m)
{
    if (module_loaded) {
        PyErr_SetString(PyExc_ImportError,
                        "cannot load module more than once per process");
        return -1;
    }
    module_loaded = 1;

    if (PyArray_ImportNumPyAPI() < 0) return -1;
    if (PyUFunc_ImportUFuncAPI() < 0) return -1;

    /* Initialize FFT library threading */
    kml_traits<double>::init_threads();
    kml_traits<float>::init_threads();

    PyObject *d = PyModule_GetDict(m);
    if (add_gufuncs(d) < 0) {
        return -1;
    }
    return 0;
}

static struct PyModuleDef_Slot _kml_fft_umath_slots[] = {
    {Py_mod_exec, (void *)_kml_fft_umath_exec},
#if PY_VERSION_HEX >= 0x030c00f0
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030d00f0
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_kml_fft_umath",
    NULL,
    0,
    NULL,
    _kml_fft_umath_slots,
};

PyMODINIT_FUNC PyInit__kml_fft_umath(void) {
    return PyModuleDef_Init(&moduledef);
}
