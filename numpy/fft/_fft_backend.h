/*
 * FFT backend abstraction header.
 *
 * Provides uniform macro names so the same C++ source compiles
 * against different FFT libraries selected at build time.
 *
 * Reference: NumPy BLAS abstraction (npy_cblas_base.h).
 */
#pragma once

#ifdef NUMPY_FFT_USE_KML

#include "kfft.h"

/* ── double precision ── */
#define FFT_COMPLEX            kml_fft_complex
#define FFT_PLAN               kml_fft_plan
#define FFT_PLAN_DFT_1D        kml_fft_plan_dft_1d
#define FFT_PLAN_MANY_DFT      kml_fft_plan_many_dft
#define FFT_PLAN_DFT_R2C       kml_fft_plan_dft_r2c
#define FFT_PLAN_DFT_R2C_1D    kml_fft_plan_dft_r2c_1d
#define FFT_PLAN_MANY_DFT_R2C  kml_fft_plan_many_dft_r2c
#define FFT_PLAN_DFT_C2R       kml_fft_plan_dft_c2r
#define FFT_PLAN_DFT_C2R_1D    kml_fft_plan_dft_c2r_1d
#define FFT_PLAN_MANY_DFT_C2R  kml_fft_plan_many_dft_c2r
#define FFT_EXECUTE_DFT        kml_fft_execute_dft
#define FFT_EXECUTE_DFT_R2C    kml_fft_execute_dft_r2c
#define FFT_EXECUTE_DFT_C2R    kml_fft_execute_dft_c2r
#define FFT_DESTROY_PLAN       kml_fft_destroy_plan
#define FFT_INIT_THREADS       kml_fft_init_threads
#define FFT_ESTIMATE           KML_FFT_ESTIMATE
#define FFT_FORWARD            KML_FFT_FORWARD
#define FFT_BACKWARD           KML_FFT_BACKWARD

/* ── single precision ── */
#define FFTF_COMPLEX            kml_fftf_complex
#define FFTF_PLAN               kml_fftf_plan
#define FFTF_PLAN_DFT_1D        kml_fftf_plan_dft_1d
#define FFTF_PLAN_MANY_DFT      kml_fftf_plan_many_dft
#define FFTF_PLAN_DFT_R2C       kml_fftf_plan_dft_r2c
#define FFTF_PLAN_DFT_R2C_1D    kml_fftf_plan_dft_r2c_1d
#define FFTF_PLAN_MANY_DFT_R2C  kml_fftf_plan_many_dft_r2c
#define FFTF_PLAN_DFT_C2R       kml_fftf_plan_dft_c2r
#define FFTF_PLAN_DFT_C2R_1D    kml_fftf_plan_dft_c2r_1d
#define FFTF_PLAN_MANY_DFT_C2R  kml_fftf_plan_many_dft_c2r
#define FFTF_EXECUTE_DFT        kml_fftf_execute_dft
#define FFTF_EXECUTE_DFT_R2C    kml_fftf_execute_dft_r2c
#define FFTF_EXECUTE_DFT_C2R    kml_fftf_execute_dft_c2r
#define FFTF_DESTROY_PLAN       kml_fftf_destroy_plan
#define FFTF_INIT_THREADS       kml_fftf_init_threads

#else
#error "Unknown FFT backend. Build with -DNUMPY_FFT_USE_KML"
#endif
