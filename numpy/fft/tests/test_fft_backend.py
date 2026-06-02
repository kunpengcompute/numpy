"""
Tests for the FFT backend management API.

Covers: get_backend, set_backend, set_global_backend, reset_backend,
and NUMPY_FFT_BACKEND environment variable.
"""
import threading
import warnings

import numpy as np
import pytest

from numpy.fft._backend import _BACKEND_MANAGER

KMLFFT_AVAILABLE = "kmlfft" in _BACKEND_MANAGER._backends


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_env_var(name):
    """Simulate NUMPY_FFT_BACKEND env var being set to *name*.

    Replicates the processing logic from _BackendManager.__init__.
    Issues RuntimeWarning for unregistered backends.
    """
    name = name.lower()
    if name in _BACKEND_MANAGER._backends:
        _BACKEND_MANAGER._env_backend = _BACKEND_MANAGER._backends[name]
    else:
        available = ", ".join(sorted(_BACKEND_MANAGER._backends))
        warnings.warn(
            f"NUMPY_FFT_BACKEND={name!r} is not a registered "
            f"backend. Available: {available}. "
            f"Falling back to default (pocketfft).",
            RuntimeWarning,
            stacklevel=2,
        )


@pytest.fixture(autouse=True)
def _reset_backend_state():
    """Ensure clean backend state for every test."""
    _BACKEND_MANAGER.reset_backend()
    _BACKEND_MANAGER._env_backend = None
    yield
    _BACKEND_MANAGER.reset_backend()
    _BACKEND_MANAGER._env_backend = None


# ---------------------------------------------------------------------------
# get_backend
# ---------------------------------------------------------------------------

class TestGetBackend:
    """Tests for get_backend()."""

    def test_default_returns_pocketfft(self):
        """With no special config, get_backend returns pocketfft."""
        assert np.fft.get_backend() == "pocketfft"

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_global_overrides_env(self):
        """When both env var and global backend are set, global takes effect."""
        _simulate_env_var("pocketfft")
        np.fft.set_global_backend("kmlfft")
        assert np.fft.get_backend() == "kmlfft"

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_context_overrides_all(self):
        """When env var, global, and context are all set, context wins."""
        _simulate_env_var("pocketfft")
        np.fft.set_global_backend("kmlfft")
        with np.fft.set_backend("pocketfft"):
            assert np.fft.get_backend() == "pocketfft"
        assert np.fft.get_backend() == "kmlfft"

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_context_overrides_global(self):
        """When both global and context are set, context wins."""
        np.fft.set_global_backend("kmlfft")
        with np.fft.set_backend("pocketfft"):
            assert np.fft.get_backend() == "pocketfft"
        assert np.fft.get_backend() == "kmlfft"

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_context_overrides_env(self):
        """When both env var and context are set, context wins."""
        _simulate_env_var("pocketfft")
        assert np.fft.get_backend() == "pocketfft"
        with np.fft.set_backend("kmlfft"):
            assert np.fft.get_backend() == "kmlfft"


# ---------------------------------------------------------------------------
# set_backend
# ---------------------------------------------------------------------------

class TestSetBackend:
    """Tests for set_backend()."""

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_restores_previous_on_exit(self):
        """set_backend restores previous backend on context exit."""
        with np.fft.set_backend("kmlfft"):
            assert np.fft.get_backend() == "kmlfft"
        assert np.fft.get_backend() == "pocketfft"

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_nested_last_wins(self):
        """Nested set_backend calls: innermost wins, restores outer on exit."""
        np.fft.set_global_backend("kmlfft")
        with np.fft.set_backend("KMLfft"):
            assert np.fft.get_backend() == "kmlfft"
            with np.fft.set_backend("POCKETFFT"):
                assert np.fft.get_backend() == "pocketfft"
            assert np.fft.get_backend() == "kmlfft"
        assert np.fft.get_backend() == "kmlfft"

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_kmlfft_backend_works(self):
        """set_backend('kmlfft') computes FFT successfully."""
        data = np.random.default_rng(42).standard_normal(128).astype(
            np.complex128
        )
        with np.fft.set_backend("kmlfft"):
            assert np.fft.get_backend() == "kmlfft"
            result = np.fft.fft(data)
        expected = np.fft.fft(data)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_invalid_backend_raises_value_error(self):
        """set_backend with invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown FFT backend"):
            with np.fft.set_backend("kmlfftfft"):
                pass
        assert np.fft.get_backend() == "pocketfft"

    def test_unregistered_backend_raises_value_error(self):
        """set_backend with unregistered backend raises ValueError."""
        kml_backend = _BACKEND_MANAGER._backends.pop("kmlfft", None)
        try:
            with pytest.raises(ValueError, match="Unknown FFT backend"):
                with np.fft.set_backend("kmlfft"):
                    pass
            assert np.fft.get_backend() == "pocketfft"
        finally:
            if kml_backend is not None:
                _BACKEND_MANAGER._backends["kmlfft"] = kml_backend

    def test_empty_string_raises_value_error(self):
        """set_backend with empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown FFT backend"):
            with np.fft.set_backend(""):
                pass
        assert np.fft.get_backend() == "pocketfft"

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_thread_isolation(self):
        """Thread-local backends do not interfere with each other."""
        results = {}
        barrier = threading.Barrier(3)

        def worker(name, backend_name):
            with np.fft.set_backend(backend_name):
                barrier.wait()
                results[name] = np.fft.get_backend()

        threads = [
            threading.Thread(target=worker, args=(f"t{i}", name))
            for i, name in enumerate(["kmlfft", "pocketfft", "kmlfft"],
                                     start=1)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["t1"] == "kmlfft"
        assert results["t2"] == "pocketfft"
        assert results["t3"] == "kmlfft"
        assert np.fft.get_backend() == "pocketfft"


# ---------------------------------------------------------------------------
# set_global_backend
# ---------------------------------------------------------------------------

class TestSetGlobalBackend:
    """Tests for set_global_backend()."""

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_set_kmlfft_and_compute(self):
        """set_global_backend('kmlfft') computes FFT successfully."""
        data = np.random.default_rng(42).standard_normal(64).astype(
            np.complex128
        )
        np.fft.set_global_backend("kmlfft")
        try:
            assert np.fft.get_backend() == "kmlfft"
            result = np.fft.fft(data)
            np.fft.reset_backend()
            expected = np.fft.fft(data)
            np.testing.assert_allclose(result, expected, rtol=1e-10)
        finally:
            np.fft.reset_backend()

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_last_call_wins(self):
        """Last set_global_backend call takes effect."""
        np.fft.set_global_backend("kmlfft")
        assert np.fft.get_backend() == "kmlfft"
        np.fft.set_global_backend("pocketfft")
        assert np.fft.get_backend() == "pocketfft"

    def test_invalid_backend_raises_value_error(self):
        """set_global_backend with invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown FFT backend"):
            np.fft.set_global_backend("kml")
        assert np.fft.get_backend() == "pocketfft"

    def test_unregistered_backend_raises_value_error(self):
        """set_global_backend with unregistered backend raises ValueError."""
        kml_backend = _BACKEND_MANAGER._backends.pop("kmlfft", None)
        try:
            with pytest.raises(ValueError, match="Unknown FFT backend"):
                np.fft.set_global_backend("kmlfft")
            assert np.fft.get_backend() == "pocketfft"
        finally:
            if kml_backend is not None:
                _BACKEND_MANAGER._backends["kmlfft"] = kml_backend

    def test_empty_string_raises_value_error(self):
        """set_global_backend with empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown FFT backend"):
            np.fft.set_global_backend("")
        assert np.fft.get_backend() == "pocketfft"


# ---------------------------------------------------------------------------
# reset_backend
# ---------------------------------------------------------------------------

class TestResetBackend:
    """Tests for reset_backend()."""

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_reset_clears_global_backend(self):
        """reset_backend clears global backend and restores default."""
        np.fft.set_global_backend("kmlfft")
        assert np.fft.get_backend() == "kmlfft"
        np.fft.reset_backend()
        assert np.fft.get_backend() == "pocketfft"

    @pytest.mark.skipif(not KMLFFT_AVAILABLE, reason="kmlfft backend not available")
    def test_reset_preserves_env_var(self):
        """reset_backend clears global but preserves env var backend."""
        _simulate_env_var("kmlfft")
        np.fft.set_global_backend("pocketfft")
        assert np.fft.get_backend() == "pocketfft"
        np.fft.reset_backend()
        assert np.fft.get_backend() == "kmlfft"


@pytest.mark.skipif(KMLFFT_AVAILABLE, reason="only relevant when kmlfft is not available")
class TestNoKMLFFT:
    """Tests that verify correct behaviour when kmlfft is NOT available.

    These only run when the build did not include the KML FFT backend.
    """

    def test_set_backend_kmlfft_raises_value_error(self):
        """set_backend('kmlfft') raises ValueError when kmlfft is not compiled."""
        with pytest.raises(ValueError, match="Unknown FFT backend"):
            with np.fft.set_backend("kmlfft"):
                pass

    def test_set_global_backend_kmlfft_raises_value_error(self):
        """set_global_backend('kmlfft') raises ValueError when kmlfft is not compiled."""
        with pytest.raises(ValueError, match="Unknown FFT backend"):
            np.fft.set_global_backend("kmlfft")

    def test_env_var_kmlfft_warns_and_falls_back(self):
        """NUMPY_FFT_BACKEND=kmlfft issues RuntimeWarning, falls back to pocketfft."""
        with pytest.warns(RuntimeWarning, match="NUMPY_FFT_BACKEND"):
            _simulate_env_var("kmlfft")
        assert np.fft.get_backend() == "pocketfft"

    def test_fft_computation_with_pocketfft_only(self):
        """FFT functions produce correct results with only pocketfft available."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(64).astype(np.complex128)
        result = np.fft.fft(data)
        expected = np.fft.ifft(np.fft.fft(data))
        np.testing.assert_allclose(np.fft.ifft(result), data, rtol=1e-10)
        np.testing.assert_allclose(expected, data, rtol=1e-10)


# ---------------------------------------------------------------------------
# NUMPY_FFT_BACKEND environment variable
# ---------------------------------------------------------------------------

class TestEnvVarBackend:
    """Tests for NUMPY_FFT_BACKEND environment variable behaviour."""

    def test_invalid_env_var_backend(self):
        """Invalid env var backend issues RuntimeWarning, falls back to default."""
        with pytest.warns(RuntimeWarning, match="NUMPY_FFT_BACKEND"):
            _simulate_env_var("pocket")
        assert np.fft.get_backend() == "pocketfft"

    def test_unregistered_env_var_backend(self):
        """Unregistered env var backend issues RuntimeWarning, falls back."""
        kml_backend = _BACKEND_MANAGER._backends.pop("kmlfft", None)
        try:
            with pytest.warns(RuntimeWarning, match="NUMPY_FFT_BACKEND"):
                _simulate_env_var("kmlfft")
            assert np.fft.get_backend() == "pocketfft"
        finally:
            if kml_backend is not None:
                _BACKEND_MANAGER._backends["kmlfft"] = kml_backend
