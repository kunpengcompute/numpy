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


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class TestFFTBackendAbstract:
    """Tests for FFTBackend abstract base class methods."""

    def test_name_raises_not_implemented(self):
        from numpy.fft._backend import FFTBackend
        backend = FFTBackend()
        with pytest.raises(NotImplementedError):
            _ = backend.name

    def test_raw_fft_raises_not_implemented(self):
        from numpy.fft._backend import FFTBackend
        backend = FFTBackend()
        with pytest.raises(NotImplementedError):
            backend._raw_fft(None, 0, 0, False, True, None)

    def test_raw_fftnd_raises_not_implemented(self):
        from numpy.fft._backend import FFTBackend
        backend = FFTBackend()
        with pytest.raises(NotImplementedError):
            backend._raw_fftnd(None)

    def test_supports_norm_valid(self):
        from numpy.fft._backend import FFTBackend
        backend = FFTBackend()
        assert backend.supports_norm(None) is True
        assert backend.supports_norm("backward") is True
        assert backend.supports_norm("ortho") is True
        assert backend.supports_norm("forward") is True
        assert backend.supports_norm("invalid") is False

    def test_supports_type_default(self):
        from numpy.fft._backend import FFTBackend
        backend = FFTBackend()
        assert backend.supports_type(np.float64) is True


# ---------------------------------------------------------------------------
# PocketFFT backend direct calls
# ---------------------------------------------------------------------------

class TestPocketFFTBackendDirect:
    """Tests for PocketFFTBackend methods called directly."""

    def test_raw_fft_direct(self):
        from numpy.fft._backend import PocketFFTBackend
        backend = PocketFFTBackend()
        assert backend.name == "pocketfft"
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = backend._raw_fft(data, 4, 0, False, True, None)
        expected = np.array([10+0j, -2+2j, -2+0j, -2-2j])
        np.testing.assert_allclose(result, expected)

    def test_raw_fftnd_direct(self):
        from numpy.fft._backend import PocketFFTBackend
        from numpy.fft._pocketfft import fft as _fft
        backend = PocketFFTBackend()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend._raw_fftnd(data, function=_fft)
        expected = np.array([[10+0j, -2+0j], [-4+0j, 0+0j]])
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Context manager with pocketfft (no KML required)
# ---------------------------------------------------------------------------

class TestBackendContextPocketFFT:
    """Tests for _BackendContext using pocketfft backend."""

    def test_set_backend_pocketfft_context(self):
        assert np.fft.get_backend() == "pocketfft"
        with np.fft.set_backend("pocketfft"):
            assert np.fft.get_backend() == "pocketfft"
            data = np.array([1.0, 2.0, 3.0, 4.0])
            result = np.fft.fft(data)
            expected = np.array([10+0j, -2+2j, -2+0j, -2-2j])
            np.testing.assert_allclose(result, expected)
        assert np.fft.get_backend() == "pocketfft"

    def test_nested_pocketfft_context(self):
        with np.fft.set_backend("pocketfft"):
            assert np.fft.get_backend() == "pocketfft"
            with np.fft.set_backend("pocketfft"):
                assert np.fft.get_backend() == "pocketfft"
            assert np.fft.get_backend() == "pocketfft"

    def test_context_exit_without_previous(self):
        tl = _BACKEND_MANAGER._thread_local
        if hasattr(tl, "backend"):
            del tl.backend
        with np.fft.set_backend("pocketfft"):
            assert hasattr(tl, "backend")
        assert not hasattr(tl, "backend") or tl.backend is None

    def test_context_exit_with_previous(self):
        with np.fft.set_backend("pocketfft"):
            with np.fft.set_backend("pocketfft"):
                assert np.fft.get_backend() == "pocketfft"
            assert np.fft.get_backend() == "pocketfft"


# ---------------------------------------------------------------------------
# set_global_backend / reset_backend with pocketfft
# ---------------------------------------------------------------------------

class TestGlobalBackendPocketFFT:
    """Tests for set_global_backend and reset_backend using pocketfft."""

    def test_set_global_pocketfft(self):
        np.fft.set_global_backend("pocketfft")
        assert np.fft.get_backend() == "pocketfft"
        np.fft.reset_backend()
        assert np.fft.get_backend() == "pocketfft"

    def test_reset_clears_global(self):
        np.fft.set_global_backend("pocketfft")
        assert _BACKEND_MANAGER._global_backend is not None
        np.fft.reset_backend()
        assert _BACKEND_MANAGER._global_backend is None

    def test_global_backend_fft_computation(self):
        np.fft.set_global_backend("pocketfft")
        try:
            data = np.array([1.0, 2.0, 3.0, 4.0])
            result = np.fft.fft(data)
            expected = np.array([10+0j, -2+2j, -2+0j, -2-2j])
            np.testing.assert_allclose(result, expected)
        finally:
            np.fft.reset_backend()


# ---------------------------------------------------------------------------
# get_backend_for_type fallback
# ---------------------------------------------------------------------------

class TestGetBackendForType:
    """Tests for _BackendManager.get_backend_for_type."""

    def test_fallback_for_unsupported_dtype(self):
        class _FakeBackend:
            name = "fake"
            def supports_type(self, dtype):
                return False
        _BACKEND_MANAGER._thread_local.backend = _FakeBackend()
        try:
            backend = _BACKEND_MANAGER.get_backend_for_type(np.float64)
            assert backend.name == "pocketfft"
        finally:
            del _BACKEND_MANAGER._thread_local.backend


# ---------------------------------------------------------------------------
# Env var backend resolution
# ---------------------------------------------------------------------------

class TestEnvVarBackendResolution:
    """Tests for env var backend resolution in get_current_backend."""

    def test_env_backend_takes_effect(self):
        _simulate_env_var("pocketfft")
        assert _BACKEND_MANAGER._env_backend is not None
        assert np.fft.get_backend() == "pocketfft"

    def test_env_backend_invalid_warns(self):
        with pytest.warns(RuntimeWarning, match="NUMPY_FFT_BACKEND"):
            _simulate_env_var("nonexistent_backend")
        assert _BACKEND_MANAGER._env_backend is None

    def test_env_var_valid_backend_in_init(self):
        import os
        from numpy.fft._backend import _BackendManager
        old = os.environ.get("NUMPY_FFT_BACKEND")
        try:
            os.environ["NUMPY_FFT_BACKEND"] = "pocketfft"
            mgr = _BackendManager()
            assert mgr._env_backend is not None
            assert mgr.get_backend_name() == "pocketfft"
        finally:
            if old is None:
                os.environ.pop("NUMPY_FFT_BACKEND", None)
            else:
                os.environ["NUMPY_FFT_BACKEND"] = old

    def test_env_var_invalid_backend_in_init(self):
        import os
        from numpy.fft._backend import _BackendManager
        old = os.environ.get("NUMPY_FFT_BACKEND")
        try:
            os.environ["NUMPY_FFT_BACKEND"] = "nonexistent"
            with pytest.warns(RuntimeWarning, match="NUMPY_FFT_BACKEND"):
                mgr = _BackendManager()
            assert mgr._env_backend is None
            assert mgr.get_backend_name() == "pocketfft"
        finally:
            if old is None:
                os.environ.pop("NUMPY_FFT_BACKEND", None)
            else:
                os.environ["NUMPY_FFT_BACKEND"] = old
