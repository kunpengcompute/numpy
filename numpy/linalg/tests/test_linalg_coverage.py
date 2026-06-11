"""
Additional tests to improve C/C++ coverage in numpy/linalg/umath_linalg.cpp
Focus on complex types, non-contiguous arrays, and edge cases.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


class TestComplexLinalg:
    """Test linalg operations with complex types to cover complex template instantiations."""

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_solve_complex(self, dtype):
        """Test solving linear systems with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j]], dtype=dtype)
        b = np.array([1+1j, 2-1j], dtype=dtype)
        x = np.linalg.solve(A, b)
        assert_allclose(A @ x, b, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_solve_complex_stacked(self, dtype):
        """Test solving stacked complex linear systems."""
        np.random.seed(42)
        n = 4
        A = np.random.randn(3, n, n).astype(dtype) + 1j * np.random.randn(3, n, n).astype(dtype)
        b = np.random.randn(3, n, 1).astype(dtype) + 1j * np.random.randn(3, n, 1).astype(dtype)
        x = np.linalg.solve(A, b)
        assert x.shape == (3, n, 1)
        for i in range(3):
            assert_allclose(A[i] @ x[i], b[i], rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_inv_complex(self, dtype):
        """Test matrix inversion with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j]], dtype=dtype)
        A_inv = np.linalg.inv(A)
        assert_allclose(A @ A_inv, np.eye(2), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_inv_complex_noncontiguous(self, dtype):
        """Test matrix inversion with non-contiguous complex arrays."""
        A = np.random.randn(5, 5).astype(dtype) + 1j * np.random.randn(5, 5).astype(dtype)
        A_noncontig = A[::2, ::2]  # Non-contiguous view
        A_copy = A_noncontig.copy()
        A_inv = np.linalg.inv(A_copy)
        assert_allclose(A_copy @ A_inv, np.eye(A_copy.shape[0]), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_det_complex(self, dtype):
        """Test determinant with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j]], dtype=dtype)
        det = np.linalg.det(A)
        expected = (1+2j) * (4+1j) - (2-1j) * (3+0j)
        assert_allclose(det, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_slogdet_complex(self, dtype):
        """Test slogdet with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j]], dtype=dtype)
        sign, logdet = np.linalg.slogdet(A)
        det = np.linalg.det(A)
        assert_allclose(sign * np.exp(logdet), det, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_eig_complex(self, dtype):
        """Test eigenvalue decomposition with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j]], dtype=dtype)
        w, v = np.linalg.eig(A)
        for i in range(len(w)):
            assert_allclose(A @ v[:, i], w[i] * v[:, i], rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_eigvals_complex(self, dtype):
        """Test eigenvalues with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j]], dtype=dtype)
        w = np.linalg.eigvals(A)
        assert w.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_svd_complex(self, dtype):
        """Test SVD with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j]], dtype=dtype)
        U, s, Vh = np.linalg.svd(A)
        assert_allclose(U @ np.diag(s) @ Vh, A, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_svd_complex_full_matrices(self, dtype):
        """Test SVD with complex matrices and full_matrices=True."""
        A = np.random.randn(3, 5).astype(dtype) + 1j * np.random.randn(3, 5).astype(dtype)
        U, s, Vh = np.linalg.svd(A, full_matrices=True)
        assert U.shape == (3, 3)
        assert Vh.shape == (5, 5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_qr_complex(self, dtype):
        """Test QR decomposition with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j]], dtype=dtype)
        Q, R = np.linalg.qr(A)
        assert_allclose(Q @ R, A, rtol=1e-5, atol=1e-5)
        assert_allclose(Q.conj().T @ Q, np.eye(2), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_cholesky_complex(self, dtype):
        """Test Cholesky decomposition with complex Hermitian positive-definite matrices."""
        A = np.array([[4+0j, 2+1j], [2-1j, 5+0j]], dtype=dtype)
        L = np.linalg.cholesky(A)
        assert_allclose(L @ L.conj().T, A, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_lstsq_complex(self, dtype):
        """Test least squares with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j], [5+1j, 6-2j]], dtype=dtype)
        b = np.array([1+1j, 2-1j, 3+0j], dtype=dtype)
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        assert x.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_matrix_rank_complex(self, dtype):
        """Test matrix rank with complex matrices."""
        A = np.array([[1+2j, 2-1j], [2+4j, 4-2j]], dtype=dtype)  # Rank 1
        rank = np.linalg.matrix_rank(A)
        assert rank == 1

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_norm_complex(self, dtype):
        """Test various norms with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j]], dtype=dtype)
        
        # Frobenius norm
        norm_fro = np.linalg.norm(A, 'fro')
        assert norm_fro > 0
        
        # Nuclear norm
        norm_nuc = np.linalg.norm(A, 'nuc')
        assert norm_nuc > 0
        
        # 2-norm
        norm_2 = np.linalg.norm(A, 2)
        assert norm_2 > 0

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_cond_complex(self, dtype):
        """Test condition number with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j]], dtype=dtype)
        cond = np.linalg.cond(A)
        assert cond > 0

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_pinv_complex(self, dtype):
        """Test pseudo-inverse with complex matrices."""
        A = np.array([[1+2j, 2-1j], [3+0j, 4+1j]], dtype=dtype)
        A_pinv = np.linalg.pinv(A)
        assert_allclose(A @ A_pinv @ A, A, rtol=1e-5)


class TestNonContiguousArrays:
    """Test linalg operations with non-contiguous arrays to cover linearize/delinearize paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_solve_noncontiguous(self, dtype):
        """Test solving linear systems with non-contiguous arrays."""
        A_full = np.random.randn(6, 6).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A_full = A_full + 1j * np.random.randn(6, 6).astype(dtype)
        
        A = A_full[::2, ::2]  # Non-contiguous
        b = np.random.randn(3).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            b = b + 1j * np.random.randn(3).astype(dtype)
        
        x = np.linalg.solve(A.copy(), b)
        assert_allclose(A @ x, b, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_inv_noncontiguous(self, dtype):
        """Test matrix inversion with non-contiguous arrays."""
        A_full = np.random.randn(6, 6).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A_full = A_full + 1j * np.random.randn(6, 6).astype(dtype)
        
        A = A_full[::2, ::2]  # Non-contiguous
        A_copy = A.copy()
        A_inv = np.linalg.inv(A_copy)
        assert_allclose(A_copy @ A_inv, np.eye(3), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_svd_noncontiguous(self, dtype):
        """Test SVD with non-contiguous arrays."""
        A_full = np.random.randn(6, 8).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A_full = A_full + 1j * np.random.randn(6, 8).astype(dtype)
        
        A = A_full[::2, ::2]  # Non-contiguous
        U, s, Vh = np.linalg.svd(A.copy())
        assert U.shape[0] == 3
        assert Vh.shape[1] == 4

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_eig_noncontiguous(self, dtype):
        """Test eigenvalue decomposition with non-contiguous arrays."""
        A_full = np.random.randn(6, 6).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A_full = A_full + 1j * np.random.randn(6, 6).astype(dtype)
        
        A = A_full[::2, ::2]  # Non-contiguous
        w, v = np.linalg.eig(A.copy())
        assert len(w) == 3

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_qr_noncontiguous(self, dtype):
        """Test QR decomposition with non-contiguous arrays."""
        A_full = np.random.randn(6, 8).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A_full = A_full + 1j * np.random.randn(6, 8).astype(dtype)
        
        A = A_full[::2, ::2]  # Non-contiguous
        Q, R = np.linalg.qr(A.copy())
        assert Q.shape == (3, 3)
        assert R.shape == (3, 4)


class TestEdgeCases:
    """Test edge cases to cover error handling and special paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_solve_singular(self, dtype):
        """Test solving singular linear systems."""
        A = np.array([[1, 2], [2, 4]], dtype=dtype)  # Singular matrix
        b = np.array([1, 2], dtype=dtype)
        
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.solve(A, b)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_inv_singular(self, dtype):
        """Test inverting singular matrices."""
        A = np.array([[1, 2], [2, 4]], dtype=dtype)  # Singular matrix
        
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.inv(A)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_cholesky_not_positive_definite(self, dtype):
        """Test Cholesky with non-positive-definite matrices."""
        A = np.array([[1, 2], [2, 1]], dtype=dtype)  # Not positive definite
        
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(A)

    def test_solve_dimension_mismatch(self):
        """Test solving with mismatched dimensions."""
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b = np.array([1, 2, 3], dtype=np.float64)
        
        with pytest.raises(ValueError):
            np.linalg.solve(A, b)

    def test_inv_non_square(self):
        """Test inverting non-square matrices."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.inv(A)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_det_zero_size(self, dtype):
        """Test determinant with zero-size matrices."""
        A = np.zeros((0, 0), dtype=dtype)
        det = np.linalg.det(A)
        assert det == 1.0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_slogdet_zero_size(self, dtype):
        """Test slogdet with zero-size matrices."""
        A = np.zeros((0, 0), dtype=dtype)
        sign, logdet = np.linalg.slogdet(A)
        assert sign == 1.0
        assert logdet == 0.0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_svd_zero_size(self, dtype):
        """Test SVD with zero-size matrices."""
        A = np.zeros((0, 0), dtype=dtype)
        U, s, Vh = np.linalg.svd(A)
        assert U.shape == (0, 0)
        assert len(s) == 0
        assert Vh.shape == (0, 0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_eig_zero_size(self, dtype):
        """Test eigenvalue decomposition with zero-size matrices."""
        A = np.zeros((0, 0), dtype=dtype)
        w, v = np.linalg.eig(A)
        assert len(w) == 0
        assert v.shape == (0, 0)


class TestStridedArrays:
    """Test linalg operations with strided arrays to cover stride handling code."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_solve_fortran_order(self, dtype):
        """Test solving with Fortran-ordered arrays."""
        A = np.asfortranarray(np.random.randn(4, 4).astype(dtype))
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.asfortranarray(np.random.randn(4, 4).astype(dtype))
        
        b = np.random.randn(4).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            b = b + 1j * np.random.randn(4).astype(dtype)
        
        x = np.linalg.solve(A, b)
        assert_allclose(A @ x, b, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_inv_fortran_order(self, dtype):
        """Test inversion with Fortran-ordered arrays."""
        A = np.asfortranarray(np.random.randn(4, 4).astype(dtype))
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.asfortranarray(np.random.randn(4, 4).astype(dtype))
        
        A_inv = np.linalg.inv(A)
        assert_allclose(A @ A_inv, np.eye(4), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_svd_fortran_order(self, dtype):
        """Test SVD with Fortran-ordered arrays."""
        A = np.asfortranarray(np.random.randn(4, 5).astype(dtype))
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.asfortranarray(np.random.randn(4, 5).astype(dtype))
        
        U, s, Vh = np.linalg.svd(A)
        assert U.shape == (4, 4)
        assert len(s) == 4
        assert Vh.shape == (5, 5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_eig_fortran_order(self, dtype):
        """Test eigenvalue decomposition with Fortran-ordered arrays."""
        A = np.asfortranarray(np.random.randn(4, 4).astype(dtype))
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.asfortranarray(np.random.randn(4, 4).astype(dtype))
        
        w, v = np.linalg.eig(A)
        assert len(w) == 4
        assert v.shape == (4, 4)


class TestLargeMatrices:
    """Test linalg operations with larger matrices to cover more code paths."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_solve_large(self, dtype):
        """Test solving large linear systems."""
        n = 50
        A = np.random.randn(n, n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.random.randn(n, n).astype(dtype)
        
        b = np.random.randn(n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            b = b + 1j * np.random.randn(n).astype(dtype)
        
        x = np.linalg.solve(A, b)
        assert_allclose(A @ x, b, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_inv_large(self, dtype):
        """Test inversion of large matrices."""
        n = 50
        A = np.random.randn(n, n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.random.randn(n, n).astype(dtype)
        
        A_inv = np.linalg.inv(A)
        assert_allclose(A @ A_inv, np.eye(n), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_svd_large(self, dtype):
        """Test SVD of large matrices."""
        m, n = 30, 40
        A = np.random.randn(m, n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.random.randn(m, n).astype(dtype)
        
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        assert U.shape == (m, min(m, n))
        assert len(s) == min(m, n)
        assert Vh.shape == (min(m, n), n)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_eig_large(self, dtype):
        """Test eigenvalue decomposition of large matrices."""
        n = 30
        A = np.random.randn(n, n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.random.randn(n, n).astype(dtype)
        
        w, v = np.linalg.eig(A)
        assert len(w) == n
        assert v.shape == (n, n)


class TestRectangularMatrices:
    """Test linalg operations with rectangular matrices."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_lstsq_overdetermined(self, dtype):
        """Test least squares with overdetermined systems."""
        m, n = 10, 5
        A = np.random.randn(m, n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.random.randn(m, n).astype(dtype)
        
        b = np.random.randn(m).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            b = b + 1j * np.random.randn(m).astype(dtype)
        
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        assert x.shape == (n,)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_lstsq_underdetermined(self, dtype):
        """Test least squares with underdetermined systems."""
        m, n = 5, 10
        A = np.random.randn(m, n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.random.randn(m, n).astype(dtype)
        
        b = np.random.randn(m).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            b = b + 1j * np.random.randn(m).astype(dtype)
        
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        assert x.shape == (n,)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_qr_rectangular(self, dtype):
        """Test QR decomposition with rectangular matrices."""
        m, n = 6, 4
        A = np.random.randn(m, n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.random.randn(m, n).astype(dtype)
        
        Q, R = np.linalg.qr(A)
        assert Q.shape == (m, n)
        assert R.shape == (n, n)
        assert_allclose(Q @ R, A, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_qr_wide(self, dtype):
        """Test QR decomposition with wide matrices."""
        m, n = 4, 6
        A = np.random.randn(m, n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.random.randn(m, n).astype(dtype)
        
        Q, R = np.linalg.qr(A)
        assert Q.shape == (m, m)
        assert R.shape == (m, n)
        assert_allclose(Q @ R, A, rtol=1e-5)


class TestMultipleRightHandSides:
    """Test linalg operations with multiple right-hand sides."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_solve_multiple_rhs(self, dtype):
        """Test solving with multiple right-hand sides."""
        n = 5
        k = 3
        A = np.random.randn(n, n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.random.randn(n, n).astype(dtype)
        
        B = np.random.randn(n, k).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            B = B + 1j * np.random.randn(n, k).astype(dtype)
        
        X = np.linalg.solve(A, B)
        assert_allclose(A @ X, B, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_lstsq_multiple_rhs(self, dtype):
        """Test least squares with multiple right-hand sides."""
        m, n, k = 10, 5, 3
        A = np.random.randn(m, n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * np.random.randn(m, n).astype(dtype)
        
        B = np.random.randn(m, k).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            B = B + 1j * np.random.randn(m, k).astype(dtype)
        
        X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        assert X.shape == (n, k)


class TestEighEigvalsh:
    """Test eigh/eigvalsh to cover eigh_lo, eigh_up, eigvalsh_lo, eigvalsh_up gufuncs."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_eigh_lower(self, dtype):
        A = np.array([[2, 1], [1, 3]], dtype=dtype)
        w, v = np.linalg.eigh(A, UPLO='L')
        assert w.shape == (2,)
        assert v.shape == (2, 2)
        assert_allclose(A @ v, v * w, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_eigh_upper(self, dtype):
        A = np.array([[2, 1], [1, 3]], dtype=dtype)
        w, v = np.linalg.eigh(A, UPLO='U')
        assert w.shape == (2,)
        assert v.shape == (2, 2)
        assert_allclose(A @ v, v * w, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_eigvalsh_lower(self, dtype):
        A = np.array([[2, 1], [1, 3]], dtype=dtype)
        w = np.linalg.eigvalsh(A, UPLO='L')
        assert w.shape == (2,)
        assert np.all(np.diff(w) >= 0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_eigvalsh_upper(self, dtype):
        A = np.array([[2, 1], [1, 3]], dtype=dtype)
        w = np.linalg.eigvalsh(A, UPLO='U')
        assert w.shape == (2,)
        assert np.all(np.diff(w) >= 0)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_eigh_complex_lower(self, dtype):
        A = np.array([[2, 1+1j], [1-1j, 3]], dtype=dtype)
        w, v = np.linalg.eigh(A, UPLO='L')
        assert w.shape == (2,)
        assert_allclose(A @ v, v * w, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_eigh_complex_upper(self, dtype):
        A = np.array([[2, 1+1j], [1-1j, 3]], dtype=dtype)
        w, v = np.linalg.eigh(A, UPLO='U')
        assert w.shape == (2,)
        assert_allclose(A @ v, v * w, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_eigvalsh_complex_lower(self, dtype):
        A = np.array([[2, 1+1j], [1-1j, 3]], dtype=dtype)
        w = np.linalg.eigvalsh(A, UPLO='L')
        assert w.shape == (2,)
        assert np.all(np.diff(w) >= -1e-10)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_eigvalsh_complex_upper(self, dtype):
        A = np.array([[2, 1+1j], [1-1j, 3]], dtype=dtype)
        w = np.linalg.eigvalsh(A, UPLO='U')
        assert w.shape == (2,)
        assert np.all(np.diff(w) >= -1e-10)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_eigh_large(self, dtype):
        n = 10
        rng = np.random.default_rng(42)
        A = rng.random((n, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((n, n)).astype(dtype)
        A = A + A.conj().T
        w, v = np.linalg.eigh(A)
        assert_allclose(A @ v, v * w, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_eigh_stacked(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((3, 4, 4)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((3, 4, 4)).astype(dtype)
        A = A + np.swapaxes(A.conj(), -2, -1)
        w, v = np.linalg.eigh(A)
        assert w.shape == (3, 4)
        assert v.shape == (3, 4, 4)


class TestSVDVariants:
    """Test SVD variants to cover svd, svd_s, svd_f gufuncs."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_svd_full_matrices_true(self, dtype):
        m, n = 6, 4
        rng = np.random.default_rng(42)
        A = rng.random((m, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((m, n)).astype(dtype)
        U, s, Vh = np.linalg.svd(A, full_matrices=True)
        assert U.shape == (m, m)
        assert s.shape == (min(m, n),)
        assert Vh.shape == (n, n)
        reconstructed = U[:, :min(m,n)] @ np.diag(s) @ Vh[:min(m,n), :]
        assert_allclose(reconstructed, A, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_svd_full_matrices_false(self, dtype):
        m, n = 6, 4
        rng = np.random.default_rng(42)
        A = rng.random((m, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((m, n)).astype(dtype)
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        assert U.shape == (m, min(m, n))
        assert s.shape == (min(m, n),)
        assert Vh.shape == (min(m, n), n)
        reconstructed = U @ np.diag(s) @ Vh
        assert_allclose(reconstructed, A, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_svd_compute_uv_false(self, dtype):
        m, n = 6, 4
        rng = np.random.default_rng(42)
        A = rng.random((m, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((m, n)).astype(dtype)
        s = np.linalg.svd(A, compute_uv=False)
        assert s.shape == (min(m, n),)
        assert np.all(s >= 0)
        assert np.all(np.diff(s) <= 1e-10)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_svd_wide_matrix(self, dtype):
        m, n = 4, 6
        rng = np.random.default_rng(42)
        A = rng.random((m, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((m, n)).astype(dtype)
        U, s, Vh = np.linalg.svd(A, full_matrices=True)
        assert U.shape == (m, m)
        assert Vh.shape == (n, n)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_svd_stacked(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((3, 4, 5)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((3, 4, 5)).astype(dtype)
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        assert U.shape == (3, 4, 4)
        assert s.shape == (3, 4)
        assert Vh.shape == (3, 4, 5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_svdvals(self, dtype):
        m, n = 6, 4
        rng = np.random.default_rng(42)
        A = rng.random((m, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((m, n)).astype(dtype)
        s = np.linalg.svdvals(A)
        assert s.shape == (min(m, n),)


class TestCholeskyVariants:
    """Test Cholesky variants to cover cholesky_lo and cholesky_up gufuncs."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_cholesky_lower(self, dtype):
        A = np.array([[4, 2], [2, 5]], dtype=dtype)
        L = np.linalg.cholesky(A)
        assert_allclose(L @ L.conj().T, A, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_cholesky_upper(self, dtype):
        A = np.array([[4, 2], [2, 5]], dtype=dtype)
        U = np.linalg.cholesky(A, upper=True)
        assert_allclose(U.conj().T @ U, A, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_cholesky_stacked(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((3, 4, 4)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((3, 4, 4)).astype(dtype)
        A = A @ np.swapaxes(A.conj(), -2, -1) + np.eye(4) * 0.1
        L = np.linalg.cholesky(A)
        assert L.shape == (3, 4, 4)
        for i in range(3):
            assert_allclose(L[i] @ L[i].conj().T, A[i], rtol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_cholesky_large(self, dtype):
        n = 20
        rng = np.random.default_rng(42)
        A = rng.random((n, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((n, n)).astype(dtype)
        A = A @ A.conj().T + np.eye(n, dtype=dtype)
        L = np.linalg.cholesky(A)
        assert_allclose(L @ L.conj().T, A, rtol=1e-4, atol=1e-4)


class TestLstsqVariants:
    """Test lstsq variants to cover rank-deficient and edge cases."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_lstsq_rank_deficient(self, dtype):
        m, n = 10, 5
        rng = np.random.default_rng(42)
        A = rng.random((m, 2)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((m, 2)).astype(dtype)
        A_full = np.hstack([A, A, A[:, :1]])
        b = rng.random(m).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            b = b + 1j * rng.random(m).astype(dtype)
        x, residuals, rank, s = np.linalg.lstsq(A_full, b, rcond=None)
        assert x.shape == (n,)
        assert rank < n

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_lstsq_exact_fit(self, dtype):
        n = 5
        rng = np.random.default_rng(42)
        A = rng.random((n, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((n, n)).astype(dtype)
        x_true = rng.random(n).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            x_true = x_true + 1j * rng.random(n).astype(dtype)
        b = A @ x_true
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        assert_allclose(x, x_true, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_lstsq_with_rcond(self, dtype):
        m, n = 10, 5
        rng = np.random.default_rng(42)
        A = rng.random((m, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((m, n)).astype(dtype)
        b = rng.random(m).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            b = b + 1j * rng.random(m).astype(dtype)
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=1e-10)
        assert x.shape == (n,)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_lstsq_zero_b(self, dtype):
        m, n = 5, 3
        rng = np.random.default_rng(42)
        A = rng.random((m, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((m, n)).astype(dtype)
        b = np.zeros(m, dtype=dtype)
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        assert_allclose(x, np.zeros(n), atol=1e-10)


class TestQRVariants:
    """Test QR variants to cover qr_r_raw, qr_reduced, qr_complete gufuncs."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_qr_reduced_mode(self, dtype):
        m, n = 6, 4
        rng = np.random.default_rng(42)
        A = rng.random((m, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((m, n)).astype(dtype)
        Q, R = np.linalg.qr(A, mode='reduced')
        assert Q.shape == (m, n)
        assert R.shape == (n, n)
        assert_allclose(Q @ R, A, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_qr_complete_mode(self, dtype):
        m, n = 6, 4
        rng = np.random.default_rng(42)
        A = rng.random((m, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((m, n)).astype(dtype)
        Q, R = np.linalg.qr(A, mode='complete')
        assert Q.shape == (m, m)
        assert R.shape == (m, n)
        assert_allclose(Q @ R, A, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_qr_r_mode(self, dtype):
        m, n = 6, 4
        rng = np.random.default_rng(42)
        A = rng.random((m, n)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((m, n)).astype(dtype)
        R = np.linalg.qr(A, mode='r')
        assert R.shape == (n, n)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_qr_stacked(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((3, 6, 4)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((3, 6, 4)).astype(dtype)
        Q, R = np.linalg.qr(A)
        assert Q.shape == (3, 6, 4)
        assert R.shape == (3, 4, 4)
        for i in range(3):
            assert_allclose(Q[i] @ R[i], A[i], rtol=1e-5)


class TestSlogdetDet:
    """Test slogdet and det to cover slogdet and det gufuncs."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_slogdet(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((5, 5)).astype(dtype)
        sign, logdet = np.linalg.slogdet(A)
        assert sign.shape == ()
        assert logdet.shape == ()

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_det(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((5, 5)).astype(dtype)
        d = np.linalg.det(A)
        assert d.shape == ()

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_slogdet_stacked(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((3, 4, 4)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((3, 4, 4)).astype(dtype)
        sign, logdet = np.linalg.slogdet(A)
        assert sign.shape == (3,)
        assert logdet.shape == (3,)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_det_stacked(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((3, 4, 4)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((3, 4, 4)).astype(dtype)
        d = np.linalg.det(A)
        assert d.shape == (3,)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_det_identity(self, dtype):
        A = np.eye(5, dtype=dtype)
        assert_allclose(np.linalg.det(A), 1.0, rtol=1e-10)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_slogdet_identity(self, dtype):
        A = np.eye(5, dtype=dtype)
        sign, logdet = np.linalg.slogdet(A)
        assert_allclose(sign, 1.0)
        assert_allclose(logdet, 0.0, atol=1e-10)


class TestNormComprehensive:
    """Test norm with various ord values."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_norm_vector_1(self, dtype):
        a = np.array([1, -2, 3, -4], dtype=dtype)
        assert_allclose(np.linalg.norm(a, ord=1), 10.0, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_norm_vector_2(self, dtype):
        a = np.array([3, 4], dtype=dtype)
        assert_allclose(np.linalg.norm(a, ord=2), 5.0, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_norm_vector_inf(self, dtype):
        a = np.array([1, -2, 3, -4], dtype=dtype)
        assert_allclose(np.linalg.norm(a, ord=np.inf), 4.0, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_norm_vector_neg_inf(self, dtype):
        a = np.array([1, -2, 3, -4], dtype=dtype)
        assert_allclose(np.linalg.norm(a, ord=-np.inf), 1.0, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_norm_matrix_fro(self, dtype):
        A = np.array([[1, 2], [3, 4]], dtype=dtype)
        expected = np.sqrt(1 + 4 + 9 + 16)
        assert_allclose(np.linalg.norm(A, ord='fro'), expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_norm_matrix_1(self, dtype):
        A = np.array([[1, 2], [3, 4]], dtype=dtype)
        assert_allclose(np.linalg.norm(A, ord=1), 6.0, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_norm_matrix_inf(self, dtype):
        A = np.array([[1, 2], [3, 4]], dtype=dtype)
        assert_allclose(np.linalg.norm(A, ord=np.inf), 7.0, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_norm_matrix_nuc(self, dtype):
        A = np.array([[1, 2], [3, 4]], dtype=dtype)
        s = np.linalg.svd(A, compute_uv=False)
        assert_allclose(np.linalg.norm(A, ord='nuc'), np.sum(s), rtol=1e-5)

    def test_norm_keepdims(self):
        a = np.arange(12).reshape(3, 4).astype(np.float64)
        res = np.linalg.norm(a, axis=1, keepdims=True)
        assert res.shape == (3, 1)

    def test_norm_stacked(self):
        A = np.random.randn(3, 4, 5)
        res = np.linalg.norm(A, axis=-1)
        assert res.shape == (3, 4)


class TestPinvComprehensive:
    """Test pinv with various configurations."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_pinv_square(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((5, 5)).astype(dtype)
        A_pinv = np.linalg.pinv(A)
        assert_allclose(A @ A_pinv @ A, A, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_pinv_rectangular_tall(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((6, 4)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((6, 4)).astype(dtype)
        A_pinv = np.linalg.pinv(A)
        assert A_pinv.shape == (4, 6)
        assert_allclose(A @ A_pinv @ A, A, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_pinv_rectangular_wide(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((4, 6)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((4, 6)).astype(dtype)
        A_pinv = np.linalg.pinv(A)
        assert A_pinv.shape == (6, 4)
        assert_allclose(A @ A_pinv @ A, A, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_pinv_hermitian(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((5, 5)).astype(dtype)
        A = A @ A.conj().T + np.eye(5, dtype=dtype)
        A_pinv = np.linalg.pinv(A, hermitian=True)
        assert_allclose(A @ A_pinv @ A, A, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_pinv_stacked(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((3, 5, 5)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((3, 5, 5)).astype(dtype)
        A_pinv = np.linalg.pinv(A)
        assert A_pinv.shape == (3, 5, 5)


class TestCondComprehensive:
    """Test cond with various ord values."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_cond_2(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((5, 5)).astype(dtype)
        c = np.linalg.cond(A, p=2)
        assert c >= 1.0 or np.isnan(c)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cond_1(self, dtype):
        A = np.array([[1, 0], [0, 2]], dtype=dtype)
        assert_allclose(np.linalg.cond(A, p=1), 2.0, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cond_inf(self, dtype):
        A = np.array([[1, 0], [0, 2]], dtype=dtype)
        assert_allclose(np.linalg.cond(A, p=np.inf), 2.0, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cond_fro(self, dtype):
        A = np.array([[1, 0], [0, 2]], dtype=dtype)
        c = np.linalg.cond(A, p='fro')
        assert c >= 1.0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cond_neg_inf(self, dtype):
        A = np.array([[1, 0], [0, 2]], dtype=dtype)
        c = np.linalg.cond(A, p=-np.inf)
        assert c >= 1.0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_cond_negative_1(self, dtype):
        A = np.array([[1, 0], [0, 2]], dtype=dtype)
        c = np.linalg.cond(A, p=-1)
        assert c >= 0


class TestMatrixPowerComprehensive:
    """Test matrix_power for various powers."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_matrix_power_positive(self, dtype):
        A = np.array([[1, 1], [0, 1]], dtype=dtype)
        res = np.linalg.matrix_power(A, 3)
        expected = np.array([[1, 3], [0, 1]], dtype=dtype)
        assert_allclose(res, expected, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_matrix_power_zero(self, dtype):
        A = np.array([[1, 2], [3, 4]], dtype=dtype)
        res = np.linalg.matrix_power(A, 0)
        assert_allclose(res, np.eye(2, dtype=dtype), rtol=1e-10)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_matrix_power_one(self, dtype):
        A = np.array([[1, 2], [3, 4]], dtype=dtype)
        res = np.linalg.matrix_power(A, 1)
        assert_allclose(res, A, rtol=1e-10)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_matrix_power_negative(self, dtype):
        A = np.array([[1, 2], [0, 1]], dtype=dtype)
        res = np.linalg.matrix_power(A, -1)
        assert_allclose(A @ res, np.eye(2, dtype=dtype), rtol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_matrix_power_large(self, dtype):
        A = np.array([[1, 1], [0, 1]], dtype=dtype)
        res = np.linalg.matrix_power(A, 10)
        expected = np.array([[1, 10], [0, 1]], dtype=dtype)
        assert_allclose(res, expected, rtol=1e-5)


class TestMultiDot:
    """Test multi_dot to cover einsum and matmul paths."""

    def test_multi_dot_two(self):
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 5)
        res = np.linalg.multi_dot([A, B])
        assert_allclose(res, A @ B, rtol=1e-10)

    def test_multi_dot_three(self):
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 5)
        C = np.random.randn(5, 6)
        res = np.linalg.multi_dot([A, B, C])
        assert_allclose(res, A @ B @ C, rtol=1e-10)

    def test_multi_dot_four(self):
        A = np.random.randn(2, 3)
        B = np.random.randn(3, 4)
        C = np.random.randn(4, 5)
        D = np.random.randn(5, 6)
        res = np.linalg.multi_dot([A, B, C, D])
        assert_allclose(res, A @ B @ C @ D, rtol=1e-10)


class TestOuterCross:
    """Test outer and cross products."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_outer(self, dtype):
        a = np.array([1, 2, 3], dtype=dtype)
        b = np.array([4, 5], dtype=dtype)
        res = np.outer(a, b)
        expected = np.array([[4, 5], [8, 10], [12, 15]], dtype=dtype)
        assert_allclose(res, expected)

    def test_cross_3d(self):
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        res = np.cross(a, b)
        assert_array_equal(res, [0, 0, 1])

    def test_cross_batch(self):
        a = np.random.randn(5, 3)
        b = np.random.randn(5, 3)
        res = np.cross(a, b)
        assert res.shape == (5, 3)


class TestTensorinvTensorsolve:
    """Test tensorinv and tensorsolve."""

    def test_tensorinv_basic(self):
        A = np.eye(24).reshape(4, 6, 6, 4)
        Ainv = np.linalg.tensorinv(A, ind=2)
        assert Ainv.shape == (6, 4, 4, 6)

    def test_tensorsolve_basic(self):
        rng = np.random.default_rng(42)
        A = rng.random((4, 6, 8, 3))
        b = rng.random((4, 6))
        x = np.linalg.tensorsolve(A, b)
        assert x.shape == (8, 3)


class TestMatrixRankComprehensive:
    """Test matrix_rank with various configurations."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_matrix_rank_full_rank(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((5, 5)).astype(dtype)
        assert np.linalg.matrix_rank(A) == 5

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_matrix_rank_rank_deficient(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((5, 2)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((5, 2)).astype(dtype)
        A_full = np.hstack([A, A, A[:, :1]])
        assert np.linalg.matrix_rank(A_full) == 2

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_matrix_rank_hermitian(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5)).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            A = A + 1j * rng.random((5, 5)).astype(dtype)
        A = A + A.conj().T
        assert np.linalg.matrix_rank(A, hermitian=True) == 5

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matrix_rank_with_tol(self, dtype):
        A = np.diag([1.0, 1e-10, 1e-15]).astype(dtype)
        assert np.linalg.matrix_rank(A, tol=1e-8) == 1

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matrix_rank_stacked(self, dtype):
        rng = np.random.default_rng(42)
        A = rng.random((3, 4, 4)).astype(dtype)
        ranks = np.linalg.matrix_rank(A)
        assert ranks.shape == (3,)
