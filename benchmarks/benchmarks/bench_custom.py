import numpy as np

# ----------------------------------------------------------------------
# 1. Dense Matrix Multiplication (BLAS)
# ----------------------------------------------------------------------

class Matmul:
    params = [
        [4000],
    ]
    param_names = ["n"]

    def setup(self, n):
        self.A = np.random.randn(n, n)
        self.B = np.random.randn(n, n)
        # 预热 BLAS 初始化
        _ = self.A[:64, :64] @ self.B[:64, :64]

    def time_matmul(self, n):
        self.A @ self.B


# ----------------------------------------------------------------------
# 2. Singular Value Decomposition (LAPACK)
# ----------------------------------------------------------------------

class SVD:
    params = [
        [4000],
    ]
    param_names = ["n"]

    def setup(self, n):
        self.A = np.random.randn(n, n)

    def time_svd(self, n):
        np.linalg.svd(self.A, full_matrices=False)


# ----------------------------------------------------------------------
# 3. FFT on large signal arrays
# ----------------------------------------------------------------------

class FFT:
    params = [
        [10_000_000],
    ]
    param_names = ["length"]

    def setup(self, length):
        t_arr = np.linspace(0.0, 1.0, length, dtype=np.float64)
        self.signal = (
            np.sin(2 * np.pi * 50 * t_arr)
            + 0.5 * np.sin(2 * np.pi * 120 * t_arr)
            + np.random.randn(length) * 0.3
        )

    def time_fft(self, length):
        np.fft.fft(self.signal)


# ----------------------------------------------------------------------
# 4. Element-wise and broadcasting operations
# ----------------------------------------------------------------------

class ElementwiseBroadcast:
    params = [
        [4000],
    ]
    param_names = ["n"]

    def setup(self, n):
        self.X = np.random.randn(n, n).astype(np.float64)

    def time_normalise_and_sigmoid(self, n):
        # Row-wise z-score normalisation (broadcasting)
        row_mean = self.X.mean(axis=1, keepdims=True)
        row_std = self.X.std(axis=1, keepdims=True) + 1e-8
        Z = (self.X - row_mean) / row_std

        # Sigmoid activation
        _ = 1.0 / (1.0 + np.exp(-Z))


# ----------------------------------------------------------------------
# 5. Random number generation + statistical reduction
# ----------------------------------------------------------------------

class RandomAndStats:
    params = [
        [4000],
    ]
    param_names = ["n"]

    def setup(self, n):
        self.total = n * n
        self.data = np.random.standard_normal(self.total)

    def time_rng(self, n):
        np.random.standard_normal(self.total)

    def time_statistics(self, n):
        mean_val = np.mean(self.data)
        std_val = np.std(self.data)
        percentiles = np.percentile(self.data, [25, 50, 75, 95, 99])
        hist_counts, _ = np.histogram(self.data, bins=100)


# ----------------------------------------------------------------------
# 6. Sorting and searching on large arrays
# ----------------------------------------------------------------------

class SortSearch:
    params = [
        [4000],
    ]
    param_names = ["n"]

    def setup(self, n):
        total = n * n
        self.data = np.random.randn(total)
        self.queries = np.random.randn(min(total, 1_000_000))
        self.sorted_arr = np.sort(self.data)

    def time_sort(self, n):
        np.sort(self.data)

    def time_argsort(self, n):
        np.argsort(self.data)

    def time_searchsorted(self, n):
        np.searchsorted(self.sorted_arr, self.queries)