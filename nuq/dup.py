import numpy as np
import hnswlib
from scipy.special import logsumexp


class Kernel:
    KERNEL_TYPES = ['RBF']

    def __init__(self, kernel_type='RBF', **params):
        if 'bandwidth' in params:
            bandwidth = params['bandwidth']
        else:
            bandwidth = 1
        self.bandwidth = bandwidth

        if kernel_type == 'RBF':
            self._function = self._log_rbf
        else:
            raise ValueError(
                f'Unknown kernel type, should be one of the following: {self.KERNEL_TYPES}'
            )

    def __call__(self, x1, x2=None):
        return self._function(x1, x2)

    def _log_rbf(self, distance, _):
        return -distance/self.bandwidth**2/2


class NuqClassifierDup:
    def __init__(self, kernel_type='RBF', n_neighbors=20):
        self.index = None
        self.labels = None
        self.n_classes = None
        self.bandwidth = 1
        self.dims = None

        self.kernel = Kernel(kernel_type, bandwidth=1)
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        self.dims = x.shape[1]
        self.index = hnswlib.Index(space='l2', dim=self.dims)
        self.index.init_index(max_elements=len(x), M=16, ef_construction=200)
        self.index.add_items(x)
        self.labels = y
        self.n_classes = np.max(y) + 1

    def predict(self, x, uncertainty_type=None):
        log_probabilities, uncertainties = self.predict_log_proba(x, uncertainty_type)
        predictions = np.argmax(log_probabilities, axis=-1)
        return predictions, uncertainties

    def predict_log_proba(self, x, uncertainty_type=None):
        if self.index is None:
            raise RuntimeError('You should fit the model first')
        uncertainty = None

        indices, distance_square = self.index.knn_query(x, self.n_neighbors)
        neighbor_labels = self.labels[indices]

        similarities = self.kernel(distance_square)
        log_normalization = logsumexp(similarities, axis=1)
        log_probabilities = np.empty((len(x), self.n_classes))

        for c in range(self.n_classes):
            log_probabilities[:, c] = logsumexp(np.where(neighbor_labels==c, similarities, -np.inf), axis=1) - log_normalization


        probabilities = np.exp(log_probabilities)
        if uncertainty_type == 'aleatoric':
            uncertainty = 1 - np.exp(np.max(log_probabilities, axis=-1))
        elif uncertainty_type == 'epistemic':
            sigma_square = probabilities * (1-probabilities)
            # Add 1e-99 to prevent log(0)
            sigma_square = np.max(sigma_square, axis=-1) + 1e-99
            # return log_probabilities, sigma_square

            h = self.kernel.bandwidth
            N = self.index.get_current_count()
            d = self.dims
            log_probability_density = log_normalization - np.log(N) - h*np.log(d)
            # return log_probabilities, log_probability_density
            log_kernel_volume = h*np.log(d) - np.log(2*np.sqrt(np.pi))
            log_tau = np.log(sigma_square) - log_probability_density - np.log(N) + log_kernel_volume
            uncertainty = np.log(2*np.sqrt(2) / np.sqrt(np.pi)) + log_tau

        return log_probabilities, uncertainty
    #
    def _update_bandwidth(self, new_bandwidth):
        self.kernel.bandwidth = new_bandwidth
