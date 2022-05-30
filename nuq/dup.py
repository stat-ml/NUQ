# level 3
# epistemic


from statistics import mode
import numpy as np
import hnswlib
from sklearn.metrics.pairwise import rbf_kernel


class Kernel:
    KERNEL_TYPES = ['RBF']

    def __init__(self, kernel_type='RBF', **params):
        if 'bandwidth' in params:
            bandwidth = params['bandwidth']
        else:
            bandwidth = 1
        self.bandwidth = bandwidth

        if kernel_type == 'RBF':
            self._function = self._rbf
        else:
            raise ValueError(
                f'Unknown kernel type, should be one of the following: {self.KERNEL_TYPES}'
            )

    def __call__(self, x1, x2=None):
        return self._function(x1, x2)

    def _rbf(self, distance, _):
        return np.exp(-distance/self.bandwidth**2)


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
        probabilities, uncertainties = self.predict_proba(x, uncertainty_type)
        predictions = np.argmax(probabilities, axis=-1)
        return predictions, uncertainties

    def predict_proba(self, x, uncertainty_type=None):
        if self.index is None:
            raise RuntimeError('You should fit the model first')
        uncertainty = None

        indices, distance_square = self.index.knn_query(x, self.n_neighbors)
        neighbor_labels = self.labels[indices]

        similarities = self.kernel(distance_square)
        normalization = np.sum(similarities, axis=1)

        probabilities = np.ones((len(x), self.n_classes))
        for c in range(self.n_classes):
            probabilities[:, c] = np.sum(similarities, axis=1, where=(neighbor_labels == c)) / normalization

        if uncertainty_type == 'aleatoric':
            uncertainty = 1 - np.max(probabilities, axis=-1)
        elif uncertainty_type == 'epistemic':
            sigma_square = probabilities * (1-probabilities)
            sigma_square = np.max(sigma_square, axis=-1)

            h = self.kernel.bandwidth
            N = self.index.get_current_count()
            probability_density = normalization / N / h**self.dims
            kernel_volume = h**self.dims / 2 / np.sqrt(np.pi)
            tau_square = (sigma_square / probability_density / N) * kernel_volume

            tau = np.sqrt(tau_square)
            uncertainty = 2*np.sqrt(2) / np.sqrt(np.pi) * tau

        return probabilities, uncertainty

    def _update_bandwidth(self, new_bandwidth):
        self.kernel.bandwidth = new_bandwidth
