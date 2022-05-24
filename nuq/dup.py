# level 2:
# kernel classification

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
        self.gamma = 1 / bandwidth**2

        if kernel_type == 'RBF':
            self._function = self._rbf
        else:
            raise ValueError(
                f'Unknown kernel type, should be one of the following: {self.KERNEL_TYPES}'
            )

    def __call__(self, x1, x2=None):
        return self._function(x1, x2)

    def _rbf(self, distance, _):
        return np.exp(-distance*self.gamma)

    # def _rbf(self, x1, x2):
    #     raise NotImplemented()
        # if x2 is None:
        #     return x1 /
        # return rbf_kernel(x1, x2, gamma=self.gamma)


class NuqClassifierDup:
    def __init__(self, kernel_type='RBF', n_neighbors=20):
        self.index = None
        self.labels = None
        self.n_classes = None
        self.bandwidth = 1

        self.kernel = Kernel(kernel_type, bandwidth=1)
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        self.index = hnswlib.Index(space='l2', dim=2)
        self.index.init_index(max_elements=len(x), M=16, ef_construction=200)
        self.index.add_items(x)
        self.labels = y
        self.n_classes = np.max(y) + 1

    def predict(self, x, uncertainty_type=None):
        if self.index is None:
            raise RuntimeError('You should fit the model first')
        uncertainty = None

        indices, squared_distances = self.index.knn_query(x, self.n_neighbors)
        neighbor_labels = self.labels[indices]

        similarities = self.kernel(squared_distances)
        # similarities = np.exp(-squared_distances / self.bandwidth**2)
        normalization = np.sum(similarities, axis=1)
        predictions = np.round(np.sum(similarities, axis=1, where=(neighbor_labels == 1)) / normalization)
        # import ipdb; ipdb.set_trace()
        # predictions = np.array([int(mode(labels)) for labels in neighbor_labels])
        return predictions, uncertainty

    def predict_proba(self, x, return_uncertainty=None):
        pass
