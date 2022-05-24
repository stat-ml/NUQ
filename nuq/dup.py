# level 1:
# k close neighbors prediction
from statistics import mode
import numpy as np
import hnswlib


class NuqClassifierDup:
    def __init__(self, kernel='RBF', n_neighbors=20):
        self.index = None
        self.labels = None

        self.kernel = kernel
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        self.index = hnswlib.Index(space='l2', dim=2)
        self.index.init_index(max_elements=len(x), M=16, ef_construction=200)
        self.index.add_items(x)
        self.labels = y

    def predict(self, x, return_uncertainty=None):
        if self.index is None:
            raise RuntimeError('You should fit the model first')
        closest = self.index.knn_query(x, self.n_neighbors)
        neighbor_labels = self.labels[closest[0]]
        predictions = np.array([int(mode(labels)) for labels in neighbor_labels])
        return predictions

    def predict_proba(self, x, return_uncertainty=None):
        pass
