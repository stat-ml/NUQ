import hnswlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_blobs
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from nw_uncertainty.method.compute_expectations import (get_kernel, compute_weights, compute_centroids)
from nw_uncertainty.utils.aux_functions import get_logsumexps, safe_logsumexp
from nw_uncertainty.utils.bandwidth_selection import tune_kernel


def plot_data(X, y):
    plt.close()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.tight_layout()
    plt.show()


def make_data(total_size=5000, centers=np.array([[-4., -4.], [0., 4.]])):
    X, y = make_blobs(n_samples=total_size, n_features=2, centers=centers)
    return X, y


class NewNW(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel_type="RBF", method="hnsw", n_neighbors=20, coeff=1e-10, tune_bandwidth=True,
                 strategy='isj',
                 bandwidth=np.array([1., ]), use_centroids=False):
        """

        :param kernel_type:
        :param method:
        :param n_neighbors:
        :param coeff: correction, which is added to numenator and denominator
        :param tune_bandwidth: whether tune bandwidth, of use fixed
        :param strategy: tuning strategy. Options are ['isj', 'silverman', 'scott', 'classification']
        :param bandwidth: bandwidth. Should be a numpy array of size () or (dim, )
        :param use_centroids: whether use centroids or not
        """
        self.tune_bandwidth = tune_bandwidth
        self.strategy = strategy
        self.coeff = coeff
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.method = method
        self.n_neighbors = n_neighbors
        self.use_centroids = use_centroids

    def fit(self, X, y):
        """
        The function fits
        :param X: Training embeggins
        :param y: Training labels
        :return: trained model
        """
        X, y = check_X_y(X, y)
        self.kernel = get_kernel(self.kernel_type, bandwidth=self.bandwidth)
        targets = y.reshape(-1)
        if not self.use_centroids:
            y_ohe = np.eye(np.max(y) + 1)[targets]
        else:
            X, y = compute_centroids(embeddings=X, labels=targets)
            y_ohe = np.eye(np.max(y) + 1)[y]

        self.training_embeddings_ = X
        self.training_labels_ = y_ohe

        if self.method == 'hnsw':
            self.fast_knn = hnswlib.Index(space='l2', dim=self.training_embeddings_.shape[1])
            self.fast_knn.init_index(max_elements=self.training_embeddings_.shape[0])
            self.fast_knn.set_ef(10)
            self.fast_knn.add_items(self.training_embeddings_)
        else:
            raise ValueError

        if self.tune_bandwidth:
            self.bandwidth = tune_kernel(X=self.training_embeddings_, y=y, strategy=self.strategy, knn=self.fast_knn,
                                         constructor=NewNW,
                                         n_neighbors=self.n_neighbors)
            self.kernel = get_kernel(self.kernel_type, bandwidth=self.bandwidth)
        return self

    def predict_proba(self, X, batch_size=50000, return_uncertainties=False):
        check_is_fitted(self)
        X = check_array(X)
        batches = [(i, i + batch_size) for i in range(0, len(X), batch_size)]
        log_in_class_all = np.array([])
        log_out_class_all = np.array([])
        log_full_all = np.array([])
        for batch in batches:
            embedding_batch = X[batch[0]: batch[1]]
            log_weights, selected_labels = compute_weights(knn=self.fast_knn, kernel=self.kernel,
                                                           current_embeddings=embedding_batch,
                                                           train_embeddings=self.training_embeddings_,
                                                           training_labels=self.training_labels_,
                                                           n_neighbors=self.n_neighbors,
                                                           method=self.method)
            in_class, out_class, full = get_logsumexps(log_weights=log_weights, targets=selected_labels)

            if log_in_class_all.shape[0] == 0:
                log_in_class_all = in_class
                log_out_class_all = out_class
                log_full_all = full
            else:
                log_in_class_all = np.concatenate([log_in_class_all, in_class])
                log_out_class_all = np.concatenate([log_out_class_all, out_class])
                log_full_all = np.concatenate([log_full_all, full])

        ###### CORRECTION ######
        log_in_class_all_corrected = safe_logsumexp(
            np.concatenate([log_in_class_all[None], np.broadcast_to(np.log(self.coeff), log_in_class_all.shape)[None]],
                           axis=0), axis=0)
        log_out_class_all_corrected = safe_logsumexp(
            np.concatenate([log_out_class_all[None], np.broadcast_to(np.log(self.coeff), log_in_class_all.shape)[None]],
                           axis=0), axis=0)
        log_prob_in_marginalized = safe_logsumexp(log_in_class_all_corrected, axis=1, keepdims=True)
        log_1mprob_pre = safe_logsumexp(log_in_class_all[..., None],
                                        b=(1 - np.eye(self.training_labels_.shape[1])[None]),
                                        axis=1)
        log_1mprob_pre = safe_logsumexp(
            np.concatenate([log_1mprob_pre[None],
                            np.log(self.training_labels_.shape[1] - 1) +
                            np.broadcast_to(np.log(self.coeff), log_1mprob_pre.shape)[None]], axis=0), axis=0)
        ########################

        log_prob = log_in_class_all_corrected - log_prob_in_marginalized
        log_1mprob = log_1mprob_pre - log_prob_in_marginalized

        Ue = None
        Ua = None
        total_uncertainty = None

        if return_uncertainties:
            mean_class_prob = np.mean(self.training_labels_, axis=0, keepdims=True)

            Ue_pre = 1.5 * np.log(2) - 0.25 * np.log(np.pi) + 0.5 * (
                    log_in_class_all_corrected + log_out_class_all_corrected - 3 * log_full_all)
            Ue = safe_logsumexp(Ue_pre + np.log(mean_class_prob), axis=1)
            Ua = safe_logsumexp(
                np.minimum(log_in_class_all_corrected, log_out_class_all_corrected) + np.log(
                    mean_class_prob), axis=1)  ## Here I also have to subtract denominator
            total_uncertainty = safe_logsumexp(np.concatenate([Ua[None], Ue[None]], axis=0), axis=0)

        return {
            "probs": log_prob,
            "1mprobs": log_1mprob,
            "epistemic": Ue,
            "aleatoric": Ua,
            "total": total_uncertainty
        }

    def predict(self, X, batch_size=50000):
        return np.argmax(self.predict_proba(X, batch_size=batch_size)["probs"], axis=-1)
