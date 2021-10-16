import hnswlib
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from .aux_functions import (
    get_kernel, compute_weights, get_nw_mean_estimate, get_nw_mean_estimate_regerssion, p_hat_x,
    asymptotic_var, half_gaussian_mean, log_asymptotic_var,
    log_half_gaussian_mean, compute_centroids, MyKNN
)
from ..utils.bandwidth_selection import tune_kernel


class NuqClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel_type="RBF", method="hnsw", n_neighbors=20, coeff=0.00001, tune_bandwidth=True,
                 strategy='isj',
                 bandwidth=np.array([1., ]), precise_computation=True, use_centroids=False, use_uniform_prior=True):
        """

        :param kernel_type:
        :param method:
        :param n_neighbors:
        :param coeff: correction, which is added to numenator and denominator (relevant for precise_computation=False)
        :param tune_bandwidth: whether tune bandwidth, of use fixed
        :param strategy: tuning strategy. Options are ['isj', 'silverman', 'scott', 'classification']
        :param bandwidth: bandwidth. Should be a numpy array of size () or (dim, )
        :param precise_computation: if True, everything is computed in terms of log
        :param use_centroids: whether use centroids or not
        :param use_uniform_prior: If true, we induce prior that at infinity we predict uniform distribution over classes
        """
        self.use_uniform_prior = use_uniform_prior
        self.tune_bandwidth = tune_bandwidth
        self.strategy = strategy
        self.coeff = coeff
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.method = method
        self.precise_computation = precise_computation
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
        self.squared_kernel_int, self.kernel = get_kernel(self.kernel_type, bandwidth=self.bandwidth,
                                                          precise_computation=self.precise_computation)
        if self.precise_computation:
            self.squared_kernel_int = np.log(self.squared_kernel_int)
        targets = y.reshape(-1)
        if not self.use_centroids:
            y_ohe = np.eye(np.max(y) + 1)[targets]
        else:
            X, y = compute_centroids(embeddings=X, labels=targets)
            y_ohe = np.eye(np.max(y) + 1)[y]

        self.training_embeddings_ = X
        self.training_labels_ = y_ohe
        self.n_classes = len(np.unique(y))

        if self.method == 'hnsw':
            self.fast_knn = hnswlib.Index(space='l2', dim=self.training_embeddings_.shape[1])
            self.fast_knn.init_index(max_elements=self.training_embeddings_.shape[0])
            self.fast_knn.set_ef(10)
            self.fast_knn.add_items(self.training_embeddings_)
        elif self.method == 'all_data':
            self.n_neighbors = self.training_embeddings_.shape[0]
            self.fast_knn = MyKNN(self.training_embeddings_)
        else:
            raise ValueError

        if self.tune_bandwidth:
            self.bandwidth = tune_kernel(X=self.training_embeddings_, y=y, strategy=self.strategy, knn=self.fast_knn,
                                         constructor=NuqClassifier, precise_computation=self.precise_computation,
                                         n_neighbors=self.n_neighbors)
            self.squared_kernel_int, self.kernel = get_kernel(self.kernel_type, bandwidth=self.bandwidth,
                                                              precise_computation=self.precise_computation)
            if self.precise_computation:
                self.squared_kernel_int = np.log(self.squared_kernel_int)
        return self

    def _get_nw_estimates(self, X, batch_size):
        batches = [(i, i + batch_size) for i in range(0, len(X), batch_size)]
        f_hat = np.array([])
        f1_hat = np.array([])
        for batch in batches:
            embedding_batch = X[batch[0]: batch[1]]
            weights, selected_labels = compute_weights(knn=self.fast_knn, kernel=self.kernel,
                                                       current_embeddings=embedding_batch,
                                                       train_embeddings=self.training_embeddings_,
                                                       training_labels=self.training_labels_,
                                                       n_neighbors=self.n_neighbors,
                                                       method=self.method)

            output = get_nw_mean_estimate(targets=selected_labels, weights=weights, coeff=self.coeff,
                                          precise_computation=self.precise_computation, n_clasees=self.n_classes,
                                          use_uniform_prior=self.use_uniform_prior)

            current_f_hat = output["f_hat"]
            current_f1_hat = output["f1_hat"]
            if f_hat.shape[0] == 0:
                f_hat = current_f_hat
                f1_hat = current_f1_hat
            else:
                f_hat = np.concatenate([f_hat, current_f_hat])
                f1_hat = np.concatenate([f1_hat, current_f1_hat])

        return f_hat, f1_hat

    def predict_uncertainty(self, X, batch_size=50000):
        batches = [(i, i + batch_size) for i in range(0, len(X), batch_size)]
        Ue_total = np.array([])
        Ua_total = np.array([])
        Ut_total = np.array([])
        for batch in batches:
            X_batch = X[batch[0]: batch[1]]
            f_hat_x_full = self.get_kde(X_batch, batch_size=batch_size)
            output = self.predict_proba(X_batch, batch_size=batch_size)
            f_hat_y_x_full = output["probs"]
            f1_hat_y_x_full = output["probsm1"]

            f_hat_x = f_hat_x_full
            f_hat_y_x = f_hat_y_x_full
            f1_hat_y_x = f1_hat_y_x_full
            if not self.precise_computation:
                sigma_hat_est = np.max(f_hat_y_x * f1_hat_y_x, axis=1, keepdims=True)
                as_var = asymptotic_var(sigma_est=sigma_hat_est, f_est=f_hat_x, bandwidth=self.bandwidth,
                                        n=self.n_neighbors)

                Ue = half_gaussian_mean(asymptotic_var=as_var).squeeze()
                Ua = np.min(f1_hat_y_x, axis=1, keepdims=True).squeeze()
                total_uncertainty = Ue + Ua
            else:
                sigma_hat_est = np.max(f_hat_y_x + f1_hat_y_x, axis=1, keepdims=True)
                if not self.use_uniform_prior:
                    broadcast_shape = (1, sigma_hat_est.shape[0], sigma_hat_est.shape[1])
                    sigma_hat_est = logsumexp(np.concatenate(
                        [sigma_hat_est[None], np.log(self.coeff) * np.ones(shape=broadcast_shape)],
                        axis=0), axis=0)
                log_as_var = log_asymptotic_var(log_sigma_est=sigma_hat_est, log_f_est=f_hat_x,
                                                bandwidth=self.bandwidth,
                                                n=self.n_neighbors, dim=self.training_embeddings_.shape[1],
                                                squared_kernel_int=self.squared_kernel_int)

                Ue = log_half_gaussian_mean(asymptotic_var=log_as_var).squeeze()
                Ua = np.min(f1_hat_y_x, axis=1, keepdims=True)
                if not self.use_uniform_prior:
                    Ua = logsumexp(
                        np.concatenate([Ua[None], np.log(self.coeff) * np.ones(shape=broadcast_shape)], axis=0),
                        axis=0)
                Ua = Ua.squeeze()

                # Ue = np.clip(Ue, a_min=-1e40, a_max=None)

                total_uncertainty = logsumexp(np.concatenate([Ua[None], Ue[None]], axis=0), axis=0).squeeze()

            if Ue_total.shape[0] == 0:
                Ue_total = Ue
                Ua_total = Ua
                Ut_total = total_uncertainty
            else:
                Ue_total = np.concatenate([Ue_total, Ue])
                Ua_total = np.concatenate([Ua_total, Ua])
                Ut_total = np.concatenate([Ut_total, total_uncertainty])
        return {"epistemic": Ue_total, "aleatoric": Ua_total, "total": Ut_total}

    def predict_proba(self, X, batch_size=50000):
        check_is_fitted(self)
        X = check_array(X)
        f_hat, f1_hat = self._get_nw_estimates(X=X, batch_size=batch_size)
        probs = f_hat
        probsm1 = f1_hat

        output = {"probs": probs, "probsm1": probsm1}

        return output

    def get_kde(self, X, batch_size=50000):
        batches = [(i, i + batch_size) for i in range(0, len(X), batch_size)]
        f_hat_x = np.array([])
        for batch in batches:
            X_batch = X[batch[0]: batch[1]]
            weights, labels = compute_weights(knn=self.fast_knn,
                                              kernel=self.kernel,
                                              current_embeddings=X_batch,
                                              train_embeddings=self.training_embeddings_,
                                              training_labels=self.training_labels_,
                                              n_neighbors=self.n_neighbors,
                                              method=self.method)
            f_hat_x_current = p_hat_x(weights=weights, n=self.n_neighbors, h=self.bandwidth,
                                      dim=self.training_embeddings_.shape[1],
                                      precise_computation=self.precise_computation)
            if f_hat_x.shape[0] == 0:
                f_hat_x = f_hat_x_current
            else:
                f_hat_x = np.concatenate([f_hat_x, f_hat_x_current])
        return f_hat_x

    def predict(self, X, batch_size=50000):
        return np.argmax(self.predict_proba(X, batch_size=batch_size)["probs"], axis=-1)




class NuqRegressor(BaseEstimator):
    def __init__(self, kernel_type="RBF", method="hnsw", n_neighbors=20, coeff=0.00001, tune_bandwidth=True,
                 strategy='isj',
                 bandwidth=np.array([1., ]), precise_computation=True):
        """

        :param kernel_type:
        :param method:
        :param n_neighbors:
        :param coeff: correction, which is added to numenator and denominator (relevant for precise_computation=False)
        :param tune_bandwidth: whether tune bandwidth, of use fixed
        :param strategy: tuning strategy. Options are ['isj', 'silverman', 'scott', 'classification']
        :param bandwidth: bandwidth. Should be a numpy array of size () or (dim, )
        :param precise_computation: if True, everything is computed in terms of log
        :param use_centroids: whether use centroids or not
        :param use_uniform_prior: If true, we induce prior that at infinity we predict uniform distribution over classes
        """
        self.tune_bandwidth = tune_bandwidth
        self.strategy = strategy
        self.coeff = coeff
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.method = method
        self.precise_computation = precise_computation
        self.n_neighbors = n_neighbors



    def fit(self, X, y):
        """
        The function fits
        :param X: Training embeggins
        :param y: Training labels
        :return: trained model
        """
        X, y = check_X_y(X, y)
        self.squared_kernel_int, self.kernel = get_kernel(self.kernel_type, bandwidth=self.bandwidth,
                                                          precise_computation=self.precise_computation)
        if self.precise_computation:
            self.squared_kernel_int = np.log(self.squared_kernel_int)
        # targets = y.reshape(-1)
        # if not self.use_centroids:
        #     y_ohe = np.eye(np.max(y) + 1)[targets]
        # else
        #     X, y = compute_centroids(embeddings=X, labels=targets)
        #     y_ohe = np.eye(np.max(y) + 1)[y]

        self.training_embeddings_ = X
        self.training_labels_ = y.reshape(*y.shape, 1)
        self.n_classes = 1

        if self.method == 'hnsw':
            self.fast_knn = hnswlib.Index(space='l2', dim=self.training_embeddings_.shape[1])
            self.fast_knn.init_index(max_elements=self.training_embeddings_.shape[0])
            self.fast_knn.set_ef(10)
            self.fast_knn.add_items(self.training_embeddings_)
        elif self.method == 'all_data':
            self.n_neighbors = self.training_embeddings_.shape[0]
            self.fast_knn = MyKNN(self.training_embeddings_)
        else:
            raise ValueError

        if self.tune_bandwidth:
            self.bandwidth = tune_kernel(X=self.training_embeddings_, y=y, strategy=self.strategy, knn=self.fast_knn,
                                         constructor=NuqRegressor, precise_computation=self.precise_computation,
                                         n_neighbors=self.n_neighbors)
            self.squared_kernel_int, self.kernel = get_kernel(self.kernel_type, bandwidth=self.bandwidth,
                                                              precise_computation=self.precise_computation)
            if self.precise_computation:
                self.squared_kernel_int = np.log(self.squared_kernel_int)
        return self

    
    def _get_nw_estimates(self, X, batch_size):
        batches = [(i, i + batch_size) for i in range(0, len(X), batch_size)]
        f_hat = np.array([])
        f_sq_hat = np.array([])
        for batch in batches:
            embedding_batch = X[batch[0]: batch[1]]
            weights, selected_labels = compute_weights(knn=self.fast_knn, kernel=self.kernel,
                                                       current_embeddings=embedding_batch,
                                                       train_embeddings=self.training_embeddings_,
                                                       training_labels=self.training_labels_,
                                                       n_neighbors=self.n_neighbors,
                                                       method=self.method)

            output = get_nw_mean_estimate_regerssion(
                targets=selected_labels, 
                weights=weights,
                precise_computation=self.precise_computation
            )

            current_f_hat = output["f_hat"]
            current_f_sq_hat = output["f_sq_hat"]
            if f_hat.shape[0] == 0:
                f_hat = current_f_hat
                f_sq_hat = current_f_sq_hat
            else:
                f_hat = np.concatenate([f_hat, current_f_hat])
                f_sq_hat = np.concatenate([f_sq_hat, current_f_sq_hat])

        return f_hat, f_sq_hat


    def get_kde(self, X, batch_size=50000):
        batches = [(i, i + batch_size) for i in range(0, len(X), batch_size)]
        f_hat_x = np.array([])
        for batch in batches:
            X_batch = X[batch[0]: batch[1]]
            weights, labels = compute_weights(knn=self.fast_knn,
                                              kernel=self.kernel,
                                              current_embeddings=X_batch,
                                              train_embeddings=self.training_embeddings_,
                                              training_labels=self.training_labels_,
                                              n_neighbors=self.n_neighbors,
                                              method=self.method)
            f_hat_x_current = p_hat_x(weights=weights, n=self.n_neighbors, h=self.bandwidth,
                                      dim=self.training_embeddings_.shape[1],
                                      precise_computation=self.precise_computation)
            if f_hat_x.shape[0] == 0:
                f_hat_x = f_hat_x_current
            else:
                f_hat_x = np.concatenate([f_hat_x, f_hat_x_current])
        return f_hat_x
        

    def predict_uncertainty(self, X, batch_size=50000, infinity=np.inf):
        batches = [(i, i + batch_size) for i in range(0, len(X), batch_size)]
        Ue_total = np.array([])
        Ua_total = np.array([])
        Ut_total = np.array([])
        for batch in batches:
            X_batch = X[batch[0]: batch[1]]
            f_hat_x = self.get_kde(X_batch, batch_size=batch_size)
            if self.precise_computation:
                ininity_points = f_hat_x < -20
                f_hat_x = np.exp(f_hat_x)
            f_hat, f_sq_hat = self._get_nw_estimates(X_batch, batch_size=batch_size)

            sigma_est = f_sq_hat - f_hat ** 2
            as_var = asymptotic_var(sigma_est=sigma_est, f_est=f_hat_x, bandwidth=self.bandwidth,
                                    n=self.n_neighbors)

            Ue = as_var
            Ue[ininity_points] = infinity
            Ue[Ue > infinity] = infinity
        
            Ua = sigma_est
            total_uncertainty = Ue + Ua
            total_uncertainty[ininity_points] = infinity
            total_uncertainty[total_uncertainty > infinity] = infinity

            if Ue_total.shape[0] == 0:
                Ue_total = Ue
                Ua_total = Ua
                Ut_total = total_uncertainty
            else:
                Ue_total = np.concatenate([Ue_total, Ue])
                Ua_total = np.concatenate([Ua_total, Ua])
                Ut_total = np.concatenate([Ut_total, total_uncertainty])
        return {"epistemic": Ue_total, "aleatoric": Ua_total, "total": Ut_total}

    def predict(self, X, batch_size=50000):
        f_hat, f_sq_hat = self._get_nw_estimates(X, batch_size=batch_size)
        return  f_hat