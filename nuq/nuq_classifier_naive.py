import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y
from tqdm.auto import tqdm


class NUQClassifierNaive(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 kernel_type="rbf",
                 bandwidth=0.01,
                 ):
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        #self.label_encoder_ = LabelEncoder()
        #y = self.label_encoder_.fit_transform(y)

        self.X_ = X
        self.y_ = y

        return self

    def _compute_eta_hat(self, X):
        # precompute kernel matrix
        self.K_ = pairwise_kernels(self.X_, X, metric=self.kernel_type, gamma=self.bandwidth)
        eta_hat = (self.K_ * self.y_[:, None]).sum(axis=0) / self.K_.sum(axis=0)

        return eta_hat

    def predict(self, X):
        # eta = P(Y = 1 | X = x)
        eta_hat = self._compute_eta_hat(X)

        #predicted_labels = self.label_encoder_.inverse_transform(
        #    np.array(eta_hat.argmax(axis=-1)).ravel()
        #)

        return np.c_[1 - eta_hat, eta_hat].argmax(axis=-1)

    def predict_proba(self, X):
        eta_hat = self._compute_eta_hat(X)

        N, d = X.shape

        # Integral of the squared kernel
        # https://math.stackexchange.com/questions/2849580/kernel-density-estimation-integral-of-squared-kernel
        self.K2_ = 1.0 / (2 ** d * (np.pi ** (d / 2.0)))

        p_hat = self.K_.sum(axis=0)
        sigma_hat2 = eta_hat * (1.0 - eta_hat)

        tau2 = 1 / (N * self.bandwidth ** d) * sigma_hat2 / p_hat * self.K2_

        tau = np.sqrt(tau2)

        unc_aleatoric = np.c_[eta_hat, 1 - eta_hat].min(axis=-1)
        unc_epistemic = 2.0 * np.sqrt(2.0 / np.pi) * tau

        return eta_hat, unc_aleatoric, unc_epistemic, unc_aleatoric + unc_epistemic
