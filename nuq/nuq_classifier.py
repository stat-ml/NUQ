from collections import defaultdict

import hnswlib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import softmax, logsumexp, log_softmax
from scipy.stats.stats import weightedtau
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.model_selection import GridSearchCV

from KDEpy.bw_selection import (
    improved_sheather_jones,
    silvermans_rule,
    scotts_rule,
)


class NuqClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        mixture_weight=0.2,
        kernel_type="RBF",
        method="hnsw",
        n_neighbors=20,
        tune_bandwidth=True,
        strategy="isj",
        bandwidth=np.ones(shape=(1, 1)),
        use_centroids=False,
        sparse=False,
    ):
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
        """
        self.mixture_weight = mixture_weight
        self.tune_bandwidth = tune_bandwidth
        self.strategy = strategy
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.method = method
        self.n_neighbors = n_neighbors
        self.use_centroids = use_centroids
        self.sparse = sparse

    @staticmethod
    def compute_centroids(embeddings, labels):
        """Computes centroid for each class.

        Args:
            embeddings ([type]): embeddings
            labels ([type]): labels (1..n_classes)

        Returns:
            centers ([type]): -//-
            labels ([type]): -//-
        """
        centers, labels_unique = [], []
        for c in np.unique(labels):
            mask = labels == c
            centers.append(np.mean(embeddings[mask], axis=0))
            labels_unique.append(c)
        return np.array(centers), np.array(labels_unique)

    @staticmethod
    def _get_log_kernel(name="RBF", bandwidth=np.ones(1)):
        """Constructs kernel function with signature:
          f(X: [N, d], Y: [N, d]) -> values [N]

        Args:
            name (str, optional): kernel type. Defaults to "RBF".

        Raises:
            ValueError: if `name` is unsupported

        Returns:
            [Callable]: kernel function
        """
        if name == "RBF":
            return lambda X, Y: -np.sum(
                (((X - Y) / bandwidth) ** 2) / 2, axis=-1
            )
        if name == "student":
            return lambda X, Y: -np.log1p(
                np.sum((((X - Y) / bandwidth) ** 2) / 2, axis=-1)
            )
        else:
            raise ValueError(f"Unsupported kernel: {name}")

    def _optimal_bandwidth(self, n_neighbors=(5, 20), n_points=5, cv=3):
        """Select optimal bandwidth value via fitting the kNN
        model with various bandwidths.

        Args:
            n_neighbors (int, optional): -//-. Defaults to 20.

        Returns:
            [type]: [description]
        """
        knn_min, knn_max = n_neighbors
        _, distances = self.index.knn_query(self.X_, k=knn_max)
        # Square root as l2 index is used
        mean_distances = distances.mean(0) ** 0.5
        classificator = NuqClassifier(
            tune_bandwidth=False,
            n_neighbors=n_neighbors,
        )
        grid = np.linspace(knn_min, knn_max, n_points, dtype=int)
        gs = GridSearchCV(
            classificator,
            param_grid={"bandwidth": grid},
            scoring="accuracy",
            cv=cv,
        )
        gs.fit(self.X_, self.y_)
        print(f"mean distance = {mean_distances}")
        print(f"{gs.best_params_['bandwidth']}")
        print("Best accuracy ", gs.best_score_)
        return gs.best_params_["bandwidth"]

    def _tune_kernel(self, strategy="isj", **kwargs):
        standard_strategies = {
            "isj": improved_sheather_jones,
            "silverman": silvermans_rule,
            "scott": scotts_rule,
        }
        if strategy in standard_strategies:
            method = standard_strategies[strategy]
            bandwidth = np.apply_along_axis(
                lambda x: method(x[..., None]), axis=0, arr=self.X_
            )
        elif strategy == "classification":
            bandwidth = self._optimal_bandwidth(**kwargs)
        else:
            raise ValueError("No such strategy")
        print(f"{bandwidth = }")
        return bandwidth

    def fit(self, X, y):
        """Prepares the model for inference:
          1. Computes centroid for each class [optional].
             This allows to reduce the number of points
             to the number of classes;
          2. Constructs kNN index;
          3. Tunes kernel bandwidth [optional].
             Uses `self.strategy` to find the optimal bandwidth.

        Args:
            X ([type]): embeddings
            y ([type]): labels (1..n_classes)

        Raises:
            ValueError: if `self.method` is unsupported

        Returns:
            self: fitted model
        """
        X, y = check_X_y(X, y)

        # 1. Compute centroid for each class
        if self.use_centroids:
            X, y = NuqClassifier.compute_centroids(embeddings=X, labels=y)

        # Save preprocessed input
        self.X_, self.y_ = X, y

        # Get prior class weights
        _, counts = np.unique(y, return_counts=True)
        self.class_weights_ = counts / len(y)
        self.log_prior_ = np.log(self.class_weights_)
        self.class_default_ = self.class_weights_.argmax()
        self.log_prior_default_ = np.log(
            self.class_weights_[self.class_default_]
        ) + np.log(self.mixture_weight)

        # 2. Construct kNN index
        if self.method == "hnsw":
            self.index = hnswlib.Index(space="l2", dim=self.X_.shape[1])
            self.index.init_index(max_elements=self.X_.shape[0])
            self.index.add_items(self.X_)
        else:
            raise ValueError

        # 3. Tune kernel bandwidth
        if self.tune_bandwidth:
            self.bandwidth = self._tune_kernel(
                n_neighbors=(5, self.n_neighbors)
            )

        self.log_kernel_ = NuqClassifier._get_log_kernel(
            self.kernel_type, self.bandwidth
        )

        return self

    def predict_proba(self, X, return_uncertainty=False):
        """Finds k = `n_neighbors` nearest neighbours for `current_embeddings` among
        `train_embeddings` and computes the `kernel` function for each pair

        Args:
            knn ([type]): object for nearest neighbours search
                        (FIX: depends on `method`)
            kernel ([type]): vectorized kernel function f(X_1, X_2)
                            (FIX: recomputes distances already available from kNN)
            current_embeddings ([type]): query embeddings
            train_embeddings ([type]): base embeddings
                                    (FIX: already loaded into knn)
            training_labels ([type]): corresponding labels
            n_neighbors ([type]): number of neighbours to compute weights for
            method (str, optional): kNN index type (FIX: available from `knn`).
                                    Defaults to 'faiss'.

        Raises:
            ValueError: if `method` is not supported

        Returns:
            w_raw ([type]): weights matrix
            selected_labels ([type]): corresponding labels
        """
        check_is_fitted(self)
        X = check_array(X)

        if self.method == "hnsw":
            I, _ = self.index.knn_query(X, k=self.n_neighbors)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        classes = []
        ps = []
        log_sigma2_totals = []

        # Get prediction for each row of nearest neighbours
        for i in range(I.shape[0]):
            idx = I[i, :]
            classes_cur, encoded = np.unique(self.y_[idx], return_inverse=True)
            # classes.append(classes_cur)

            # # If only single class is present
            # if len(classes_cur) == 1:
            #     ps.append(np.ones(1, dtype=np.float32))
            #     if return_uncertainty:
            #         log_sigma2_totals.append(-np.inf)
            #     continue

            log_kernel = self.log_kernel_(self.X_[idx, :], X[i, :])

            log_ps_cur = np.zeros(classes_cur.shape[0])
            # print()
            # print(f"{X[i, :] = }")

            # Get positions for each class
            indices = defaultdict(list)
            for i, v in enumerate(encoded):
                indices[v].append(i)

            log_ps_cur = np.array(
                [
                    logsumexp(log_kernel[indices[k]])
                    for k in range(len(classes_cur))
                ]
            )
            log_ps_total_cur = logsumexp(
                np.c_[
                    np.log1p(-self.mixture_weight) + log_ps_cur,
                    np.log(self.mixture_weight) + self.log_prior_[classes_cur],
                ],
                axis=1,
            )

            log_denominator = logsumexp(
                np.r_[
                    np.log1p(-self.mixture_weight) + log_ps_cur,
                    np.log(self.mixture_weight),
                ]
            )

            idx_max = np.argmax(log_ps_total_cur)
            if log_ps_total_cur[idx_max] > self.log_prior_default_:
                classes.append([classes_cur[idx_max]])
                log_numerator_p = log_ps_total_cur[idx_max]

            else:
                classes.append([self.class_default_])
                log_numerator_p = self.log_prior_default_

            log_ps_pred = log_numerator_p - log_denominator

            # print(f"{log_ps_pred = }")
            # Compute actual probabilities from logarithms
            ps_pred = np.exp(log_ps_pred)
            ps.append([ps_pred])

            if return_uncertainty:
                if log_ps_total_cur[idx_max] > self.log_prior_default_:
                    log_numerator_1mp = logsumexp(
                        np.r_[
                            log_ps_cur[:idx_max],
                            log_ps_cur[idx_max + 1 :],
                            np.log(self.mixture_weight)
                            + np.log1p(-self.log_prior_[idx_max]),
                        ]
                    )
                else:
                    log_numerator_1mp = logsumexp(
                        np.r_[
                            log_ps_cur,
                            np.log(self.mixture_weight)
                            + np.log1p(-self.log_prior_default_),
                        ]
                    )

                # print(f"{log_numerator_p = }")
                # print(f"{log_numerator_1mp = }")
                # print(f"{log_denominator = }")

                log_sigma2_total = (
                    log_numerator_p
                    + log_numerator_1mp
                    - 3 * log_denominator
                    + X.shape[1] / 2 * np.log(np.pi)
                )
                log_sigma2_totals.append(log_sigma2_total)

        sizes = np.array([len(arr) for arr in classes])
        indptr = np.r_[0, np.cumsum(sizes)]
        indices = np.concatenate(classes)
        data = np.concatenate(ps, dtype=np.float32)

        n_classes = self.y_.max() + 1
        probs = csr_matrix(
            (data, indices, indptr), shape=(X.shape[0], n_classes)
        )
        if not self.sparse:
            probs = probs.toarray()

        if return_uncertainty:
            return probs, np.array(log_sigma2_totals)
        else:
            return probs

    def predict(self, X, return_uncertainty=False):
        if return_uncertainty:
            probs, uncertainty = self.predict_proba(X, return_uncertainty=True)
            return probs.argmax(axis=1), uncertainty
        else:
            probs = self.predict_proba(X, return_uncertainty=False)
            return probs.argmax(axis=1)