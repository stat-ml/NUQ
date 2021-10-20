from functools import partial

import numpy as np
import pandas as pd
import ray
from KDEpy.bw_selection import (
    improved_sheather_jones,
    scotts_rule,
    silvermans_rule,
)
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y
from tqdm.auto import tqdm

from .misc import parse_param
from .ray_utils import HNSWActor, predict_log_proba_batch, to_iterator


class NuqClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        log_pN=0.0,
        kernel_type="RBF",
        n_neighbors=20,
        tune_bandwidth="classification",
        bandwidth=1.0,
        use_centroids=False,
        sparse=True,
        verbose=False,
        batch_size=1000,
        random_seed=42,
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
        self.log_pN = log_pN
        self.tune_bandwidth = tune_bandwidth
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.n_neighbors = n_neighbors
        self.use_centroids = use_centroids
        self.sparse = sparse
        self.verbose = verbose
        self.batch_size = batch_size
        self.random_seed = random_seed

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
    def _get_log_kernel(name="RBF", bandwidth=1.0):
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

    def _optimal_bandwidth(
        self, X, y, n_points=5, n_folds=10, n_samples=3, verbose=0
    ):
        """Select optimal bandwidth value via fitting the kNN
        model with various bandwidths.

        Args:
            n_neighbors (int, optional): -//-. Defaults to 20.

        Returns:
            [type]: [description]
        """
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.random_seed
        )
        it = skf.split(X, y)

        grid = None
        results = {}
        for i, (train_idx, val_idx) in tqdm(
            zip(range(n_samples), it),
            total=n_samples,
            disable=(not verbose),
            desc="Tuning bandwidth",
        ):
            # Prepare a new Nuq instance with disabled `tune_bandwidth`
            params = self.get_params()
            params.update(tune_bandwidth=None, verbose=False)
            nuq = NuqClassifier(**params)

            # Build kNN index
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            nuq.fit(X_train, y_train)

            # If grid is not set yet, initialize it
            if grid is None:
                _, dists = ray.get(
                    nuq.index_.knn_query.remote(nuq.X_ref_, return_dist=True)
                )
                dists_mean = dists.mean(axis=0)
                left, right = dists_mean[1], dists_mean[-1]
                grid = np.linspace(left, right, n_points)

            # Compute score for every bandwidth
            scores = []
            for bandwidth in grid:
                nuq.log_kernel_ = NuqClassifier._get_log_kernel(
                    nuq.kernel_type, bandwidth
                )
                scores.append(nuq.score(X_val, y_val))
            results[f"fold_{i}"] = scores
            del nuq

        results = pd.DataFrame(results)
        scores_mean = results.mean(axis=1)
        best_idx = scores_mean.argmax()

        return grid[best_idx], scores_mean[best_idx]

    def _tune_kernel(self, X, y, strategy="isj"):
        standard_strategies = {
            "isj": improved_sheather_jones,
            "silverman": silvermans_rule,
            "scott": scotts_rule,
        }
        if strategy in standard_strategies:
            method = standard_strategies[strategy]
            bandwidth = np.apply_along_axis(
                lambda x: method(x[..., None]), axis=0, arr=X
            )
            print(f"  [{strategy.upper()}] {bandwidth = }")
        elif "classification" in strategy:
            _, kwargs = parse_param(strategy)
            kwargs = dict(map(lambda x: (x[0], int(x[1])), kwargs.items()))
            bandwidth, score = self._optimal_bandwidth(X, y, **kwargs)
            print(f"  [{strategy.upper()}] {bandwidth = } ({score = })")
        else:
            raise ValueError("No such strategy")
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
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y)

        # 1. Compute centroid for each class
        if self.use_centroids:
            X, y = NuqClassifier.compute_centroids(embeddings=X, labels=y)
        self.n_classes_ = y.max() + 1

        # Get prior class weights
        _, counts = np.unique(y, return_counts=True)

        log_prior = np.log(counts) - np.log(len(y))
        self.class_default_ = log_prior.argmax()
        self.log_prior_default_ = log_prior[self.class_default_] + self.log_pN

        # Move log prior vector to shared memory
        self.log_prior_ref_ = ray.put(log_prior)

        # 2. Tune kernel bandwidth
        if self.tune_bandwidth is not None:
            self.bandwidth = self._tune_kernel(
                X, y, strategy=self.tune_bandwidth
            )

        self.log_kernel_ = NuqClassifier._get_log_kernel(
            self.kernel_type, self.bandwidth
        )

        # 3. Construct kNN index on remote

        # Move preprocessed input to shared memory
        self.X_ref_, self.y_ref_ = ray.put(X), ray.put(y)

        self.index_ = HNSWActor.remote(
            self.X_ref_,
            n_neighbors=self.n_neighbors,
            random_seed=self.random_seed,
        )

        return self

    def predict_proba(self, X, return_uncertainty=False, batch_size=None):
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
        if batch_size is None:
            batch_size = self.batch_size

        # Move query to the shared memory
        X_ref = ray.put(X)
        idx_ref = self.index_.knn_query.remote(X_ref, return_dist=False)

        predict_log_proba_handle = partial(
            predict_log_proba_batch.remote,
            self.X_ref_,
            self.y_ref_,
            X_ref,
            idx_ref,
            batch_size=batch_size,
            log_kernel=self.log_kernel_,
            log_prior=self.log_prior_ref_,
            log_pN=self.log_pN,
            log_prior_default=self.log_prior_default_,
            class_default=self.class_default_,
            return_uncertainty=return_uncertainty,
        )

        res_refs = []
        for i in range(0, X.shape[0], batch_size):
            res_refs.append(predict_log_proba_handle(i))

        res = []
        for x in tqdm(
            to_iterator(res_refs),
            total=len(res_refs),
            disable=(not self.verbose),
        ):
            res.append(x)

        pred, log_proba, log_unc = map(np.concatenate, zip(*res))

        indptr = np.r_[np.arange(pred.shape[0] + 1, dtype=np.int64)]
        indices = pred
        data = np.exp(log_proba)

        probs = csr_matrix(
            (data, indices, indptr), shape=(X.shape[0], self.n_classes_)
        )
        if not self.sparse:
            probs = probs.toarray()

        if return_uncertainty:
            return probs, log_unc
        else:
            return probs

    def predict(self, X, return_uncertainty=False):
        probs = self.predict_proba(X, return_uncertainty=return_uncertainty)
        if return_uncertainty:
            probs, uncertainty = probs
        probs = self.label_encoder_.inverse_transform(
            np.array(probs.argmax(axis=1)).ravel()
        )

        if return_uncertainty:
            return probs, uncertainty
        else:
            return probs
