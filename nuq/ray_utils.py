from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple

import hnswlib
import numpy as np
import ray
from scipy.special import logsumexp


@ray.remote
class HNSWActor:
    """Ray Actor interface for HNSW index to leverage the shared storage of `X_base` (vector index)."""

    def __init__(
        self, X_base, M=16, ef_construction=200, random_seed=42, n_neighbors=20
    ):
        self.M = M
        self.ef_construction = ef_construction
        self.random_seed = random_seed
        self.n_neighbors = n_neighbors

        N, dim = X_base.shape
        self.index = hnswlib.Index(space="l2", dim=dim)
        self.index.init_index(
            max_elements=N, M=16, ef_construction=200, random_seed=42
        )
        self.index.add_items(X_base)

    def knn_query(self, X, sl=np.s_[:, :], return_dist=True):
        idx, dist = self.index.knn_query(X[sl], k=self.n_neighbors)
        if return_dist:
            return idx, dist
        else:
            return idx


def predict_log_proba_single(
    X_base: np.ndarray,
    y_base: np.ndarray,
    X_query: np.ndarray,
    log_kernel: Callable,
    log_prior: np.ndarray,
    log_pN: float,
    log_prior_default: float,
    class_default: int,
    return_uncertainty: bool = False,
) -> Tuple[int, float, Optional[float]]:
    """Predict uncertainty for a single entry `X_query` given its
    neighbors (`base`) and their labels.

    Parameters
    ----------
        X_base : np.ndarray[k, dim] of floats
            Neighbors of the query point
        y_base : np.ndarray[k]
            Corresponding labels
        X_query : np.ndarray[dim]
            A single point to make prediction for
        log_kernel : Callable
            Kernel function to be called: `log_kernel(X_base, X_query)`
        log_prior : np.ndarray[n_classes]
            Prior distribution on the classes
        log_pN : float
            Prior importance hyperparameter
        log_prior_default : float
            Highest priot probability
        class_default : int
            Class with highest priot probability
        return_uncertainty : bool, optional
            Whether to compute uncertainty, by default False

    Returns
    -------
        pred : int
            Class with top probability
        log_proba : float
            Predicted class log probability
        log_unc : float
            Optional log uncertainty, -1. by default
    """
    # Create list of all present classes and their corresponding positions
    classes_cur, encoded = np.unique(y_base, return_inverse=True)

    # Compute kernel values for each pair of points
    log_kernel_vals = log_kernel(X_base, X_query)

    # Get positions for each class
    indices = defaultdict(list)
    for i, v in enumerate(encoded):
        indices[v].append(i)

    log_ps_cur = np.array(
        [
            logsumexp(log_kernel_vals[indices[k]])
            for k in range(len(classes_cur))
        ]
    )

    # Compute numerator for each class
    log_ps_total_cur = logsumexp(
        np.c_[
            log_ps_cur,
            log_pN + log_prior[classes_cur],
        ],
        axis=1,
    )

    # Compute denominator (it is the same for all classes)
    log_denominator = logsumexp(
        np.r_[
            log_ps_cur,
            log_pN,
        ]
    )

    # Select class with top probability
    idx_max = np.argmax(log_ps_total_cur)
    # If max probability is greater than all prior probabilities,
    # predict the corresponding class
    if log_ps_total_cur[idx_max] > log_prior_default:
        class_pred = classes_cur[idx_max]
        log_numerator_p = log_ps_total_cur[idx_max]
    # If max probability is still less than any prior probability
    # then just predict the top prior class
    else:
        class_pred = class_default
        log_numerator_p = log_prior_default

    # Compute the Nadaraya-Watson estimator
    log_ps_pred = log_numerator_p - log_denominator

    # In case we don't need the uncertainty just return None
    if not return_uncertainty:
        return class_pred, log_ps_pred, -1.0

    # Uncertainty prediction has the same two cases as probability
    # prediction. By default, the numerator is given by p*(1-p)
    # For convenience, (1-p) is denoted with _1mp
    if log_ps_total_cur[idx_max] > log_prior_default:
        log_numerator_1mp = logsumexp(
            np.r_[
                log_ps_cur[:idx_max],
                log_ps_cur[idx_max + 1 :],
                log_pN + np.log1p(-log_prior[idx_max]),
            ]
        )
    else:
        log_numerator_1mp = logsumexp(
            np.r_[
                log_ps_cur,
                log_pN + np.log1p(-log_prior_default),
            ]
        )

    log_uncertainty_total = (
        log_numerator_p + log_numerator_1mp - 3 * log_denominator
    )

    return class_pred, log_ps_pred, log_uncertainty_total


@ray.remote
def predict_log_proba_batch(
    X_base: np.ndarray,
    y_base: np.ndarray,
    X_query: np.ndarray,
    idx_query: np.ndarray,
    i: int,
    batch_size: int,
    log_kernel: Callable,
    log_prior: np.ndarray,
    log_pN: float,
    log_prior_default: float,
    class_default: int,
    return_uncertainty: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs prediction pipeline on a single batch.

    Parameters
    ----------
    X_base : np.ndarray[N, dim] of floats
        Base data points (index database)
    y_base : np.ndarray[N]
        Corresponding labels of base data points
    X_query : np.ndarray[n, dim]
        Query of points to make prediction for
    idx_query : np.ndarray[n, k]
        Indices of neighbors for each `X_query` entry
    i : int
        Batch start position
    batch_size : int
        Batch size, `X_query[i: i + batch_size]` are taken
    log_kernel : Callable
        Kernel function to be called: `log_kernel(X_base, X_query)`
    log_prior : np.ndarray[n_classes]
        Prior distribution on the classes
    log_pN : float
        Prior importance hyperparameter
    log_prior_default : float
        Highest priot probability
    class_default : int
        Class with highest priot probability
    return_uncertainty : bool, optional
        Whether to compute uncertainty, by default False

    Returns
    -------
        pred : np.ndarray[int]
            Predicted class for each entry
        log_proba : np.ndarray[float]
            Predicted log probability for each entry
        log_unc : np.ndarray[float]
            Optional log uncertainty for each entry, -1. by default
    """
    predict_log_proba = partial(
        predict_log_proba_single,
        log_kernel=log_kernel,
        log_prior=log_prior,
        log_pN=log_pN,
        log_prior_default=log_prior_default,
        class_default=class_default,
        return_uncertainty=return_uncertainty,
    )
    sl = np.s_[i : i + batch_size]
    size = X_query[sl].shape[0]

    preds = np.empty(size, dtype=np.int64)
    log_probas = np.empty(size, dtype=np.float32)
    log_uncs = np.empty(size, dtype=np.float32)

    for j, (X, idx) in enumerate(zip(X_query[sl], idx_query[sl])):
        preds[j], log_probas[j], log_uncs[j] = predict_log_proba(
            X_base[idx], y_base[idx], X
        )

    return preds, log_probas, log_uncs
