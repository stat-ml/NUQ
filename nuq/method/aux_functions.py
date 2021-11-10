import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors


class MyKNN():
    def __init__(self, X):
        self.knn = NearestNeighbors(n_neighbors=X.shape[0])
        self.knn.fit(X)

    def knn_query(self, X, k=None):
        distances, indices = self.knn.kneighbors(X=X)
        return indices, distances


def plot_data(X, y):
    plt.close()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.tight_layout()
    plt.show()


def make_data(total_size=5000, centers=np.array([[-4., -4.], [0., 4.]])):
    X, y = make_blobs(n_samples=total_size, n_features=2, centers=centers)
    return X, y


def get_kernel(name='RBF', **kwargs):
    multipliers = {
        'RBF': 1 / (2 * np.sqrt(np.pi)),
        'logistic': 1 / 6.,
        'sigmoid': 2 / np.pi ** 2
    }
    bandwidth = kwargs.get('bandwidth', np.array([[1., ]]))
    if name == 'RBF':
        if len(bandwidth.shape) < 2:
            bandwidth = bandwidth[None]

        return multipliers[name], lambda x, y: -np.sum(np.square((x - y) / bandwidth) / 2., axis=-1) - 0.5 * \
                                                   x.shape[-1] * np.log(
                2 * np.pi)
    if name == 'logistic':
        def log_logistic_kernel(x, y):
            diff = x - y
            output = -np.sum(
                logsumexp(np.concatenate(
                    [(diff / bandwidth)[None], (-diff / bandwidth)[None],
                     np.log(2) * np.ones((1, x.shape[0], y.shape[1], x.shape[-1]))],
                    axis=0), axis=0), axis=-1)
            return output

        return multipliers[name], log_logistic_kernel
    if name == 'sigmoid':
        def log_sigmoid_function(x, y):
            diff = x - y
            output = x.shape[-1] * np.log(2 / np.pi) - np.sum(
                logsumexp(np.concatenate([(diff / bandwidth)[None],
                                          (-diff / bandwidth)[None]], axis=0),
                          axis=0),
                axis=-1)
            return output
        return multipliers[name], log_sigmoid_function
    else:
        raise ValueError("Wrong kernel name")


def compute_centroids(embeddings, labels):
    centers, lbls = [], []
    for c in np.unique(labels):
        mask = labels == c
        centers.append(np.mean(embeddings[mask], axis=0))
        lbls.append(c)
    return np.array(centers), np.array(lbls)


def compute_weights(knn, kernel, current_embeddings, train_embeddings, training_labels, n_neighbors, method='hnsw'):
    # expects valid number of nearest neighbors (0 < ... <= train_embeddings.shape[0])
    if len(current_embeddings.shape) < 2:
        current_embeddings = current_embeddings.reshape(1, -1)
    if method == 'hnsw' or method == 'all_data':
        indices, distances = knn.knn_query(current_embeddings, k=n_neighbors)
    elif method == 'faiss':
        distances, indices = knn.search(current_embeddings, k=n_neighbors)
    else:
        raise ValueError
    selected_embeddings = train_embeddings[indices]
    selected_labels = training_labels[indices]

    w_raw = kernel(current_embeddings[:, None, :], selected_embeddings)[..., None]

    assert w_raw.shape == (current_embeddings.shape[0], n_neighbors, 1)
    return w_raw, selected_labels

# maybe divide this function in two? and rename it
def compute_logsumexp(log_weights, targets, coeff, n_classes, log_denomerator=None, use_uniform_prior=False):
    log_numerator_pre = logsumexp(log_weights, axis=1, b=targets)
    if use_uniform_prior:
        broadcast_shape = (1, log_numerator_pre.shape[0], log_numerator_pre.shape[1])
        concated_array = np.concatenate([log_numerator_pre[None], np.log(coeff) * np.ones(shape=broadcast_shape)],
                                        axis=0)
        log_numerator = logsumexp(concated_array, axis=0)
    else:
        log_numerator = log_numerator_pre

    if log_denomerator is None:
        log_denomerator_pre = logsumexp(log_weights, axis=1)
        if use_uniform_prior:
            broadcast_shape = (1, log_denomerator_pre.shape[0], log_denomerator_pre.shape[1])
            concated_array = np.concatenate(
                [log_denomerator_pre[None], np.log(n_classes * coeff) * np.ones(shape=broadcast_shape)],
                axis=0)
            log_denomerator = logsumexp(concated_array, axis=0)
        else:
            log_denomerator = log_denomerator_pre
    f_hat = log_numerator - log_denomerator
    return f_hat, log_denomerator


def get_nw_mean_estimate(targets, weights, n_clasees, use_uniform_prior, coeff=1.):
    if len(weights.shape) < 2:
        weights = weights.reshape(1, -1)[..., None]
    assert weights.shape[1] == targets.shape[1]

    log_weights = weights

    f_hat, log_denomerator = compute_logsumexp(log_weights=log_weights, targets=targets, n_classes=n_clasees,
                                               coeff=coeff, use_uniform_prior=use_uniform_prior)
    f1_hat, _ = compute_logsumexp(log_weights=log_weights, targets=1 - targets, coeff=(n_clasees - 1.) * coeff,
                                  n_classes=n_clasees,
                                  log_denomerator=log_denomerator, use_uniform_prior=use_uniform_prior)

    assert f_hat.shape == (log_weights.shape[0], targets.shape[-1])
    assert f1_hat.shape == (log_weights.shape[0], targets.shape[-1])
    return {
        "f_hat": f_hat,
        "f1_hat": f1_hat
    }

def get_nw_mean_estimate_regerssion(targets, weights):
    '''
    :return: f_hat = log(probability), f1_hat = log(1 - probability)
    '''
    if len(weights.shape) < 2:
        weights = weights.reshape(1, -1)[..., None]
    assert weights.shape[1] == targets.shape[1]

    """
    kernel is logarithmed yet - yet??
    """
    log_weights = weights
    max_weights = np.max(log_weights).reshape(-1, 1)

    denominator = np.sum(np.exp(log_weights - max_weights), axis=1)
    numerator_lin = np.sum(np.exp(log_weights - max_weights) * targets, axis=1)
    numerator_sq = np.sum(np.exp(log_weights - max_weights) * (targets ** 2), axis=1)

    f_hat = np.zeros(numerator_lin.shape)
    f_sq_hat = np.zeros(numerator_sq.shape)

    non_zero_indices = (denominator > 0)

    f_hat[non_zero_indices] = numerator_lin[non_zero_indices] / denominator[non_zero_indices]
    f_sq_hat[non_zero_indices] = numerator_sq[non_zero_indices] / denominator[non_zero_indices]

    return {
        "f_hat": f_hat,
        "f_sq_hat": f_sq_hat
    }




def p_hat_x(weights, n, h, dim):
    if len(weights.shape) < 2:
        weights = weights.reshape(1, -1)[..., None]
    # np.prod!! instead of np.mean
    # make this neat? also why comment above?
    dim_multiplier = dim if h.shape == () or h.shape == (1,) else 1.
    f_hat_x = -np.log(n) - dim_multiplier * np.sum(np.log(h)) + logsumexp(weights, axis=1)

    return f_hat_x


def asymptotic_var(sigma_est, f_est, bandwidth, n):
    if len(f_est.shape) < 2:
        f_est = f_est[..., None]
    # np.prod!! instead of np.mean
    return (sigma_est * np.sqrt(np.pi)) / (np.prod(bandwidth) * f_est * n + 1e-20)


def log_asymptotic_var(log_sigma_est, log_f_est, bandwidth, n, dim, squared_kernel_int):
    if len(log_f_est.shape) < 2:
        log_f_est = log_f_est[..., None]
    dim_multiplier = dim if bandwidth.shape == () or bandwidth.shape == (1,) else 1.
    log_numerator = log_sigma_est + dim * squared_kernel_int
    log_denominator = np.log(n) + dim_multiplier * np.sum(np.log(bandwidth)) + log_f_est
    # broadcast_shape = (1, log_denominator.shape[0], log_denominator.shape[1])
    # log_denominator_safe = logsumexp(np.concatenate([log_denominator[None], -40 * np.ones(broadcast_shape)], axis=0),
    #                                  axis=0)
    return log_numerator - log_denominator  # - log_denominator_safe


def half_gaussian_mean(asymptotic_var):
    return 2 * np.sqrt(asymptotic_var) * np.sqrt(2) / np.sqrt(np.pi)


def log_half_gaussian_mean(asymptotic_var):
    return 1.5 * np.log(2.) - 0.5 * np.log(np.pi) + 0.5 * asymptotic_var


def half_gaussian_var(asymptotic_var):
    return 4 * asymptotic_var * (1. - 2. / np.pi)
