import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.datasets import make_blobs


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
    if name == 'RBF':
        bandwidth = kwargs.get('bandwidth', np.array([[1., ]]))
        if len(bandwidth.shape) < 2:
            bandwidth = bandwidth[None]
        if not kwargs.get('precise_computation'):
            return lambda x, y: np.exp(-np.sum(np.square(x - y) / bandwidth, axis=-1))
        else:
            return lambda x, y: -np.sum(np.square(x - y) / bandwidth, axis=-1)
    else:
        raise ValueError("Wrong kernel name")


def compute_centroids(embeddings, labels):
    centers, lbls = [], []
    for c in np.unique(labels):
        mask = labels == c
        centers.append(np.mean(embeddings[mask], axis=0))
        lbls.append(c)
    return np.array(centers), np.array(lbls)


def compute_weights(knn, kernel, current_embeddings, train_embeddings, training_labels, n_neighbors, method='faiss'):
    if len(current_embeddings.shape) < 2:
        current_embeddings = current_embeddings.reshape(1, -1)
    if method == 'hnsw':
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


def compute_logsumexp(log_weights, targets, coeff, n_classes, log_denomerator=None):
    log_numerator_pre = logsumexp(log_weights, axis=1, b=targets)
    broadcast_shape = (1, log_numerator_pre.shape[0], log_numerator_pre.shape[1])
    concated_array = np.concatenate([log_numerator_pre[None], np.log(coeff) * np.ones(shape=broadcast_shape)],
                                    axis=0)
    log_numerator = logsumexp(concated_array, axis=0)

    if log_denomerator is None:
        log_denomerator_pre = logsumexp(log_weights, axis=1)
        broadcast_shape = (1, log_denomerator_pre.shape[0], log_denomerator_pre.shape[1])
        concated_array = np.concatenate(
            [log_denomerator_pre[None], np.log(n_classes * coeff) * np.ones(shape=broadcast_shape)],
            axis=0)
        log_denomerator = logsumexp(concated_array, axis=0)
    f_hat = log_numerator - log_denomerator
    return f_hat, log_denomerator


def get_nw_mean_estimate(targets, weights, precise_computation, n_clasees, coeff=1.):
    if len(weights.shape) < 2:
        weights = weights.reshape(1, -1)[..., None]
    assert weights.shape[1] == targets.shape[1]
    if not precise_computation:
        f_hat = (np.sum(weights * targets, axis=1) + coeff) / (np.sum(weights, axis=1) + n_clasees * coeff)
        f1_hat = 1. - f_hat
        assert f_hat.shape == (weights.shape[0], targets.shape[-1])
        assert f1_hat.shape == (weights.shape[0], targets.shape[-1])
    else:
        log_weights = weights
        f_hat, log_denomerator = compute_logsumexp(log_weights=log_weights, targets=targets, n_classes=n_clasees,
                                                   coeff=coeff)
        f1_hat, _ = compute_logsumexp(log_weights=log_weights, targets=1 - targets, coeff=(n_clasees -1.) * coeff,
                                      n_classes=n_clasees,
                                      log_denomerator=log_denomerator)

        assert f_hat.shape == (log_weights.shape[0], targets.shape[-1])
        assert f1_hat.shape == (log_weights.shape[0], targets.shape[-1])
    return {
        "f_hat": f_hat,
        "f1_hat": f1_hat
    }


def p_hat_x(weights, n, h, precise_computation, dim):
    if len(weights.shape) < 2:
        weights = weights.reshape(1, -1)[..., None]
    # np.prod!! instead of np.mean
    if not precise_computation:
        f_hat_x = np.sum(weights.squeeze(-1) / (n * np.prod(h)), axis=-1)
    else:
        log_weights = weights
        dim_multiplier = dim if h.shape == () or h.shape == (1,) else 1.
        f_hat_x = -np.log(n) - dim_multiplier * np.sum(np.log(h)) + logsumexp(log_weights, axis=1)

    assert (
        f_hat_x.shape[0] == weights.shape[0],
        f"Received shapes are: f_hat shape is {f_hat_x.shape[0]} and weights shape is {weights.shape[0]}"
    )
    return f_hat_x


def asymptotic_var(sigma_est, f_est, bandwidth, n):
    if len(f_est.shape) < 2:
        f_est = f_est[..., None]
    # np.prod!! instead of np.mean
    return (sigma_est * np.sqrt(np.pi)) / (np.prod(bandwidth) * f_est * n + 1e-20)


def log_asymptotic_var(log_sigma_est, log_f_est, bandwidth, n, dim):
    if len(log_f_est.shape) < 2:
        log_f_est = log_f_est[..., None]
    dim_multiplier = dim if bandwidth.shape == () or bandwidth.shape == (1,) else 1.
    log_numerator = log_sigma_est + 0.5 * np.log(np.pi)
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
