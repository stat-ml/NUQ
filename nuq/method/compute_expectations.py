import numpy as np
from scipy.special import logsumexp


def get_kernel(name="RBF", **kwargs):
    """[deprecated] Constructs kernel function with signature:
    f(x: [N, d], y: [N, d]) -> values [N]
      * x, y: arrays of vectors to compute kernel function on
      * values: -//-

    Args:
        name (str, optional): kernel type. Defaults to "RBF".

    Raises:
        ValueError: if `name` is unsupported

    Returns:
        [Callable]: kernel function
    """
    if name == "RBF":
        bandwidth = kwargs.get(
            "bandwidth",
            np.array(
                [
                    [
                        1.0,
                    ]
                ]
            ),
        )
        if len(bandwidth.shape) < 2:
            bandwidth = bandwidth[None]
        if not kwargs.get("precise_computation"):
            return lambda x, y: np.exp(
                -np.sum(np.square(x - y) / bandwidth ** 2, axis=-1)
            )
        else:
            return lambda x, y: -np.sum(
                np.square(x - y) / bandwidth ** 2, axis=-1
            )
    else:
        raise ValueError(f"Unsupported kernel: {name}")


def get_log_kernel(name="RBF", *args, **kwargs):
    """Constructs kernel function with signature:
    f(d: [N]) -> values [N]
      * d: array of so-called distances (euclidean, l2, cosine, ...)
           to be used as kernel arguments
      * values: -//-

    Args:
        name (str, optional): kernel type. Defaults to "RBF".

    Raises:
        ValueError: if `name` is unsupported

    Returns:
        [Callable]: kernel function
    """
    if name == "RBF":
        bandwidth = kwargs.get(
            "bandwidth",
            np.array(
                [
                    [
                        1.0,
                    ]
                ]
            ),
        )
        if len(bandwidth.shape) < 2:
            bandwidth = bandwidth[None]
        return lambda d: -d / bandwidth ** 2
    else:
        raise ValueError(f"Unsupported kernel: {name}")


def compute_centroids(embeddings, labels):
    """Computes centroid for each class.

    Args:
        embeddings ([type]): embeddings
        labels ([type]): labels (1..n_classes)

    Returns:
        centers ([type]): -//-
        labels ([type]): -//-
    """
    centers, lbls = [], []
    for c in np.unique(labels):
        mask = labels == c
        centers.append(np.mean(embeddings[mask], axis=0))
        lbls.append(c)
    return np.array(centers), np.array(lbls)


def compute_weights(
    knn,
    kernel,
    current_embeddings,
    train_embeddings,
    training_labels,
    n_neighbors,
    method="faiss",
):
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
    if len(current_embeddings.shape) < 2:
        current_embeddings = current_embeddings.reshape(1, -1)
    if method == "hnsw":
        indices, distances = knn.knn_query(current_embeddings, k=n_neighbors)
    elif method == "faiss":
        distances, indices = knn.search(current_embeddings, k=n_neighbors)
    else:
        raise ValueError
    selected_embeddings = train_embeddings[indices]
    selected_labels = training_labels[indices]

    # ####
    # selected_embeddings = train_embeddings
    # selected_labels = np.repeat(training_labels[None], training_labels.shape[0], axis=0)
    # # ####

    w_raw = kernel(current_embeddings[:, None, :], selected_embeddings)[
        ..., None
    ]

    assert w_raw.shape == (current_embeddings.shape[0], n_neighbors, 1)
    return w_raw, selected_labels


def compute_logsumexp(log_weights, targets, coeff, log_denomerator=None):
    log_numerator_pre = logsumexp(log_weights, axis=1, b=targets)
    broadcast_shape = (
        1,
        log_numerator_pre.shape[0],
        log_numerator_pre.shape[1],
    )
    concated_array = np.concatenate(
        [
            log_numerator_pre[None],
            np.log(coeff) * np.ones(shape=broadcast_shape),
        ],
        axis=0,
    )
    log_numerator = logsumexp(concated_array, axis=0)

    if log_denomerator is None:
        log_denomerator = logsumexp(log_weights, axis=1)
    f_hat = log_numerator - log_denomerator
    return f_hat, log_denomerator


def get_nw_mean_estimate(targets, weights, precise_computation, coeff=1.0):
    if len(weights.shape) < 2:
        weights = weights.reshape(1, -1)[..., None]
    assert weights.shape[1] == targets.shape[1]
    if not precise_computation:
        idx = np.argwhere(np.sum(weights, axis=1) > 0.0)[:, 0]
        f_hat = np.zeros((weights.shape[0], targets.shape[-1]))
        # print("NW_MEAN_ESTIMATE:")
        # print(f"{weights.shape = }")
        # print(weights)
        # print(f"{targets.shape = }")
        # print(targets)
        f_hat[idx] = (np.sum(weights[idx] * targets[idx], axis=1) + coeff) / (
            np.sum(weights[idx], axis=1) + 2.0 * coeff
        )
        f1_hat = 1.0 - f_hat
        assert f_hat.shape == (weights.shape[0], targets.shape[-1])
    else:
        log_weights = weights
        f_hat, log_denomerator = compute_logsumexp(
            log_weights=log_weights, targets=targets, coeff=coeff
        )
        f1_hat, _ = compute_logsumexp(
            log_weights=log_weights,
            targets=1 - targets,
            coeff=coeff,
            log_denomerator=log_denomerator,
        )

        assert f_hat.shape == (log_weights.shape[0], targets.shape[-1])
        assert f1_hat.shape == (log_weights.shape[0], targets.shape[-1])
    return {"f_hat": f_hat, "f1_hat": f1_hat}


def p_hat_x(weights, n, h, precise_computation, dim):
    if len(weights.shape) < 2:
        weights = weights.reshape(1, -1)[..., None]
    # np.prod!! instead of np.mean
    if not precise_computation:
        f_hat_x = np.sum(weights.squeeze(-1) / (n * np.mean(h)), axis=-1)
    else:
        log_weights = weights
        dim_multiplier = dim if h.shape == () or h.shape == (1,) else 1.0
        f_hat_x = (
            -np.log(n)
            - dim_multiplier * np.sum(np.log(h))
            + logsumexp(log_weights, axis=1)
        )

    assert (
        f_hat_x.shape[0] == weights.shape[0]
    ), f"Received shapes are: f_hat shape is {f_hat_x.shape[0]} and weights shape is {weights.shape[0]}"

    return f_hat_x


def asymptotic_var(sigma_est, f_est, bandwidth, n):
    if len(f_est.shape) < 2:
        f_est = f_est[..., None]
    # np.prod!! instead of np.mean
    return (sigma_est * np.sqrt(np.pi)) / (
        np.mean(bandwidth) * f_est * n + 1e-20
    )


def log_asymptotic_var(log_sigma_est, log_f_est, bandwidth, n, dim):
    if len(log_f_est.shape) < 2:
        log_f_est = log_f_est[..., None]
    dim_multiplier = (
        dim if bandwidth.shape == () or bandwidth.shape == (1,) else 1.0
    )
    log_numerator = log_sigma_est + 0.5 * np.log(np.pi)
    log_denominator = (
        np.log(n) + dim_multiplier * np.sum(np.log(bandwidth)) + log_f_est
    )
    broadcast_shape = (1, log_denominator.shape[0], log_denominator.shape[1])
    log_denominator_safe = logsumexp(
        np.concatenate(
            [log_denominator[None], -40 * np.ones(broadcast_shape)], axis=0
        ),
        axis=0,
    )
    return log_numerator - log_denominator_safe


def half_gaussian_mean(asymptotic_var):
    return 2 * np.sqrt(asymptotic_var) * np.sqrt(2) / np.sqrt(np.pi)


def log_half_gaussian_mean(asymptotic_var):
    return 1.5 * np.log(2.0) - 0.5 * np.log(np.pi) + 0.5 * asymptotic_var


def half_gaussian_var(asymptotic_var):
    return 4 * asymptotic_var * (1.0 - 2.0 / np.pi)
