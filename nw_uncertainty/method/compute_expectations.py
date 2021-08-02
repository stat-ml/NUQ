import numpy as np


def get_kernel(name='RBF', **kwargs):
    if name == 'RBF':
        bandwidth = kwargs.get('bandwidth', np.array([[1., ]]))
        if len(bandwidth.shape) < 2:
            bandwidth = bandwidth[None]

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
