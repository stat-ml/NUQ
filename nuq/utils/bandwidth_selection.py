import numpy as np
from KDEpy.bw_selection import improved_sheather_jones, silvermans_rule, scotts_rule
from sklearn.model_selection import GridSearchCV


def to_multidim(X, method):
    # Compute the BW, scale, then scale back
    bws = []
    for i in range(X.shape[1]):
        bw = method(X[:, i][..., None])
        bws.append(bw)
    bandwidth = np.array(bws)[None]
    return bandwidth


def classification_selection(X, y, knn, constructor, precise_computation, n_neighbors, use_centroids, kernel_type,
                             use_uniform_prior, mean_distances):
    classificator = constructor(tune_bandwidth=False, precise_computation=precise_computation, n_neighbors=n_neighbors,
                                use_centroids=use_centroids, kernel_type=kernel_type,
                                use_uniform_prior=use_uniform_prior)
    gs = GridSearchCV(classificator,
                      param_grid={
                          'bandwidth': [np.array(i) for i in mean_distances][5::5]
                      }, scoring='accuracy', cv=3)
    gs.fit(X, y)
    print(f"mean distance = {mean_distances}")
    print(f"{gs.best_params_['bandwidth']}")
    print('Best accuracy ', gs.best_score_)
    return np.array([gs.best_params_['bandwidth']])


def std_deviation_selection(X, y):
    stds = []
    for c in np.unique(y):
        # batches = [embeddings[i:i + batch_size] for i in range(0, len(embeddings), batch_size)]
        # mean = covs = 0
        # for batch in batches:
        mask = y == c
        stds.append(np.std(X[mask], axis=0))
    return np.array(stds)


def tune_kernel(X, y, strategy, constructor=None, knn=None, precise_computation=None, n_neighbors=None,
                use_centroids=None, kernel_type=None,
                use_uniform_prior=None, mean_distances=None):
    if strategy == 'isj':
        bandwidth = np.sqrt(X.shape[1]) * to_multidim(X=X, method=improved_sheather_jones)
    elif strategy == 'silverman':
        bandwidth = np.sqrt(X.shape[1]) * to_multidim(X=X, method=silvermans_rule)
    elif strategy == 'std':
        bandwidth = np.sqrt(X.shape[1]) * np.mean(std_deviation_selection(X=X, y=y), axis=0)
    elif strategy == 'scott':
        bandwidth = np.sqrt(X.shape[1]) * to_multidim(X=X, method=scotts_rule)
    elif strategy == 'classification':
        bandwidth = classification_selection(X=X, y=y, knn=knn, constructor=constructor,
                                             precise_computation=precise_computation, n_neighbors=n_neighbors,
                                             use_centroids=use_centroids, kernel_type=kernel_type,
                                             use_uniform_prior=use_uniform_prior, mean_distances=mean_distances)
    else:
        raise ValueError("No such strategy")
    return bandwidth
