import numpy as np
from KDEpy.bw_selection import improved_sheather_jones, silvermans_rule, scotts_rule
from sklearn.model_selection import GridSearchCV


def to_multidim(X, method):
    # Compute the BW, scale, then scale back
    bws = []
    for i in range(X.shape[1]):
        bw = method(X[:, [i]])
        bws.append(bw)
    bandwidth = np.array(bws)[None]
    return bandwidth


def classification_selection(X, y, knn, constructor, precise_computation, n_neighbors):
    _, distances = knn.knn_query(X, k=n_neighbors)
    mean_distances = distances.mean(0)
    classificator = constructor(tune_bandwidth=False, precise_computation=precise_computation, n_neighbors=n_neighbors)
    gs = GridSearchCV(classificator,
                      param_grid={
                          'bandwidth': [np.array(i) for i in mean_distances][5::5]
                      }, scoring='accuracy', cv=3)
    gs.fit(X, y)
    print(f"mean distance = {mean_distances}")
    print(f"{gs.best_params_['bandwidth']}")
    print('Best accuracy ', gs.best_score_)
    return gs.best_params_['bandwidth']

def std_deviation_selection(X, y):
    stds = []
    for c in np.unique(y):
        # batches = [embeddings[i:i + batch_size] for i in range(0, len(embeddings), batch_size)]
        # mean = covs = 0
        # for batch in batches:
        mask = y == c
        stds.append(np.std(X[mask], axis=0))
    return np.array(stds)


def tune_kernel(X, y, knn=None, strategy="isj", constructor=None, precise_computation=True, n_neighbors=20):
    
    if strategy == 'isj':
        bandwidth = (4. * X.shape[1]) * to_multidim(X=X, method=improved_sheather_jones)

    elif strategy == 'silverman':
        bandwidth = X.shape[1] * to_multidim(X=X, method=silvermans_rule)

    elif strategy == 'std':
        bandwidth = np.mean(std_deviation_selection(X=X, y=y), axis=0)
    elif strategy == 'scott':
        bandwidth = X.shape[1] * to_multidim(X=X, method=scotts_rule)
    elif strategy == 'classification':
        bandwidth = classification_selection(X=X, y=y, knn=knn, constructor=constructor,
                                             precise_computation=precise_computation, n_neighbors=n_neighbors)
    else:
        raise ValueError("No such strategy")
    return bandwidth
