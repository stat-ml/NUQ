import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from pytorchcv.model_provider import get_model

from nw_uncertainty import NewNW

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 25})


def plot_data(X, y):
    plt.close()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.tight_layout()
    plt.show()


def make_data(total_size=5000, centers=np.array([[-4., -4.], [0., 4.]])):
    X, y = make_blobs(n_samples=total_size, n_features=2, centers=centers)
    return X, y


if __name__ == '__main__':
    ## Make a dataset
    X, y = make_data(total_size=10000,
                     centers=np.array([[2., 0.], [0., 2.], [-7., 0.]]))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    plot_data(X, y)
    # strategy options: 'isj', 'silverman', 'scott', 'classification'
    # strategy = "classification"
    # nw_classifier = NewNW(bandwidth=np.array([1., 1.]), strategy=strategy.lower(), tune_bandwidth=False,
    #                       precise_computation=False, n_neighbors=50, use_centroids=False)
    # nw_classifier.fit(X=X_train, y=y_train)
    #
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    X_test = np.c_[xx.ravel(), yy.ravel()]
    #
    # f_hat_y_x = nw_classifier.predict_proba(X_test)["probs"]

    # plt.figure(figsize=(10, 7), dpi=200)
    # plt.title('Nadaraya-Watson classification')
    # plt.contourf(xx, yy, np.max(f_hat_y_x, axis=-1).reshape(*xx.shape))
    # plt.show()
    #
    # Ue = nw_classifier.predict_uncertartainty(X_test)
    # plt.figure(figsize=(10, 7), dpi=200)
    # plt.title('Nadaraya-Watson Uncertainties')
    # plt.contourf(xx, yy, Ue.reshape(*xx.shape))
    # plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 6), dpi=200, sharey=True)
    # plt.suptitle(f"{strategy} strategy for bandwidth selection")
    ax[0].set_title('Raw data')
    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train)

    # strategy options: 'isj', 'silverman', 'scott', 'classification'
    # uncertainty_type = "epistemic"
    strategy = "isj"
    nw_classifier = NewNW(bandwidth=np.array([0.04, 0.04]), strategy=strategy.lower(), tune_bandwidth=True,
                          n_neighbors=50, coeff=1e-5)
    nw_classifier.fit(X=X_train, y=y_train)
    output = nw_classifier.predict_proba(X=X_test, return_uncertainties=True)
    ax[1].set_title('Classification')
    ax[1].contourf(xx, yy, np.exp(np.max(output["probs"], axis=-1).reshape(*xx.shape)))
    for i, uncertainty_type in enumerate(['epistemic', 'aleatoric', 'total']):
        ax[i + 2].set_title(f'Uncertainty {uncertainty_type}, {strategy}')
        Ue = output[uncertainty_type]
        ax[i + 2].contourf(xx, yy, Ue.reshape(*xx.shape))

        print(
            f"{strategy}, {[20., 0.]}: {nw_classifier.predict_proba(np.array([[20., 0.]]), return_uncertainties=True)['probs']}, "
            f"{[0., 20.]}: {nw_classifier.predict_proba(np.array([[0., 20.]]), return_uncertainties=True)['probs']},"
            f"{[20., 20.]}: {nw_classifier.predict_proba(np.array([[20., 20.]]), return_uncertainties=True)['probs']}")
    plt.tight_layout()
    plt.savefig('./pics/nw_res_log.pdf', format='pdf')
    plt.show()
