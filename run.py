import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from nuq import NuqClassifier

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
                     centers=np.array([[3., 0.], [3., 2.], [0., 10]]))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    plot_data(X, y)

    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    X_test = np.c_[xx.ravel(), yy.ravel()]

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 6), dpi=200, sharey=True)
    ax[0].set_title('Raw data')
    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train)

    # strategy options: 'isj', 'silverman', 'scott', 'classification'
    uncertainty_type = "aleatoric"
    for i, strategy in enumerate(['ISJ', 'Classification']):
        # strategy = "classification"
        precise_computation = True
        # nuq = NuqClassifier(bandwidth=np.array([0.4, 0.4]), strategy=strategy.lower(), tune_bandwidth=True,
        #                       precise_computation=precise_computation, n_neighbors=100, coeff=1e-10)
        nuq = NuqClassifier(bandwidth=np.array([0.4, 0.4]), strategy=strategy.lower(), tune_bandwidth=True,
                            precise_computation=precise_computation, n_neighbors=10, coeff=1e-10)
        nuq.fit(X=X_train, y=y_train)

        ax[1].set_title('Classification')
        f_hat_y_x = nuq.predict_proba(X_test)["probs"]
        ax[1].contourf(xx, yy, np.max(f_hat_y_x, axis=-1).reshape(*xx.shape))

        ax[i + 2].set_title(f'Uncertainty {uncertainty_type}, {strategy}')
        uncertainty = nuq.predict_uncertartainty(X_test)
        Ue = uncertainty[uncertainty_type]
        if precise_computation:
            ax[i + 2].contourf(xx, yy, Ue.reshape(*xx.shape))
        else:
            ax[i + 2].contourf(xx, yy, np.log(Ue.reshape(*xx.shape)))

        print(f"{strategy}, {[10., 0.]}: {nuq.predict_uncertartainty(np.array([[10., 0.]]))}, "
              f"{[0., 0.]}: {nuq.predict_uncertartainty(np.array([[0., 0.]]))},"
              f"{[20., 20.]}: {nuq.predict_uncertartainty(np.array([[20., 20.]]))}")
    plt.tight_layout()
    # plt.savefig('./pics/nuq_res_log.pdf', format='pdf')
    plt.savefig('./pics/nuq_res_log.png')
    plt.show()
