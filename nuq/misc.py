import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def plot_data(X, y, title=None):
    plt.close()
    plt.figure()
    if title:
        plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.tight_layout()
    plt.show()


def make_data(total_size=5000, centers=np.array([[-4.0, -4.0], [0.0, 4.0]])):
    X, y = make_blobs(n_samples=total_size, n_features=2, centers=centers)
    return X, y


def parse_param(param_string):
    res = param_string.split(":", maxsplit=1)
    if len(res) == 1:
        return res[0], {}
    name, vals_string = res
    vals_strings = filter(None, vals_string.split(";"))
    vals = [v.split("=", maxsplit=1) for v in vals_strings]
    return name, dict(vals)
