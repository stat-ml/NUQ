import numpy as np
from sklearn.model_selection import train_test_split

from nuq import NuqClassifier


def test_overfitting():
    # Number of points
    N = 500
    # Number of classes
    k = 3
    # Dimensonality
    d = 2

    X = np.random.uniform(size=(N, d))
    y = np.random.choice(np.arange(k), N)

    nuq = NuqClassifier()
    nuq.fit(X, y)
    score = nuq.score(X, y)

    assert score > 0.95, "Overfitting failed"
