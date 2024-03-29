import numpy as np

from nuq import NuqClassifier, NuqRegressor


def test_overfitting_classifier():
    np.random.seed(42)

    # Number of points
    N = 500
    # Number of classes
    k = 3
    # Dimensionality
    d = 2

    X = np.random.uniform(size=(N, d))
    y = np.random.choice(np.arange(k), N)

    nuq = NuqClassifier()
    nuq.fit(X, y)
    score = nuq.score(X, y)

    assert score > 0.9, "Overfitting failed"


def test_overfitting_regressor():
    np.random.seed(42)

    # Number of points
    N = 500
    # Dimensionality
    d = 50

    X = np.random.uniform(size=(N, d))
    y = np.random.uniform(size=N)

    nuq = NuqRegressor()
    nuq.fit(X, y)
    score = nuq.score(X, y)
    print(score)

    assert score > 0.0, "Overfitting failed"
