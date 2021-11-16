import numpy as np
from sklearn.model_selection import train_test_split

from nuq import NuqClassifier, NuqRegressor


def test_finite_classifier():
    # Number of points
    N = 500
    # Number of classes
    k = 3
    # Dimensionality
    d = 2

    X = np.random.uniform(size=(N, d))
    y = np.random.choice(np.arange(k), N)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    nuq = NuqClassifier(sparse=False)
    nuq.fit(X_train, y_train)

    probs, log_unc = nuq.predict_proba(X_val, return_uncertainty="epistemic")
    all_finite = np.all(np.isfinite(probs)) and np.all(np.isfinite(log_unc))
    assert all_finite, "All entries must be finite"


def test_finite_regressor():
    # Number of points
    N = 500
    # Dimensionality
    d = 50

    X = np.random.uniform(size=(N, d))
    y = np.random.uniform(size=N)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    nuq = NuqRegressor()
    nuq.fit(X_train, y_train)

    y_pr, log_unc = nuq.predict(X_val, return_uncertainty="epistemic")
    all_finite = np.all(np.isfinite(y_pr)) and np.all(np.isfinite(log_unc))
    assert all_finite, "All entries must be finite"
