import numpy as np
from sklearn.model_selection import train_test_split

from nuq import NuqClassifier


def test_uniform():
    # Number of points
    N = 500
    # Number of classes
    k = 3
    # Dimensonality
    d = 2

    X = np.random.uniform(size=(N, d))
    y = np.random.choice(np.arange(k), N)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    nuq = NuqClassifier(sparse=False)
    nuq.fit(X_train, y_train)

    probs, unc = nuq.predict_proba(X_val, return_uncertainty=True)
    all_finite = np.all(np.isfinite(probs)) and np.all(np.isfinite(unc))
    assert all_finite, "All entries must be finite"
