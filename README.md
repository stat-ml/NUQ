# NUQ uncertainty

It's an implementation of an uncertainty estimation kernel method based on Nadaraya-Watson kernel regression (see 'NUQ: Nonparametric Uncertainty Quantification for Deterministic Neural Networks' by N. Kotelevskii et al.)


## How to use
```python
from nuq import NuqClassifier

nuq = NuqClassifier(
    strategy='isj', tune_bandwidth=True, n_neighbors=100
)
nuq.fit(X_train, y_train)
uncertainty = nuq.predict_uncertainty(X_test)['total']
```
(see example.py for a full example)


## Installation
(temporary local only)
```sh
pip install -e <path_to_local_repo_folder>
```
