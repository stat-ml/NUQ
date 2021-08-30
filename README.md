# NUQ uncertainty

It's an implementation of uncertainty estimation kernel method based on Nadaraya-Watson kernel regression (see 'NUQ: Nonparametric Uncertainty Quantification for Deterministic Neural Networks' by N. Kotelevskii et al)


```python
from nuq import NuqClassifier

nuq = NuqClassifier(
    strategy='isj', tune_bandwidth=True, n_neighbors=100
)
nuq.fit(X_train, y_train)
uncertainty = nuq.predict_uncertartainty(X_test)['total']
```
(see example.py for a full example)






