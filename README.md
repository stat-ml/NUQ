# NUQ Uncertainty

This repository implements an uncertainty estimation kernel method based on Nadaraya-Watson kernel regression (see *NUQ: Nonparametric Uncertainty Quantification for Deterministic Neural Networks* by N. Kotelevskii et al.)

## Environment

**Important:** we recommend having [`Jupyter Lab`](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) installed in the `base` conda environment. For the best experience, you may also install [`nb_conda_kernels`](https://github.com/Anaconda-Platform/nb_conda_kernels) and [`ipywidgets`](https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-in-jupyterlab-3-0).

1. Create conda environment:
   ```bash
   $ conda env create --file environment.yaml
   ```
2. Activate it:
   ```bash
   $ conda activate ray-env
   ```

## How to use
See [classification](examples/classification.ipynb) and [regression](examples/regression.ipynb) examples for more details.

### Classification
```python
from nuq import NuqClassifier

nuq = NuqClassifier()
nuq.fit(X_train, y_train)
preds, log_uncs = nuq.predict(X_test, return_uncertainty="epistemic")
```

### Regression
```python
from nuq import NuqRegressor

nuq = NuqRegressor()
nuq.fit(X_train, y_train)
preds, log_uncs = nuq.predict(X_test, return_uncertainty="epistemic")
```

## Installation
(temporary local only)
Pip doesn't handle .toml files correctly, thus using `setup.py`:
```bash
python setup.py develop
```

To uninstall:
```bash
pip uninstall nuq
```

## Development

1. Install [`pre-commit`](https://pre-commit.com/#3-install-the-git-hook-scripts) (config provided in this repo)
   ```bash
   $ pre-commit install
   ```
2. (optional) Run against all the files to check the consistency
   ```bash
   $ pre-commit run --all-files
   ```
3. You may also run [`black`](https://github.com/psf/black) and [`isort`](https://github.com/PyCQA/isort) to keep the files style-compliant
   ```bash
   $ isort .; black .
   ```
4. Proposed linter is [`flake8`](https://flake8.pycqa.org/en/latest/)
   ```bash
   $ flake8 .
   ```
5. One can run tests via [`pytest`](https://pytest.org/)
   ```bash
   $ pytest
   ```
