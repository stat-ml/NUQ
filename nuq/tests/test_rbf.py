import nuq
import numpy as np

def test_small_data():
    x1 = np.array([1, 0])
    x = np.array([0, 0])
    d = x.shape[0]
    n = 1
    h = 4

    method = 'all_data' # all_data
    kernel = 'RBF'
    use_uniform_prior = False
    nuq_tr = nuq.NuqClassifier(
        kernel_type=kernel,
        use_uniform_prior=use_uniform_prior,
        method=method,
        tune_bandwidth=False,
        bandwidth=np.array([h, h])
    )

    nuq_tr.fit(x1.reshape(1, -1), np.array([0]))
    kernel_output = nuq_tr.kernel(x1, x)

    assert np.allclose(kernel_output, -d * np.log(np.sqrt(2 * np.pi)) - np.linalg.norm(x1 - x) ** 2 / (2 * h ** 2)), \
            "kernel output should be -d * log(sqrt{2pi}) - \frac{||x_1 - x_2||^2_2}{2h^2}"

    kde = nuq_tr.get_kde(x)
    assert np.allclose(kde, - np.log(n * h ** d) + kernel_output), \
            "kernel output in point x should be -log(nh^d) + log K(x1, x2)"

    kde = nuq_tr.get_kde(x1)
    assert np.allclose(kde, - np.log(n * h ** d) - np.log(2 * np.pi)), \
            "kernel output in point x1 should be -log(nh^d) - log 2pi"
