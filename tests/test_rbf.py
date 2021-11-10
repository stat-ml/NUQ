import nuq
import numpy as np
import pytest

@pytest.fixture
def two_points_setup():
    x_train = np.array([[1, 0], [0, 1]])
    y_train = np.array([0, 1])
    x_test = np.array([[0, 0], [1, 1]])

    d = 2
    n = 2
    h = 4

    knn = nuq.MyKNN(x_train)
    _, kernel = nuq.get_kernel('RBF', bandwidth=np.array([h]))

    # for model_config parameters are in the following order
        # kernel name, method, n_neighbors, coeff, tune_bandwidth, strategy, bandwidth, use_centroids, use_uniform_prior
    return {'data': (x_train, y_train, x_test), 'constants': (d, n, h), 'knn_and_kernel': (knn, kernel), \
            'model_config': ('RBF', 'all_data', 0, 0, False, 'isj', np.array([h, h]), False, False)}

def test_two_points_aux_functions(two_points_setup):
    x_train, y_train, x_test = two_points_setup['data']
    d, n, h = two_points_setup['constants']
    knn, kernel = two_points_setup['knn_and_kernel']

    # compute_weights test
    weights, labels = nuq.compute_weights(knn, kernel, x_test, x_train, y_train, n)
    assert np.allclose(weights, (- d * np.log(np.sqrt(2 * np.pi)) - 1 / 2 / (h**2)) * np.ones_like(weights))

    # tests for probabilities
    proba = nuq.get_nw_mean_estimate(y_train.reshape(1, -1), weights, 2, False)
    f_train, f1_train = proba['f_hat'], proba['f1_hat']
    assert np.allclose(np.ones_like(f_train), np.exp(f_train) + np.exp(f1_train)), 'probabilities should sum up to 1'

    proba = nuq.get_nw_mean_estimate((1 - y_train).reshape(1, -1), weights, 2, False)
    f_reversed, f1_reversed = proba['f_hat'], proba['f1_hat']
    assert np.allclose(f_train, f1_reversed), 'when switching labels, probas should be switched as well'
    assert np.allclose(f_reversed, f1_train)

def test_two_points_methods(two_points_setup):
    x_train, y_train, x_test = two_points_setup['data']
    d, n, h = two_points_setup['constants']
    knn, kernel = two_points_setup['knn_and_kernel']
    print(x_train)

    # get_kde tests
    ue_model = nuq.NuqClassifier(*two_points_setup['model_config'])
    ue_model.fit(x_train, y_train)

    kde_whole = ue_model.get_kde(x_test)
    kde_per_sample_batch = ue_model.get_kde(x_test, batch_size=1)
    assert np.allclose(kde_whole, kde_per_sample_batch)

    knn, kernel = two_points_setup['knn_and_kernel']
    weights, labels = nuq.compute_weights(knn, kernel, x_test, x_train, y_train, n)

    assert np.allclose(kde_whole, -np.log(n * h**d) + np.log(np.sum(np.exp(weights), axis=1)))
    assert True

def test_one_point():
    x_train = np.array([[1, 0]])
    y_train = np.array([0])
    x_test = np.array([[0, 0]])

    d = 2
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
        coeff=0.,
        bandwidth=np.array([h, h])
    )

    nuq_tr.fit(x_train, y_train)
    kernel_output = nuq_tr.kernel(x_train, x_test)[..., None]

    assert np.allclose(
        kernel_output, \
        -d * np.log(np.sqrt(2 * np.pi)) - np.linalg.norm(x_train - x_test) ** 2 / (2 * h ** 2)), \
        "kernel output should be -d * log(sqrt{2pi}) - \frac{||x_1 - x_2||^2_2}{2h^2}"

    kde = nuq_tr.get_kde(x_test)
    assert np.allclose(kde, - np.log(n * h ** d) + kernel_output), \
            "kde output in point x should be -log(nh^d) + log K(x1, x2)"

    kde = nuq_tr.get_kde(x_train)
    assert np.allclose(kde, - np.log(n * h ** d) - np.log(2 * np.pi)), \
            "kde output in point x1 should be -log(nh^d) - log 2pi"

    mean_0class, denominator = nuq.compute_logsumexp(kernel_output, np.array([0]), 1, 2)
    assert mean_0class == float("-inf"),  "there is no factors in numerator, so the probability is 0"
    assert denominator == kernel_output, "for one point in train denominator and kernel_output should be the same"

    mean_1class, denominator = nuq.compute_logsumexp(kernel_output, np.array([1]), 1, 2)
    assert mean_1class == 0,  "there is 1 factor in numerator and in denominator, so the probability is 1"
    assert denominator == kernel_output, "for one point in train denominator and kernel_output should be the same"

    # then everything breaks when you try y = 2, 3, etc.

    knn = nuq.MyKNN(x_train)
    weights, labels = nuq.compute_weights(knn, nuq_tr.kernel, x_test, x_train, y_train, 1)
    assert np.allclose(kernel_output, weights), "weights using kernel and compute_weights functions"

    proba = nuq.get_nw_mean_estimate(np.array([[0]]), weights, 2, False)
    proba_0class, proba_1_0class = proba["f_hat"], proba["f1_hat"]
    proba = nuq.get_nw_mean_estimate(np.array([[1]]), weights, 2, False)
    proba_1class, proba_0_1class = proba["f_hat"], proba["f1_hat"]

    assert np.allclose(mean_0class, proba_0class), "check that nw_mean_estimate == logsumexp"
    assert np.allclose(mean_1class, proba_1class), "check that nw_mean_estimate == logsumexp"

    assert np.allclose(proba_0_1class, proba_0class), "check for consistency between prediction of class and 1 - class"
    assert np.allclose(proba_1_0class, proba_1class)

    uncertainty_dict = nuq_tr.predict_uncertainty(x_test)
    assert np.allclose(uncertainty_dict["aleatoric"], \
        np.log(np.minimum(1 - np.exp(proba_0class), 1 - np.exp(proba_1_0class)))), "aleatoric uncertainty"

    epistemic_uncertainty = np.log(2 * np.sqrt(2 / np.pi)) - 0.5 * nuq_tr.get_kde(x_test)
    epistemic_uncertainty += 0.5 * np.maximum(proba_0class + proba_1_0class, proba_1class + proba_0_1class)
    epistemic_uncertainty += 1 / (2 * np.sqrt(np.pi))

    assert np.allclose(uncertainty_dict["epistemic"], epistemic_uncertainty), "epistemic uncertainty"

    # assert False
