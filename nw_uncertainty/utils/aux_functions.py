import numpy as np


def logsumexp_with_coeff(array1, array2, broadcast_shape, mask=None):
    # broadcast_shape = (1, log_numerator_pre.shape[0], log_numerator_pre.shape[1])
    concated_array = np.concatenate([array1[None], array2 * np.ones(shape=broadcast_shape)],
                                    axis=0)
    res = safe_logsumexp(concated_array, b=mask, axis=0)
    return res


def get_logsumexps(log_weights, targets):
    full = safe_logsumexp(log_weights, axis=1)
    in_class = safe_logsumexp(log_weights, axis=1, b=targets)
    out_class = safe_logsumexp(log_weights, axis=1, b=1 - targets)
    return in_class, out_class, full


def compute_logsumexp_ratio(log_weights, targets, log_denomerator=None, coeff1=None, coeff2=None, mask=None):
    log_numerator_pre = safe_logsumexp(log_weights, axis=1, b=targets)
    if coeff1 is not None:
        broadcast_shape = (1, log_numerator_pre.shape[0], log_numerator_pre.shape[1])
        log_numerator = logsumexp_with_coeff(array1=log_numerator_pre, array2=coeff1, broadcast_shape=broadcast_shape,
                                             mask=mask)
    else:
        log_numerator = log_numerator_pre

    if log_denomerator is None:
        if coeff2 is not None:
            log_denomerator_pre = safe_logsumexp(log_weights, axis=1)
            broadcast_shape = (1, log_denomerator_pre.shape[0], log_denomerator_pre.shape[1])
            log_denomerator = \
                logsumexp_with_coeff(array1=log_denomerator_pre, array2=coeff2, broadcast_shape=broadcast_shape,
                                     mask=mask)
        else:
            log_denomerator = safe_logsumexp(log_weights, axis=1)
    f_hat = log_numerator - log_denomerator
    return f_hat, log_denomerator


def safe_logsumexp(a, axis=None, b=None, keepdims=False):
    a_max = np.amax(a, axis=axis, keepdims=True)
    if b is not None:
        # a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            # a[b == 0] = -np.inf

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    # out[out == -np.inf] = -10000.
    # out = np.clip(out, a_min=-500., a_max=None)
    return out
