from __future__ import division

import numpy as np
import math


def generate_random_data(x_len, n_samples, binary=False):
    data = 4.0 * (np.random.random(size=(x_len, n_samples)) - 0.5)
    if binary:
        data = data >= np.zeros_like(data)
    return np.float32(data)


def shifted(x, s):
    assert x.ndim == 1 and s.ndim == 1, "x and s must be vectors"
    assert len(s) <= len(x), "x must be at least as long as s"
    spos = np.nonzero(s)[0]
    assert len(spos) == 1, "exactly one entry of s must be one"
    assert s[spos[0]] == 1, "entries of s must be either 0 or 1"
    shift = spos[0]
    y = np.roll(x, shift)
    return y


def generate_data(x_len, s_len, n_samples, binary=False):
    # TODO: optimize or move to C code
    assert s_len <= x_len
    inputs = np.zeros((x_len, n_samples))
    shifts = np.zeros((s_len, n_samples))
    targets = np.zeros((x_len, n_samples))

    for s in range(n_samples):
        inputs[:,s] = generate_random_data(x_len, 1, binary=binary)[:,0]
        if np.sum(inputs[:,s]) == 0:
            inputs[0,s] = 1
        shft = np.random.randint(0, s_len)
        shifts[shft,s] = 1
        targets[:,s] = shifted(inputs[:,s], shifts[:,s])

    return np.float32(inputs), np.float32(shifts), np.float32(targets)


def generate_id_data(x_len, n_samples, binary=False):
    data = generate_random_data(x_len, n_samples, binary=binary)
    return data, data


