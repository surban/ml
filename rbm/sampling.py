# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np

def mixing_quality(samples):
    samples = gp.as_numpy_array(samples)
    n_steps = samples.shape[0]

    avg_dists = []
    for step in range(n_steps-1):
        v_now = samples[step, :, :]
        v_next = samples[step+1, :, :]

        dists = np.sqrt(np.sum(np.power(v_next - v_now, 2), axis=1))
        avg_dists.append(np.mean(dists))

    return np.mean(avg_dists)


