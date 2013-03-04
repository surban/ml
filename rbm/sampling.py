# -*- coding: utf-8 -*-

from __future__ import division

import gnumpy as gp
import numpy as np
import matplotlib.pyplot as plt

import common.stats
import util

class Stability(object):
    def __init__(self, label, n_samples, n_success, classifier_accuracy,
                 fe_mean, fe_variance):
        self.label = label
        self.n_samples = n_samples
        self.n_success = n_success
        self.classifier_accuracy = classifier_accuracy
        self.fe_mean = fe_mean
        self.fe_variance = fe_variance

    def generator_accuracy_interval(self, alpha=0.05):
        low, high = common.stats.binomial_p_confint(self.n_success, self.n_samples,
                                                    alpha=alpha)
        gen_low = util.gen_acc_from_total_acc(low, self.classifier_accuracy)
        gen_mle = util.gen_acc_from_total_acc(self.n_success / self.n_samples,
                                              self.classifier_accuracy)
        gen_high = util.gen_acc_from_total_acc(high, self.classifier_accuracy)
        return gen_low, gen_high, gen_mle

    def fe_interval(self, alpha=0.05):
        return common.stats.normal_mean_confint_ss(self.n_samples, self.fe_mean,
                                                   self.fe_variance, 
                                                   alpha=alpha)


def plot_box(x, lower, upper, middle):
    width = 0.2
    plt.gca().add_patch(plt.Rectangle((x-width/2,lower), width, upper-lower, fill=False))
    plt.hlines(middle, x-width/2, x+width/2, 'r')


def plot_stability(stability_data, alpha=0.05):
    plt.subplot(2, 1, 1)
    for i, s in enumerate(stability_data):
        lower, upper, mle = s.generator_accuracy_interval(alpha=alpha)
        lower = max(0, lower)
        upper = min(1, upper)
        plot_box(i+1, lower, upper, mle)
    plt.xlim(0, len(stability_data)+1)
    plt.xticks(range(1, len(stability_data)+1), ["" for s in stability_data])
    plt.ylabel("accuracy")

    plt.subplot(2, 1, 2)
    for i, s in enumerate(stability_data):
        lower, upper = s.fe_interval(alpha=alpha)
        mle = (lower + upper) / 2
        plot_box(i+1, lower, upper, mle)
    plt.xlim(0, len(stability_data)+1)
    plt.xticks(range(1, len(stability_data)+1), [s.label for s in stability_data])
    plt.ylabel("free energy")



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


