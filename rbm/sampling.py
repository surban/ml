# -*- coding: utf-8 -*-

from __future__ import division

import gnumpy as gp
import numpy as np
import matplotlib.pyplot as plt

import common.stats
import util

def generation_accuracy(label, svc, p_svc_acc, tmpl_X, gen_X, tmpl_Z, 
                        exclude_incorrectly_classified_tmpl=True,
                        alpha=0.02, gen_Z=None):

    # classify generated data
    if gen_Z is None:
        gen_Z = common.util.map(gp.as_numpy_array(gen_X), 100, svc.predict,
                                caption="Classifying results with SVM")
    tmpl_Z = gp.as_numpy_array(tmpl_Z)
    n_samples = tmpl_X.shape[0]

    # count correctly classified samples
    diff = tmpl_Z - gen_Z
    errs = np.count_nonzero(diff)
    corr = n_samples - errs

    # create stability object
    s = Stability(label, n_samples, corr, p_svc_acc, 0, 0)
    acc_low, acc_high, acc_mle = s.generator_accuracy_interval(alpha=alpha)
    print "Generator error probability: [%g, %g]" % (1-acc_high, 1-acc_low)

    # collect incorrectly classified samples
    err_tmpl_X = []
    err_gen_X = []
    err_tmpl_Z = []
    err_gen_Z = []
    err_tmpl_CZ = []
    for n, k in enumerate(np.nonzero(diff)[0]):
        if n % 100 == 0:
            common.progress.status(n, errs, "Collecting misclassified samples")
        cz = svc.predict(gp.as_numpy_array(tmpl_X[k, :]))[0]
        if not exclude_incorrectly_classified_tmpl or tmpl_Z[k] == cz:
            err_tmpl_X.append(tmpl_X[k, :])
            err_gen_X.append(gen_X[k, :])
            err_tmpl_Z.append(tmpl_Z[k])
            err_gen_Z.append(int(gen_Z[k]))
            err_tmpl_CZ.append(cz)

    return (s,
            np.asarray(err_tmpl_X), np.asarray(err_gen_X), 
            np.asarray(err_tmpl_Z), np.asarray(err_gen_Z), 
            np.asarray(err_tmpl_CZ), gen_Z)


def gen_acc_from_total_acc(total_acc, p_svc_acc):
    return (9*total_acc + p_svc_acc - 1) / (10*p_svc_acc - 1)

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
        gen_low = gen_acc_from_total_acc(low, self.classifier_accuracy)
        gen_mle = gen_acc_from_total_acc(self.n_success / self.n_samples,
                                         self.classifier_accuracy)
        gen_high = gen_acc_from_total_acc(high, self.classifier_accuracy)
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


