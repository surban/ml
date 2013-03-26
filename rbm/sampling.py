# -*- coding: utf-8 -*-

from __future__ import division

import gnumpy as gp
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import sys

import common.stats
import util

class ColumnNotFound(Exception):
    pass

def find_column(sheet, name):
    for col in range(sh.ncols):
        if sheet.cell(0, col).value.lower() == name.lower():
            return col
    raise ColumnNotFound()

def read_stabilities_from_spreadsheet(filename):
    with xlrd.open_workbook(filename) as book:
        sh = book.sheet_by_index(0)
   
        ss = []
        for row in range(1, sh.nrows):
            s = Stability()
            for name in s.__dict__.iterkeys():
                try:
                    s.__dict__[name] = sh.cell(row, find_column(sh, name))
                except ColumnNotFound:
                    pass
            ss.append(s)
        return ss


def generation_accuracy(label, ref_predict, myrbm,
                        tmpl_X, tmpl_Z, tmpl_ref_Z,
                        gen_X, gen_Z=None,
                        output_data_line=True, store_samples=True):

    tmpl_Z = gp.as_numpy_array(tmpl_Z)
    n_samples = gen_X.shape[0]

    if n_samples < tmpl_X.shape[0]:
        print "Warning: less generated samples than template samples were provided"
        tmpl_X = tmpl_X[0:n_samples,:]
        tmpl_Z = tmpl_Z[0:n_samples]
        tmpl_ref_Z = tmpl_ref_Z[0:n_samples]

    # calculate accuracy of reference classifier
    diff = tmpl_Z - tmpl_ref_Z
    errs = np.count_nonzero(diff)
    corr = n_samples - errs
    svc_acc = corr / n_samples

    # classify generated data
    if gen_Z is None:
        gen_Z = common.util.map(gp.as_numpy_array(gen_X), 100, ref_predict,
                                caption="Classifying results with reference predictor")

    # count correctly classified samples
    diff = tmpl_Z - gen_Z
    errs = np.count_nonzero(diff)
    corr = n_samples - errs

    # find incorrect samples
    incorrect_samples = np.nonzero(diff)[0]
    guilty_samples = \
        incorrect_samples[tmpl_ref_Z[incorrect_samples] == 
                          tmpl_Z[incorrect_samples]]

    # calculate free energy
    fes = common.util.map(gen_X, 1000, myrbm.free_energy, 
                          caption="Calculating free energy")
    fes = gp.as_numpy_array(fes)
    fe_mean = np.mean(fes)
    fe_variance = common.stats.unbiased_varince(fes)

    # create stability object
    s = Stability(label, n_samples, corr, svc_acc, 
                  fe_mean, fe_variance, incorrect_samples, guilty_samples)
    if store_samples:
        s.tmpl_X = tmpl_X
        s.tmpl_Z = tmpl_Z
        s.tmpl_ref_Z = tmpl_ref_Z
        s.gen_X = gen_X
        s.gen_Z = gen_Z

    # output performance data in table format
    if output_data_line:
        s.output_data_line()

    return s


def gen_acc_from_total_acc(total_acc, p_svc_acc):
    return (9*total_acc + p_svc_acc - 1) / (10*p_svc_acc - 1)

class Stability(object):
    def __init__(self, label=None, n_samples=None, n_success=None, 
                 classifier_accuracy=None, fe_mean=None, fe_variance=None,
                 incorrect_samples=None, guilty_samples=None):
        self.label = label
        self.n_samples = n_samples
        self.n_success = n_success
        self.classifier_accuracy = classifier_accuracy
        self.fe_mean = fe_mean
        self.fe_variance = fe_variance
        self.incorrect_samples = incorrect_samples
        self.guilty_samples = guilty_samples

        self.tmpl_X = None
        self.tmpl_Z = None
        self.tmpl_ref_Z = None
        self.gen_X = None
        self.gen_Z = None

        self.tag = None

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

    def output_data_line(self):
        common.util.output_table(("label", "n_samples", "n_success",
                                  "classifier_accuracy", "fe_mean",
                                  "fe_variance"),
                                 (self.label, self.n_samples, self.n_success, 
                                  self.classifier_accuracy, self.fe_mean, 
                                  self.fe_variance))

    def output_errors(self, n_plot=100000, alpha=0.05,
                      plot_only_guilty_samples=True):

        # output statistics
        acc_low, acc_high, acc_mle = \
            self.generator_accuracy_interval(alpha=alpha)
        print "Error probability: %g [%g, %g]" % (1-acc_mle, 
                                                  1-acc_high, 1-acc_low)
        fe_low, fe_high = self.fe_interval(alpha=alpha)
        print "Free energy:       [%g, %g]" % (fe_low, fe_high)
 
        # collect incorrectly classified samples
        tmpl_X = gp.as_numpy_array(self.tmpl_X)
        gen_X = gp.as_numpy_array(self.gen_X)
        if plot_only_guilty_samples:
            s = self.guilty_samples
        else:
            s = self.incorrect_samples

        err_tmpl_X = tmpl_X[s,:]
        err_gen_X = gen_X[s,:]
        err_tmpl_Z = self.tmpl_Z[s]
        err_gen_Z = np.asarray(self.gen_Z[s], dtype='uint8')
    
        # output
        print "Misclassified samples:"
        print "True labels:      ", err_tmpl_Z[0:n_plot]
        print "Generated labels: ", err_gen_Z[0:n_plot]
        if err_tmpl_X.shape[0] > 0:
            myplt = np.concatenate((common.util.plot_samples(err_tmpl_X[0:n_plot]), 
                                    common.util.plot_samples(err_gen_X[0:n_plot])))
            plt.imshow(myplt, interpolation='none')

def plot_box(x, lower, upper, middle):
    width = 0.5
    plt.gca().add_patch(plt.Rectangle((x-width/2,lower), width, upper-lower, fill=False))
    plt.hlines(middle, x-width/2, x+width/2, 'r')


def plot_stability(stability_data, alpha=0.05):
    plt.subplot(2, 1, 1)
    for i, s in enumerate(stability_data):
        lower, upper, mle = s.generator_accuracy_interval(alpha=alpha)
        lower = max(0, lower)
        upper = min(1, upper)
        mle = max(0, min(1, mle))
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


