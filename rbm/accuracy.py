# -*- coding: utf-8 -*-

from __future__ import division

import gnumpy as gp
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import sys
import math

import common.stats
import util


def calc_separation_accuracy(label, ref_predict, myrbm,
                             tmpl_X, tmpl_XZ, tmpl_ref_XZ,
                             tmpl_Y, tmpl_YZ, tmpl_ref_YZ,
                             comb,
                             sep_X, sep_Y,
                             output_data_line=True, store_samples=True):

    tmpl_X = gp.as_numpy_array(tmpl_X)
    tmpl_Y = gp.as_numpy_array(tmpl_Y)
    comb = gp.as_numpy_array(comb)
    sep_X = gp.as_numpy_array(sep_X)
    sep_Y = gp.as_numpy_array(sep_Y)

    n_samples = sep_X.shape[0]

    if n_samples < tmpl_X.shape[0]:
        print "Warning: less separated samples than template samples were provided"
        tmpl_X = tmpl_X[0:n_samples]
        tmpl_XZ = tmpl_XZ[0:n_samples]
        tmpl_ref_XZ = tmpl_ref_XZ[0:n_samples]
        tmpl_Y = tmpl_Y[0:n_samples]
        tmpl_YZ = tmpl_YZ[0:n_samples]
        tmpl_ref_YZ = tmpl_ref_YZ[0:n_samples]
        comb = comb[0:n_samples]

    # calculate accuracy of reference classifier
    diff_X = tmpl_XZ - tmpl_ref_XZ
    diff_Y = tmpl_YZ - tmpl_ref_YZ
    errs = np.count_nonzero(diff_X) + np.count_nonzero(diff_Y)
    corr = 2*n_samples - errs
    svc_acc = corr / (2*n_samples)

    # classify generated data
    sep_XZ = common.util.map(sep_X, 1000, ref_predict,
                             caption="Classifying X results with reference predictor")
    sep_YZ = common.util.map(sep_Y, 1000, ref_predict,
                             caption="Classifying Y results with reference predictor")

    # count correctly classified samples
    diff_X = sep_XZ - tmpl_XZ
    diff_Y = sep_YZ - tmpl_YZ
    diff = (diff_X != 0) | (diff_Y != 0)
    errs = np.count_nonzero(diff)
    corr = n_samples - errs

    # find incorrect samples
    incorrect_samples = np.nonzero(diff)[0]
    guilty_samples = \
        incorrect_samples[(tmpl_ref_XZ[incorrect_samples] ==  tmpl_XZ[incorrect_samples]) &
                          (tmpl_ref_YZ[incorrect_samples] ==  tmpl_YZ[incorrect_samples])]

    # create accuracy object
    s = SeparationAccuracy()
    s.label = label
    s.classifier_accuracy = svc_acc
    s.n_samples = n_samples
    s.n_success = corr
    s.incorrect_samples = incorrect_samples
    s.guilty_samples = guilty_samples

    if store_samples:
        s.tmpl_X = tmpl_X
        s.tmpl_XZ = tmpl_XZ
        s.tmpl_ref_XZ = tmpl_ref_XZ
        s.tmpl_Y = tmpl_Y
        s.tmpl_YZ = tmpl_YZ
        s.tmpl_ref_YZ = tmpl_ref_YZ
        s.comb = comb
        s.sep_X = sep_X
        s.sep_XZ = sep_XZ
        s.sep_Y = sep_Y
        s.sep_YZ = sep_YZ

    # output performance data in table format
    if output_data_line:
        s.output_data_line()

    return s


def sep_acc_from_total_acc(total_acc, p_svc_acc):
    single_tot_acc = math.sqrt(total_acc)
    single_sep_acc = (9*single_total_acc + p_svc_acc - 1) / (10*p_svc_acc - 1)
    return math.pow(single_sep_acc, 2)

class SeparationAccuracy(object):
    def __init__(self):
        self.label = None
        self.classifier_accuracy = None

        self.n_samples = None
        self.n_success = None
        self.incorrect_samples = None
        self.guilty_samples = None

        self.tmpl_X = None
        self.tmpl_XZ = None
        self.tmpl_ref_XZ = None

        self.tmpl_Y = None
        self.tmpl_YZ = None
        self.tmpl_ref_YZ = None

        self.comb = None
        self.sep_X = None
        self.sep_XZ = None
        self.sep_Y = None
        self.sep_YZ = None

        self.tag = None

    def accuracy_interval(self, alpha=0.05):
        tot_low, tot_high = common.stats.binomial_p_confint(self.n_success, 
                                                            self.n_samples,
                                                            alpha=alpha)
        tot_mle = self.n_success / self.n_samples
        low = sep_acc_from_total_acc(tot_low, self.classifier_accuracy)
        mle = sep_acc_from_total_acc(tot_mle, self.classifier_accuracy)
        high = sep_acc_from_total_acc(tot_high, self.classifier_accuracy)
        return low, high, mle

    def output_data_line(self):
        common.util.output_table(("label", "n_samples", "n_success",
                                  "classifier_accuracy"),
                                 (self.label, self.n_samples, self.n_success, 
                                  self.classifier_accuracy))

    def output_errors(self, n_plot=100000, alpha=0.05,
                      plot_only_guilty_samples=True):

        # output statistics
        acc_low, acc_high, acc_mle = \
            self.accuracy_interval(alpha=alpha)
        print "Error probability: %g [%g, %g]" % (1-acc_mle, 
                                                  1-acc_high, 1-acc_low)
 
        if self.comb is not None:
            # collect incorrectly classified samples
            if plot_only_guilty_samples:
                s = self.guilty_samples
            else:
                s = self.incorrect_samples

            err_tmpl_X = self.tmpl_X[s][0:n_plot]
            err_tmpl_XZ = self.tmpl_XZ[s][0:n_plot]
            err_tmpl_Y = self.tmpl_Y[s][0:n_plot]
            err_tmpl_YZ = self.tmpl_YZ[s][0:n_plot]
            err_comb = self.comb[s][0:n_plot]
            err_sep_X = self.sep_X[s][0:n_plot]
            err_sep_XZ = np.asarray(self.sep_XZ[s][0:n_plot], dtype='uint8')
            err_sep_Y = self.sep_Y[s][0:n_plot]
            err_sep_YZ = np.asarray(self.sep_YZ[s][0:n_plot], dtype='uint8')
   
            # output
            if err_comb.shape[0] > 0:
                print "Misclassified samples:"
                print "True labels:      ", ["(%d, %d)" % (xz, yz) for xz, yz in 
                                             zip(err_tmpl_XZ, err_tmpl_YZ)]
                print "Separated labels: ", ["(%d, %d)" % (xz, yz) for xz, yz in
                                             zip(err_sep_XZ, err_sep_YZ)]

                err_plt = np.concatenate((common.util.plot_samples(err_tmpl_X, twod=True), 
                                          common.util.plot_samples(err_sep_X, twod=True),
                                          common.util.plot_samples(err_tmpl_Y, twod=True),
                                          common.util.plot_samples(err_sep_Y, twod=True)))
                plt.imshow(myplt, interpolation='none')

                plt.figure()
                comb_plt = common.util.plot_samples(err_comb, twod=True)
                plt.imshow(comb_plt, interpolation='none')

