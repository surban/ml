# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import scipy.stats

from math import sqrt

def normal_mean_confint(x, alpha=0.05):
    """Returns the level alpha confidence interval for the mean of a normal
    distribution fitted to the data x"""
    M, pm = normal_mean(x, alpha)
    return M-pm, M+pm


def normal_mean(x, alpha=0.05):
    """Returns the mean of x and the half length of the confidence interval
    to level alpha. We assume that the data is normally distributed with
    unknown mean and variance."""
    M = np.mean(x)
    n = len(x)
    if n >= 2:
        Vs = np.var(x, ddof=1)
        t = scipy.stats.t.ppf(alpha/2, n-1)
        pm = abs(t*sqrt(Vs/n))
    else:
        pm = float('inf')
    return M, pm


def binomial_p_confint(successes, trails, alpha=0.05):
    phat, pm = binomial_p(successes, trails, alpha)
    return phat-pm, phat+pm

def binomial_p(successes, trails, alpha=0.05):
    """Return the success probability of the binomial distribution and the
    half length of the confidence interval to level alpha. We assume that
    the data is binomially distributed with unknown success probability."""
    phat = successes / trails
    z = scipy.stats.norm.ppf(alpha/2)
    pm = abs(z*sqrt((phat*(1-phat))/trails))
    return phat, pm
