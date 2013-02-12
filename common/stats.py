# -*- coding: utf-8 -*-

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
    to level alpha"""
    M = np.mean(x)
    n = len(x)
    if n >= 2:
        Vs = np.var(x, ddof=1)
        t = scipy.stats.t.ppf(alpha/2, n-1)
        pm = abs(t*sqrt(Vs/n))
    else:
        pm = float('inf')
    return M, pm


