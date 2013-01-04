# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np


def sample_binomial(p):
    """Samples elementwise from the binomial distribution with 
    probability p"""
    r = gp.rand(p.shape)
    return r < p



