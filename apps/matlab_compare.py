# -*- coding: utf-8 -*-

import os
import sys
import Image as pil
import numpy as np
import gnumpy as gp
import scipy.io

import common.util as util
import common.dlutil as dlutil
import rbm.util as rbmutil
import mnist_rbm_config as cfg

from rbm.rbm import RestrictedBoltzmannMachine 
from rbm.util import sample_binomial
from common.util import myrand as mr


np.set_printoptions(precision=15)

rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, cfg.n_gibbs_steps,
                                 "compare.txt", clean=False)

for epoch in range(cfg.epochs):
    print "epoch: %d (using matlab counting: %d)" % (epoch, epoch+1)

    mdata = scipy.io.loadmat("matlab_epoch%d.mat" % (epoch + 1))
    m_weights = mdata['vishid']
    m_bias_vis = mdata['visbiases']
    m_bias_hid = mdata['hidbiases']

    pdata = np.load("weights-%02d.npz" % epoch)
    p_weights = pdata['weights']
    p_bias_vis = pdata['bias_vis']
    p_bias_hid = pdata['bias_hid']

    # compare 
    d_weights = np.abs(m_weights - p_weights)
    d_bias_vis = np.abs(m_bias_vis - p_bias_vis)
    d_bias_hid = np.abs(m_bias_hid - p_bias_hid)

    print "max(d_weights)   = ", np.max(d_weights)
    print "max(d_bias_vis)  = ", np.max(d_bias_vis)
    print "max(d_bias_hid)  = ", np.max(d_bias_hid)
    print
