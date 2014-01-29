# -*- coding: utf-8 -*-

import gc

import numpy as np
import scipy.io

import gnumpy as gp
from apps.rbm import mnist_rbm_config as cfg
import rbm.util as rbmutil


from_epoch = 1
epoch = 1

np.set_printoptions(precision=15)

rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, cfg.n_gibbs_steps,
                                 "compare.txt", clean=False)

print "epoch:      %d (using matlab counting: %d)" % (epoch, epoch+1)
print "from_epoch: %d (using matlab counting: %d)" % (from_epoch, from_epoch+1)
print


#for seen_epoch_samples in range(100, 60000, 100):
for seen_epoch_samples in range(50000, 60000 + 100, 100):
    #print "seen_samples:    %d"  % seen_epoch_samples
    gc.collect()

    filename = "matlab_from%d_epoch%d-%05d.mat" % (from_epoch+1, epoch+1, seen_epoch_samples)
    f=open(filename)
    f.close()
    mdata = scipy.io.loadmat(filename)
    m_weights = mdata['vishid']
    m_bias_vis = mdata['visbiases']
    m_bias_hid = mdata['hidbiases']

    pdata = np.load("weights-from%02d-%02d-batch%05d.npz" % 
                    (from_epoch, epoch, seen_epoch_samples))
    p_weights = pdata['weights']
    p_bias_vis = pdata['bias_vis']
    p_bias_hid = pdata['bias_hid']

    # compare 
    d_weights = np.abs(m_weights - p_weights)
    d_bias_vis = np.abs(m_bias_vis - p_bias_vis)
    d_bias_hid = np.abs(m_bias_hid - p_bias_hid)

    #print "max(d_weights)   = ", np.max(d_weights)
    #print "max(d_bias_vis)  = ", np.max(d_bias_vis)
    #print "max(d_bias_hid)  = ", np.max(d_bias_hid)
    #print

    print "samples %5d: max(d_weights)=%.8e  max(d_bias_vis)=%.8e  max(d_bias_hid)=%.8e" % \
        (seen_epoch_samples, np.max(d_weights), np.max(d_bias_vis), np.max(d_bias_hid))