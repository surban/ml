# -*- coding: utf-8 -*-

import os
import sys
import time
import Image as pil
import numpy as np
import gnumpy as gp
import scipy.io

import common.util as util
import common.dlutil as dlutil
import rbm.util as rbmutil
import mnist_rbm_config as cfg

from rbm.rbm import RestrictedBoltzmannMachine 
from rbm.ais import AnnealedImportanceSampler

# numeric overflow handling
#np.seterr(all='raise')
#gp.acceptable_number_types = 'no nans or infs'

# parameters
epoch = cfg.epochs - 1
#epoch = 9
use_ruslan = False

# load dataset
X, TX = rbmutil.load_mnist(False)

# load ruslan's training set
mdata = scipy.io.loadmat("mnist.mat")
X = gp.as_garray(mdata['fbatchdata'])

# enter output directory
rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, cfg.n_gibbs_steps,
                                 "prob.txt", clean=False)

# Build RBM
rbm = RestrictedBoltzmannMachine(0, cfg.n_vis, cfg.n_hid, 0) 

# load Ruslan's RBM
if use_ruslan:
    print "Loading Ruslan's RBM..."
    mdata = scipy.io.loadmat("matlab_epoch%d.mat" % (epoch + 1))
    rbm.bias_vis = gp.as_garray(mdata['visbiases'][0,:])
    rbm.bias_hid = gp.as_garray(mdata['hidbiases'][0,:])
    rbm.weights = gp.as_garray(mdata['vishid'])
else:
    rbmutil.load_parameters(rbm, "weights-%02i.npz" % epoch)

# load pratition function
if use_ruslan:
    filename = "matlab-lpf-%02d.npz" % (epoch+1)
else:
    filename = "lpf-%02d.npz" % epoch
print "Loading partition function %s" % filename
lpf = np.load(filename)
rbm.log_pf = lpf['lpf']

# calculate log probability of training set
tr_lp = gp.mean(rbm.normalized_log_p_vis(X))
tst_lp = gp.mean(rbm.normalized_log_p_vis(TX))

print "Average log p(x from training set) =  %f" % tr_lp
print "Average log p(x from test set) =      %f" % tst_lp
