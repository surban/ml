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

gp.seed_rand(int(time.time()))

# parameters
use_ruslan = True
#epoch = cfg.epochs - 1
epoch = 9
batch = 0

#X, TX = rbmutil.load_mnist(False)
mdata = scipy.io.loadmat("mnist.mat")
X = mdata['fbatchdata']

# enter output directory
rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, cfg.n_gibbs_steps,
                                 "sample.txt", clean=False)

# Build RBM
rbm = RestrictedBoltzmannMachine(0, cfg.n_vis, cfg.n_hid, 0) 

# load Ruslan's RBM
if use_ruslan:
    print "Loading Ruslan's RBM..."
    epoch = 99
    mdata = scipy.io.loadmat("mnistvh.mat")
    rbm.bias_vis = gp.as_garray(mdata['visbiases'][0,:])
    rbm.bias_hid = gp.as_garray(mdata['hidbiases'][0,:])
    rbm.weights = gp.as_garray(mdata['vishid'])
else:
    rbmutil.load_parameters(rbm, "weights-%02i.npz" % epoch)

# compute hidden activations
v = X[batch*cfg.batch_size:(batch+1)*cfg.batch_size]
hid_act = gp.as_numpy_array(rbm.p_hid_given_vis(v))

print "hidden activations:"
print hid_act[0,:]



