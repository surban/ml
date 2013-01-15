# -*- coding: utf-8 -*-

import os
import sys
import time
import Image as pil
import numpy as np
import gnumpy as gp

import common.util as util
import common.dlutil as dlutil
import rbm.util as rbmutil
import mnist_rbm_config as cfg

from rbm.rbm import RestrictedBoltzmannMachine 

# numeric overflow handling
#np.seterr(all='raise')
#gp.acceptable_number_types = 'no nans or infs'

gp.seed_rand(int(time.time()))

# Parameters for exact estimation of partition function
epoch = cfg.epochs - 1
exact_pf_batch_size = 10000
exact_pf_prec = 20

# enter output directory
rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, cfg.n_gibbs_steps,
                                 "exact_pf.txt", clean=False)

# Build RBM
rbm = RestrictedBoltzmannMachine(0, cfg.n_vis, cfg.n_hid, 0) 
rbmutil.load_parameters(rbm, "weights-%02i.npz" % epoch)

# calculate exact partition function
print "Calculating exact partiton function for RBM with %d hidden units..." \
    % (rbm.n_hid)
exact_pf = rbm.partition_function(exact_pf_batch_size, exact_pf_prec)
exact_lpf = exact_pf.ln()
print "RBM exact log partition function: %f" % exact_lpf