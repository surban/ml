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
from rbm.ais import AnnealedImportanceSampler

# numeric overflow handling
#np.seterr(all='raise')
#gp.acceptable_number_types = 'no nans or infs'

gp.seed_rand(int(time.time()))

# Parameters for exact estimation of partition function
exact_pf_batch_size = 10000
exact_pf_prec = 300

# AIS parameters
epoch = cfg.epochs - 1
ais_runs = 100
ais_gibbs_steps = 1
ais_betas = np.concatenate((np.linspace(0.0, 0.5,   500, endpoint=False),
                            np.linspace(0.5, 0.9,  4000, endpoint=False),
                            np.linspace(0.9, 1.0, 10000)))
ais_base_samples = 100
ais_base_chains = 10
ais_base_gibbs_steps_between_samples = 1

#ais_base_samples = 50000
#ais_base_chains = 1000
#ais_base_gibbs_steps_between_samples = 1000

# enter output directory
rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, cfg.n_gibbs_steps,
                                 clean=False)

# Build RBM
rbm = RestrictedBoltzmannMachine(0, cfg.n_vis, cfg.n_hid, 0) 
rbmutil.load_parameters(rbm, "weights-%02i.npz" % epoch)

# calculate exact partition function
print "Calculating exact partiton function for RBM with %d hidden units..." \
    % (rbm.n_hid)
exact_pf = rbm.partition_function(exact_pf_batch_size, exact_pf_prec)
exact_lpf = exact_pf.ln()
print "RBM exact log partition function: %f" % exact_lpf

sys.exit(0)

# init AIS estimator
print "Calculating base RBM biases using %d samples with %d Gibbs steps " \
    "inbetween..." % (ais_base_samples, ais_base_gibbs_steps_between_samples)
ais = AnnealedImportanceSampler(rbm, ais_base_samples, ais_base_chains,
                                ais_base_gibbs_steps_between_samples)
#print "Base RBM visible biases:"
#print ais.base_bias_vis
print "Base RBM log partition function:  %f" % ais.base_log_partition_function()

# perform estimation of partition function
print "Estimating partition function using %d AIS runs with %d intermediate "\
    "RBMs and %d Gibbs steps..." % (ais_runs, len(ais_betas), ais_gibbs_steps)
lpf = ais.log_partition_function(ais_betas, ais_runs, ais_gibbs_steps)
print "RBM log partition function:       %f" % lpf


