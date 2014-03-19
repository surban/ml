# -*- coding: utf-8 -*-

import time
import numpy as np
import gnumpy as gp
from ml.apps.rbm import mnist_rbm_config as cfg

import ml.rbm.util as rbmutil

from ml.rbm.rbm import RestrictedBoltzmannMachine
from ml.rbm.ais import AnnealedImportanceSampler

# numeric overflow handling
#np.seterr(all='raise')
#gp.acceptable_number_types = 'no nans or infs'

gp.seed_rand(int(time.time()))

# AIS parameters
iterations = 10
epoch = cfg.epochs - 1
#ais_base_samples = 50000
ais_base_samples = 5000
ais_base_chains = 1000
ais_base_gibbs_steps_between_samples = 1000

# enter output directory
rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, cfg.n_gibbs_steps,
                                 "base_rbm.txt", clean=False)

# Build RBM
rbm = RestrictedBoltzmannMachine(0, cfg.n_vis, cfg.n_hid, 0) 
rbmutil.load_parameters(rbm, "weights-%02i.npz" % epoch)

# calculate base RBM biases
print "Calculating base RBM biases using %d samples with %d Gibbs steps " \
    "inbetween..." % (ais_base_samples, ais_base_gibbs_steps_between_samples)
base_biases = np.zeros((iterations, ml.rbm.n_vis))
base_log_pf = []
for i in range(iterations):
    ais = AnnealedImportanceSampler(rbm, ais_base_samples, ais_base_chains,
                                    ais_base_gibbs_steps_between_samples)
    lpf = ais.base_log_partition_function()
    print "%2d: Base RBM log partition function:  %f" % (i, lpf)
    base_biases[i,:] = gp.as_numpy_array(ais.base_bias_vis)
    base_log_pf.append(lpf)

# calculate mean biases
base_biases_mean = np.mean(base_biases, axis=0)
base_biases_std = np.std(base_biases, axis=0)
base_log_pf_mean = np.mean(base_log_pf)
base_log_pf_std = np.std(base_log_pf)

print
print "Base RBM bias mean: \n", base_biases_mean
print
print "Base RBM bias std deviation: \n", base_biases_std
print
print "Base RBM bias mean of std deviation: %f" % np.mean(base_biases_std)
print "Base RBM log partition function:  %f +/- %f" % (base_log_pf_mean,
                                                       base_log_pf_std)

