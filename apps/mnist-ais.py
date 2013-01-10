# -*- coding: utf-8 -*-

import cPickle
import itertools
import gzip
import os
import Image as pil
import numpy as np
import gnumpy as gp

import common.util as util
import common.dlutil as dlutil

from rbm.rbm import RestrictedBoltzmannMachine 
from rbm.ais import AnnealedImportanceSampler

#np.seterr(all='raise')
#gp.acceptable_number_types = 'no nans or infs'
epoch = 14
do_sampling = True

# Hyperparameters
use_pcd = True
batch_size = 20
n_vis = 784
n_hid = 512
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


# make data
f = gzip.open('mnist.pkl.gz', 'rb')
(X, Z), (VX, VZ), (TX, TZ) = cPickle.load(f)
X = gp.as_garray(X)
TX = gp.as_garray(TX)
f.close()

# create output directory
outdir = "mnist-rbm"
if use_pcd:
    outdir += "-pcd"
util.enter_plot_directory(outdir, clean=False)

# Build RBM
rbm = RestrictedBoltzmannMachine(batch_size, n_vis, n_hid, 0) 

# load RBM state
filename = "weights-%02i.npz" % epoch
#filename = "../../../DeepLearningTutorials/code/rbm_plots/GPU-PCD/weights.npz"
#epoch = 99
print "Loading state %s" % filename
state = np.load(filename)
rbm.weights = gp.as_garray(state['weights'])
rbm.bias_vis = gp.as_garray(state['bias_vis'])
rbm.bias_hid = gp.as_garray(state['bias_hid'])

# init AIS estimator
print "Calculating base RBM biases using %d samples with %d Gibbs steps " \
    "inbetween..." % (ais_base_samples, ais_base_gibbs_steps_between_samples)
ais = AnnealedImportanceSampler(rbm, ais_base_samples, ais_base_chains,
                                ais_base_gibbs_steps_between_samples)
#print "Base RBM visible biases:"
#print ais.base_bias_vis
print "Base RBM log partition function:   %f" % ais.base_log_partition_function()

# perform estimation of partition function
print "Estimating partition function using %d AIS runs with %d intermediate "\
    "RBMs and %d Gibbs steps..." % (ais_runs, len(ais_beta), ais_gibbs_steps)
pf = ais.partition_function(ais_betas, ais_runs, ais_gibbs_steps)
print "RBM partiiton function:        %f" % pf


