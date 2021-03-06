# -*- coding: utf-8 -*-

import time

import scipy.io

import gnumpy as gp
from ml.apps.rbm import mnist_rbm_config as cfg
import ml.rbm.util as rbmutil
from ml.rbm.rbm import RestrictedBoltzmannMachine


# numeric overflow handling
#np.seterr(all='raise')
#gp.acceptable_number_types = 'no nans or infs'

gp.seed_rand(int(time.time()))

# Parameters for exact estimation of partition function
#use_ruslan = True
use_ruslan = False
epoch = cfg.epochs - 1
exact_pf_batch_size = 10000
#exact_pf_prec = 20
exact_pf_prec = 0

# enter output directory
rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, cfg.n_gibbs_steps,
                                 "exact_pf.txt", clean=False)

# Build RBM
rbm = RestrictedBoltzmannMachine(0, cfg.n_vis, cfg.n_hid, 0) 

# load Ruslan's RBM
if use_ruslan:
    print "Loading Ruslan's ml.rbm..."
    epoch = 99
    mdata = scipy.io.loadmat("mnistvh.mat")
    ml.rbm.bias_vis = gp.as_garray(mdata['visbiases'][0,:])
    ml.rbm.bias_hid = gp.as_garray(mdata['hidbiases'][0,:])
    ml.rbm.weights = gp.as_garray(mdata['vishid'])
else:
    rbmutil.load_parameters(rbm, "weights-%02i.npz" % epoch)


# calculate exact partition function
print "Calculating exact partiton function for RBM with %d hidden units..." \
    % (rbm.n_hid)
exact_pf = ml.rbm.partition_function(exact_pf_batch_size, exact_pf_prec)
if exact_pf_prec != 0:
    exact_lpf = exact_pf.ln()
else:
    exact_lpf = exact_pf
print "RBM exact log partition function: %f" % exact_lpf

