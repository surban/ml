# -*- coding: utf-8 -*-

import os
import sys
import Image as pil
import numpy as np
import gnumpy as gp
import scipy.io
import time

import common.util as util
import common.dlutil as dlutil
import rbm.util as rbmutil
import mnist_rbm_config as cfg
import mnist_rbm

from rbm.rbm import RestrictedBoltzmannMachine 
from rbm.ais import AnnealedImportanceSampler
from rbm.util import sample_binomial
from common.util import myrand as mr


# parameters
n_iterations = 100
rbmutil.use_debug_rng = True
#rbmutil.use_debug_rng = False
ais_runs = 100
ais_gibbs_steps = 1
ais_betas = np.concatenate((np.linspace(0.0, 0.5,   500, endpoint=False),
                            np.linspace(0.5, 0.9,  4000, endpoint=False),
                            np.linspace(0.9, 1.0, 10000)))
ais_base_samples = 10000
ais_base_chains = 1000
ais_base_gibbs_steps_between_samples = 1000


# initialize
tr_lps = []
tst_lps = []

for i in range(n_iterations):
    seed = int(time.time())
    print "iteration %d / %d with seed %d" % (i, n_iterations, seed)

    # create output directory
    rbmutil.enter_rbm_plot_directory("mnist-%015d" % seed, cfg.n_hid, cfg.use_pcd, 
                                     cfg.n_gibbs_steps, "performance.txt")

    # train rbm
    print "Training RBM..."
    rbm = mnist_rbm.train_rbm(seed=seed, plot_samples=False)

    # estimate PF using AIS
    print "Estimating partition function using %d AIS runs with %d intermediate "\
          "RBMs and %d Gibbs steps..." % (ais_runs, len(ais_betas), ais_gibbs_steps)
    ais = AnnealedImportanceSampler(rbm, ais_base_samples, ais_base_chains,
                                    ais_base_gibbs_steps_between_samples)    
    lpf, lpf_m_3s, lpf_p_3s = ais.log_partition_function(ais_betas, ais_runs, 
                                                         ais_gibbs_steps)    
    rbm.log_pf = lpf

    # calculate log probability of training and test set
    tr_lp = gp.mean(rbm.normalized_log_p_vis(mnist_rbm.X))
    tst_lp = gp.mean(rbm.normalized_log_p_vis(mnist_rbm.TX))
    print "Average log p(x from training set) =  %f" % tr_lp
    print "Average log p(x from test set) =      %f" % tst_lp

    # accumulate statistics
    tr_lps.append(tr_lp)
    tst_lps.append(tst_lp)

    rbmutil.leave_rbm_plot_directory()
    np.savez_compressed("performance.npz", tr_lps=tr_lps, tst_lps=tst_lps)

    # output statistics
    print
    print "#############################################################"
    print "Runs:                              %d" % len(tr_lps)
    print 
    print "<log p(x from training set)>    =  %f" % np.mean(tr_lps)
    print "std[log p(x from training set)] =  %f" % np.std(tr_lps)
    print
    print "<log p(x from test set)>        =  %f" % np.mean(tst_lps)
    print "std[log p(x from test set)]     =  %f" % np.std(tst_lps)
    print "#############################################################"
    print




