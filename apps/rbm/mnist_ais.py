# -*- coding: utf-8 -*-

import time
import numpy as np
import gnumpy as gp
from apps.rbm import mnist_rbm_config as cfg

import rbm.util as rbmutil

from rbm.rbm import RestrictedBoltzmannMachine 
from rbm.ais import AnnealedImportanceSampler

# AIS parameters
gp.seed_rand(int(time.time()))
epoch = cfg.epochs - 1
#epoch = 9
ais_runs = 100
ais_gibbs_steps = 1
#ais_betas = np.linspace(0.0, 1.0,  1000)
ais_betas = np.concatenate((np.linspace(0.0, 0.5,   500, endpoint=False),
                            np.linspace(0.5, 0.9,  4000, endpoint=False),
                            np.linspace(0.9, 1.0, 10000)))
ais_base_samples = 10000
ais_base_chains = 1000
ais_base_gibbs_steps_between_samples = 1000
#ais_iterations = 10
ais_iterations = 1

# debug
check_base_rbm_partition_function = False
#np.seterr(all='raise')
#gp.acceptable_number_types = 'no nans or infs'


# enter output directory
rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, cfg.n_gibbs_steps,
                                 "ais.txt", clean=False)

# load RBM
rbm = RestrictedBoltzmannMachine(0, cfg.n_vis, cfg.n_hid, 0) 
rbmutil.load_parameters(rbm, "weights-%02i.npz" % epoch)

# init AIS estimator
print "Calculating base RBM biases using %d samples with %d Gibbs steps " \
    "inbetween..." % (ais_base_samples, ais_base_gibbs_steps_between_samples)
ais = AnnealedImportanceSampler(rbm, ais_base_samples, ais_base_chains,
                                ais_base_gibbs_steps_between_samples)
print "Saving base RBM biases to %s..." % filename
np.savez_compressed(filename, 
                    base_bias_vis=gp.as_numpy_array(ais.base_bias_vis))
print "Base RBM log partition function:  %f" % ais.base_log_partition_function()

# check base rbm log partition function
if check_base_rbm_partition_function:
    baserbm = RestrictedBoltzmannMachine(0, cfg.n_vis, cfg.n_hid, 0)
    baserbm.weights = gp.zeros(baserbm.weights.shape)
    baserbm.bias_hid = gp.zeros(baserbm.bias_hid.shape)
    baserbm.bias_vis = ais.base_bias_vis
    print "Base RBM log partition function using partition_func:  %f" % baserbm.partition_function(20, 50).ln()

# perform estimation of partition function
print "Estimating partition function using %dx %d AIS runs with %d intermediate "\
    "RBMs and %d Gibbs steps..." % (ais_iterations, ais_runs, len(ais_betas), ais_gibbs_steps)

with open("ais_iterations.csv", 'w') as outfile:
    outfile.write("iterations\tlog Z\n")
    lpfs = []
    for i in range(ais_iterations):
        lpf, lpf_m_3s, lpf_p_3s = ais.log_partition_function(ais_betas, ais_runs, 
                                                             ais_gibbs_steps)
        lpfs.append(lpf)
        print "%3d: ln Z = %3.6f; ln(Z-3s) = %3.6f; ln(Z+3s) = %3.6f" \
            % (i, lpf, lpf_m_3s, lpf_p_3s)
        outfile.write("%d\t%f\n" % (i, lpf))

lpf_mean = np.mean(lpfs)
lpf_std = np.std(lpfs)
print
print "mean: ln Z = %f +/- %f" % (lpf_mean, lpf_std)

np.savez_compressed("lpf-%02d.npz" % epoch, lpf=lpf_mean, lpf_std=lpf_std)



