# -*- coding: utf-8 -*-

import os
import sys
import Image as pil
import numpy as np
import gnumpy as gp
import scipy.io

import common.util as util
import common.dlutil as dlutil
import rbm.util as rbmutil
import mnist_rbm_config as cfg

from rbm.rbm import RestrictedBoltzmannMachine 
from rbm.util import sample_binomial
from common.util import myrand as mr

# parameters
plot_samples = False
rbmutil.use_debug_rng = True

# load dataset
X, TX = rbmutil.load_mnist(False)

# load ruslan's training set
mdata = scipy.io.loadmat("mnist.mat")
X = gp.as_garray(mdata['fbatchdata'])

# create output directory
rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, 
                                 cfg.n_gibbs_steps, "training.txt")

# Build RBM
rbm = RestrictedBoltzmannMachine(cfg.batch_size, cfg.n_vis, cfg.n_hid, 
                                 cfg.n_gibbs_steps) 
mr.seed(30)
rbm.weights = 0.01 * (0.5 - mr.rand(rbm.weights.shape))

# initialize momentums
weights_m1 = 0
bias_vis_m1 = 0
bias_hid_m1 = 0

print "weights:"
print rbm.weights[0:5,0:5]
#cfg.epochs = 2

# train
for epoch in range(cfg.epochs):
    seen_epoch_samples = 0
    pl_bit = 0
    pl_sum = 0
    rc_sum = 0

    for x in util.draw_slices(X, cfg.batch_size, kind='sequential', 
                              samples_are='rows', stop=True):
        print >>sys.stderr, "%d / %d (epoch: %d / %d)\r" % (seen_epoch_samples, 
                                                            X.shape[0], 
                                                            epoch, cfg.epochs),

        # binaraize x
        if cfg.binarize_data:
            x = sample_binomial(x)

        # perform weight update
        if cfg.use_pcd:
            weights_step, bias_vis_step, bias_hid_step = rbm.pcd_update(x)
        else:
            weights_step, bias_vis_step, bias_hid_step = rbm.cd_update(x)

        if epoch >= cfg.use_final_momentum_from_epoch:
            momentum = cfg.final_momentum
        else:
            momentum = cfg.initial_momentum

        weights_update = momentum * weights_m1 + \
            cfg.step_rate * (weights_step - cfg.weight_cost * rbm.weights)
        bias_vis_update = momentum * bias_vis_m1 + cfg.step_rate * bias_vis_step
        bias_hid_update = momentum * bias_hid_m1 + cfg.step_rate * bias_hid_step
    
        rbm.weights += weights_update
        rbm.bias_vis += bias_vis_update
        rbm.bias_hid += bias_hid_update

        weights_m1 = weights_update
        bias_vis_m1 = bias_vis_update
        bias_hid_m1 = bias_hid_update

        seen_epoch_samples += cfg.batch_size

        # calculate part of pseudo-likelihood
        #pl_sum += gp.sum(rbm.pseudo_likelihood_for_bit(x > 0.5, pl_bit))
        #pl_bit = (pl_bit + 1) % X.shape[1]

        # calculate part of reconstruction cost
        #rc_sum += gp.sum(rbm.reconstruction_cross_entropy(x > 0.5))

        #break 

    #############################################
    # end of batch: evaluate performance of model

    # save parameters
    rbmutil.save_parameters(rbm, epoch)

    # plot weights, samples from RBM and current state of PCD chains
    rbmutil.plot_weights(rbm, epoch)
    if plot_samples:
        rbmutil.plot_samples(rbm, epoch, TX[0:cfg.batch_size,:], 
                             cfg.n_plot_samples, cfg.n_gibbs_steps_between_samples)
    if cfg.use_pcd:
        rbmutil.plot_pcd_chains(rbm, epoch)

    # calculate pseudo likelihood and reconstruction cost
    pl = pl_sum / seen_epoch_samples * X.shape[1]
    rc = rc_sum / seen_epoch_samples

    print "Epoch %02d: reconstruction cost=%f, pseudo likelihood=%f" % \
        (epoch, rc, pl)

print "after weights:"
print rbm.weights[0:5,0:5]

print "random state:"
print mr.get_uint32()



