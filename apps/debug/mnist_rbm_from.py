# -*- coding: utf-8 -*-

import sys

import numpy as np
import scipy.io

import gnumpy as gp
from apps.rbm import mnist_rbm_config as cfg
import common.util as util
import rbm.util as rbmutil
from rbm.rbm import RestrictedBoltzmannMachine
from rbm.util import sample_binomial
from common.util import myrand as mr


# parameters
plot_samples = False
rbmutil.use_debug_rng = True
from_epoch = 1
#use_ruslans_start_weights = False
use_ruslans_start_weights = True
cfg.epochs = 2

# load dataset
X, TX = rbmutil.load_mnist(False)

# load ruslan's training set
mdata = scipy.io.loadmat("mnist.mat")
X = gp.as_garray(mdata['fbatchdata'])

# create output directory
rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, 
                                 cfg.n_gibbs_steps, "training_from.txt", 
                                 clean=False)

# Build RBM
rbm = RestrictedBoltzmannMachine(cfg.batch_size, cfg.n_vis, cfg.n_hid, 
                                 cfg.n_gibbs_steps) 
mr.seed(30)
rbm.weights = 0.01 * (0.5 - mr.rand(rbm.weights.shape))

# load weights
if use_ruslans_start_weights:
    filename = "matlab_epoch%d.mat" % (from_epoch-1 + 1)
    print "Loading Ruslan's start weights from %s" % filename
    mdata = scipy.io.loadmat(filename)
    rbm.weights = gp.as_garray(mdata['vishid'])
    rbm.bias_vis = gp.as_garray(mdata['visbiases'])
    rbm.bias_hid = gp.as_garray(mdata['hidbiases'])
else:
    print "Loading python start weights"
    filename = "weights-%02d.npz" % (from_epoch-1)
    rbmutil.load_parameters(rbm, filename)

# initialize momentums
weights_m1 = 0
bias_vis_m1 = 0
bias_hid_m1 = 0

np.set_printoptions(precision=15)
print "start weights after epoch %d:" % (from_epoch-1)
print rbm.weights[0:5,0:5]

print "random state: ", mr.get_uint32()

# train
for epoch in range(from_epoch, cfg.epochs):
    seen_epoch_samples = 0
    pl_bit = 0
    pl_sum = 0
    rc_sum = 0

    matlab_batch = 0
    for x in util.draw_slices(X, cfg.batch_size, kind='sequential', 
                              samples_are='rows', stop=True):
        matlab_batch += 1
        #if matlab_batch > 599:
        #    break

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
        
        if False:
            print "weights_step:"
            print weights_step[0:5,0:5]
            print "bias_vis_step:"
            print bias_vis_step[0:5]
            print "bias_hid_step:"
            print bias_hid_step[0:5]

            print "max(weights_step): ", gp.max(weights_step)

            sys.exit(0)         

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

        if True:
            rbmutil.save_parameters(rbm, "weights-from%02d-%02d-batch%05d.npz" % 
                                    (from_epoch, epoch, seen_epoch_samples))

        # calculate part of pseudo-likelihood
        #pl_sum += gp.sum(rbm.pseudo_likelihood_for_bit(x > 0.5, pl_bit))
        #pl_bit = (pl_bit + 1) % X.shape[1]

        # calculate part of reconstruction cost
        #rc_sum += gp.sum(rbm.reconstruction_cross_entropy(x > 0.5))

        #break 

    #############################################
    # end of batch: evaluate performance of model

    # save parameters
    rbmutil.save_parameters(rbm, "weights-from%02d-%02d.npz" % (from_epoch, epoch))

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

#print "after weights:"
#print rbm.weights[0:5,0:5]

#print "after bias_vis:"
#print rbm.bias_vis[0:5]

#print "after bias_hid:"
#print rbm.bias_hid[0:5]


print "random state: ", mr.get_uint32()



