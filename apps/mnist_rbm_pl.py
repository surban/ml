# -*- coding: utf-8 -*-

import os
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

# parameters
epoch = 14
do_sampling = True

# load dataset
X, VX, TX = rbmutil.load_mnist()

# enter output directory
rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, cfg.n_gibbs_steps,
                                 clean=False)

# Build RBM
rbm = RestrictedBoltzmannMachine(0, cfg.n_vis, cfg.n_hid, 0) 
rbmutil.load_parameters(rbm, "weights-%02i.npz" % epoch)
#rbmutil.load_parameters("../../../DeepLearningTutorials/code/rbm_plots/GPU-PCD/weights.npz")
#epoch = 99

# calculate statistics
seen_epoch_samples = 0
pl_bit = 0
pl_sum = 0
rc_sum = 0

for x in util.draw_slices(X, cfg.batch_size, kind='sequential', 
                          samples_are='rows', stop=True):
    print "%d / %d   \r" % (seen_epoch_samples, X.shape[0]),
    seen_epoch_samples += cfg.batch_size

    # calculate part of pseudo-likelihood
    pl_sum += gp.sum(rbm.pseudo_likelihood_for_bit(x > 0.5, pl_bit))
    pl_bit = (pl_bit + 1) % X.shape[1]

    # calculate part of reconstruction cost
    rc_sum += gp.sum(rbm.reconstruction_cross_entropy(x > 0.5))

#############################################
# end of batch: evaluate performance of model

# plot weights
rbmutil.plot_weights(rbm, epoch)
if do_sampling:
    rbmutil.plot_samples(rbm, epoch, TX[0:cfg.batch_size,:], 
                         cfg.n_plot_samples, cfg.n_gibbs_steps_between_samples)

# calculate pseudo likelihood and reconstruction cost
pl = pl_sum / seen_epoch_samples * X.shape[1]
rc = rc_sum / seen_epoch_samples

print "Epoch %02d: reconstruction cost=%f, pseudo likelihood=%f" % \
    (epoch, rc, pl)
