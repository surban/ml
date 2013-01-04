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


# Hyperparameters
batch_size = 20
n_vis = 784
n_hid = 512
step_rate = 1E-1
#momentum = 0.9
momentum = 0
n_gibbs_steps = 15
epochs = 15
use_pcd = True
#use_pcd = False
n_plot_samples = 10
#n_gibbs_steps_between_samples = 1000
n_gibbs_steps_between_samples = 20

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
util.enter_plot_directory(outdir)

# Build RBM
rbm = RestrictedBoltzmannMachine(batch_size, n_vis, n_hid, n_gibbs_steps) 

# initialize momentums
weights_m1 = 0
bias_vis_m1 = 0
bias_hid_m1 = 0

# train
for epoch in range(epochs):
    seen_epoch_samples = 0
    pl_bit = 0
    pl_sum = 0
    rc_sum = 0

    for x in util.draw_slices(X, batch_size, kind='sequential', 
                              samples_are='rows', stop=True):
        print "%d / %d (epoch: %d / %d)\r" % (seen_epoch_samples, X.shape[0], 
                                              epoch, epochs),

        # perform weight update
        if use_pcd:
            weights_step, bias_vis_step, bias_hid_step = rbm.pcd_update(x)
        else:
            weights_step, bias_vis_step, bias_hid_step = rbm.cd_update(x)

        weights_update = momentum * weights_m1 + step_rate * weights_step
        bias_vis_update = momentum * bias_vis_m1 + step_rate * bias_vis_step
        bias_hid_update = momentum * bias_hid_m1 + step_rate * bias_hid_step
    
        rbm.weights += weights_update
        rbm.bias_vis += bias_vis_update
        rbm.bias_hid += bias_hid_update

        weights_m1 = weights_update
        bias_vis_m1 = bias_vis_update
        bias_hid_m1 = bias_hid_update

        seen_epoch_samples += batch_size

        # calculate part of pseudo-likelihood
        pl_sum += gp.sum(rbm.pseudo_likelihood_for_bit(x > 0.5, pl_bit))
        pl_bit = (pl_bit + 1) % X.shape[1]

        # calculate part of reconstruction cost
        rc_sum += gp.sum(rbm.reconstruction_cross_entropy(x > 0.5))

        #break 

    #############################################
    # end of batch: evaluate performance of model

    # plot weights
    W = gp.as_numpy_array(rbm.weights)
    A = dlutil.tile_raster_images(W.T, (28, 28), (10, 10)).astype('float64')
    pilimage = pil.fromarray(A).convert('RGB')
    pilimage.save('filters-%02i.png' % epoch)

    # plot pcd chains
    if use_pcd:
        v = gp.as_numpy_array(rbm.persistent_vis)
        A = dlutil.tile_raster_images(v, (28, 28), (8, 8)).astype('float64')
        pilimage = pil.fromarray(A).convert('RGB')
        pilimage.save('pcd-vis-%02i.png' % epoch)

    # sample from RBM and plot
    pv = TX[0:batch_size,:]
    v = pv > 0.5
    img = np.zeros((29 * n_plot_samples + 1, 29 * batch_size - 1),
                   dtype='uint8')
    for i in range(n_plot_samples):
        print "Sampling %02d / %02d                \r" % (i, n_plot_samples),
        A = dlutil.tile_raster_images(gp.as_numpy_array(pv), 
                                      img_shape=(28, 28), 
                                      tile_shape=(1, batch_size),
                                      tile_spacing=(1, 1))
        img[29*i:29*i+28,:] = A

        if i != n_plot_samples - 1:
            v, pv = rbm.gibbs_sample(v, n_gibbs_steps_between_samples)
            pv = gp.as_numpy_array(pv)
    pilimage = pil.fromarray(img) 
    pilimage.save('samples-%02i.png' % epoch)

    # save parameters
    np.savez_compressed("weights-%02i.npz" % epoch, 
                        weights=gp.as_numpy_array(rbm.weights),
                        bias_vis=gp.as_numpy_array(rbm.bias_vis),
                        bias_hid=gp.as_numpy_array(rbm.bias_hid))

    # calculate pseudo likelihood
    pl = pl_sum / seen_epoch_samples * X.shape[1]

    # calculate reconstruction cost
    rc = rc_sum / seen_epoch_samples

    print "Epoch %02d: reconstruction cost=%f, pseudo likelihood=%f" % \
        (epoch, rc, pl)

