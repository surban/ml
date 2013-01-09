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

#gp.acceptable_number_types = 'no nans or infs'
epoch = 14
do_sampling = True

# Hyperparameters
use_pcd = True
batch_size = 20
n_vis = 784
n_hid = 512
n_gibbs_steps = 15
n_plot_samples = 10
n_gibbs_steps_between_samples = 1000

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
rbm = RestrictedBoltzmannMachine(batch_size, n_vis, n_hid, n_gibbs_steps) 

# load RBM state
filename = "weights-%02i.npz" % epoch
#filename = "../../../DeepLearningTutorials/code/rbm_plots/GPU-PCD/weights.npz"
#epoch = 99
print "Loading state %s" % filename
state = np.load(filename)
rbm.weights = gp.as_garray(state['weights'])
rbm.bias_vis = gp.as_garray(state['bias_vis'])
rbm.bias_hid = gp.as_garray(state['bias_hid'])

# calculate statistics
seen_epoch_samples = 0
pl_bit = 0
pl_sum = 0
rc_sum = 0

for x in util.draw_slices(X, batch_size, kind='sequential', 
                            samples_are='rows', stop=True):
    print "%d / %d   \r" % (seen_epoch_samples, X.shape[0]),
    seen_epoch_samples += batch_size

    # calculate part of pseudo-likelihood
    pl_sum += gp.sum(rbm.pseudo_likelihood_for_bit(x > 0.5, pl_bit))
    pl_bit = (pl_bit + 1) % X.shape[1]

    # calculate part of reconstruction cost
    rc_sum += gp.sum(rbm.reconstruction_cross_entropy(x > 0.5))

#############################################
# end of batch: evaluate performance of model

# plot weights
W = gp.as_numpy_array(rbm.weights)
A = dlutil.tile_raster_images(W.T, (28, 28), (10, 10)).astype('float64')
pilimage = pil.fromarray(A).convert('RGB')
pilimage.save('filters-%02i.png' % epoch)

# sample from RBM and plot
if do_sampling:
    pv = TX[0:batch_size,:]
    #v = pv > 0.5
    v = pv
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

# calculate pseudo likelihood
pl = pl_sum / seen_epoch_samples * X.shape[1]

# calculate reconstruction cost
rc = rc_sum / seen_epoch_samples

print "Epoch %02d: reconstruction cost=%f, pseudo likelihood=%f" % \
    (epoch, rc, pl)

