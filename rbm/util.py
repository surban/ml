# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np
import Image as pil
import cPickle
import gzip
import os

from common import util
from common import dlutil
from common.util import myrand as mr

use_debug_rng = False

def sample_binomial(p):
    """Samples elementwise from the binomial distribution with 
    probability p"""
    if use_debug_rng:
        r = mr.rand(p.shape)
    else:
        r = gp.rand(p.shape)
    return r < p

def all_states(size):
    c = 0L
    while not c & (1 << size):
        bits = [int((c >> bit) & 1) for bit in range(size)]
        yield bits
        c += 1

def enter_rbm_plot_directory(dataset, n_hid, use_pcd, n_gibbs_steps,
                             logfilename, clean=True):
    if use_pcd:
        pcd_str = "p"
    else:
        pcd_str = ""
    outdir = "%s-rbm-%03d-%scd%02d" % (dataset, n_hid, pcd_str, n_gibbs_steps)
    util.enter_plot_directory(outdir, clean=clean)
    util.tee_output_to_log(logfilename)

def leave_rbm_plot_directory():
    util.untee_output()
    os.chdir("..")

def plot_samples(rbm, epoch, init_samples, 
                 n_plot_samples, n_gibbs_steps_between_samples):
    batch_size = init_samples.shape[0]
    pv = init_samples
    v = init_samples
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

def plot_weights(rbm, epoch):
    W = gp.as_numpy_array(rbm.weights)
    A = dlutil.tile_raster_images(W.T, (28, 28), (10, 10)).astype('float64')
    pilimage = pil.fromarray(A).convert('RGB')
    pilimage.save('filters-%02i.png' % epoch)

def plot_pcd_chains(rbm, epoch):
    v = gp.as_numpy_array(rbm.persistent_vis)
    A = dlutil.tile_raster_images(v, (28, 28), (8, 8)).astype('float64')
    pilimage = pil.fromarray(A).convert('RGB')
    pilimage.save('pcd-vis-%02i.png' % epoch)

def save_parameters(rbm, epoch_or_filename):
    if type(epoch_or_filename) == str:
        filename = epoch_or_filename
    else:
        filename = "weights-%02i.npz" % epoch_or_filename
    np.savez_compressed(filename, 
                        weights=gp.as_numpy_array(rbm.weights),
                        bias_vis=gp.as_numpy_array(rbm.bias_vis),
                        bias_hid=gp.as_numpy_array(rbm.bias_hid))

def load_parameters(rbm, filename):
    print "Loading RBM parameters form file %s" % filename
    state = np.load(filename)
    rbm.weights = gp.as_garray(state['weights'])
    rbm.bias_vis = gp.as_garray(state['bias_vis'])
    rbm.bias_hid = gp.as_garray(state['bias_hid'])

def load_mnist(with_verification_set):
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        (X, Z), (VX, VZ), (TX, TZ) = cPickle.load(f)

    TX = gp.as_garray(TX)
    if with_verification_set:
        X = gp.as_garray(X)
        VX = gp.as_garray(VX)
        return X, VX, TX
    else:
        X = gp.as_garray(np.concatenate((X,VX), axis=0))
        return X, TX

