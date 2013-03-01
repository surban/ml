# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np
import Image as pil
import cPickle
import gzip
import os
import scipy.io

from common import util
from common import dlutil
from common.util import myrand 
from common.util import get_base_dir

use_debug_rng = False

gp.expensive_check_probability = 0

loaded_datasets = {}


def generation_performance(svc, tmpl_X, gen_X, tmpl_Z):
    gen_Z = svc.predict(gp.as_numpy_array(gen_X))
    tmpl_Z = gp.as_numpy_array(tmpl_Z)

    diff = tmpl_Z - gen_Z
    errs = np.count_nonzero(diff)
    err_prob = errs / float(tmpl_X.shape[0])

    err_tmpl_X = gp.zeros((errs, tmpl_X.shape[1]))
    err_gen_X = gp.zeros((errs, gen_X.shape[1]))
    err_tmpl_Z = np.zeros((errs,), dtype='uint8')
    err_gen_Z = np.zeros((errs,), dtype='uint8')
    for n, k in enumerate(np.nonzero(diff)[0]):
        err_tmpl_X[n, :] = tmpl_X[k, :]
        err_gen_X[n, :] = gen_X[k, :]
        err_tmpl_Z[n] = tmpl_Z[k]
        err_gen_Z[n] = int(gen_Z[k])

    err_tmpl_CZ = svc.predict(gp.as_numpy_array(err_tmpl_X))
    return err_prob, err_tmpl_X, err_gen_X, err_tmpl_Z, err_gen_Z, err_tmpl_CZ


def sample_binomial(p):
    """Samples elementwise from the binomial distribution with 
    probability p"""
    if use_debug_rng:
        r = myrand.rand(p.shape)
    else:
        r = gp.rand(p.shape)
    #n = np.random.random(p.shape)
    #n = gp.rand(p.shape)
    #r = gp.zeros(p.shape)
    return r < p

def all_states(size):
    c = 0L
    while not c & (1 << size):
        bits = [int((c >> bit) & 1) for bit in range(size)]
        yield bits
        c += 1

def enter_rbm_plot_directory(tcfg, logfilename=None, clean=False):
    util.enter_plot_directory(tcfg.output_dir, clean=clean)
    if logfilename is not None:
        util.tee_output_to_log(logfilename)

def leave_rbm_plot_directory():
    util.untee_output()
    util.leave_plot_directory()

def plot_samples(init_samples, samples, save_to_file=False, epoch=None):
    all_samples = gp.concatenate((init_samples.reshape((1, 
                                                        init_samples.shape[0],
                                                        init_samples.shape[1])),
                                  samples))
    n_samples = all_samples.shape[0]
    n_chains = all_samples.shape[1]
    img = np.zeros((29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')

    for step in range(n_samples):
        v = all_samples[step, :, :]
        A = dlutil.tile_raster_images(gp.as_numpy_array(v), 
                                      img_shape=(28, 28), 
                                      tile_shape=(1, n_chains),
                                      tile_spacing=(1, 1))
        img[29*step:29*step+28,:] = A

    if save_to_file:
        assert epoch is not None
        pilimage = pil.fromarray(img) 
        pilimage.save('samples-%02i.png' % epoch)
    return img

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

def load_parameters(rbm, epoch_or_filename):
    if type(epoch_or_filename) == str:
        filename = epoch_or_filename
    else:
        filename = "weights-%02i.npz" % epoch_or_filename
    print "Loading RBM parameters form file %s" % filename
    state = np.load(filename)
    rbm.weights = gp.as_garray(state['weights'])
    rbm.bias_vis = gp.as_garray(state['bias_vis'])
    rbm.bias_hid = gp.as_garray(state['bias_hid'])

def load_ruslan_parameters(rbm, filename):
    print "Loading RBM parameters form file %s" % filename
    mdata = scipy.io.loadmat(filename)
    rbm.weights = gp.as_garray(mdata['vishid'])
    rbm.bias_vis = gp.as_garray(mdata['visbiases'][0])
    rbm.bias_hid = gp.as_garray(mdata['hidbiases'][0])

def load_mnist(with_verification_set):
    with gzip.open(os.path.join(get_base_dir(), "datasets", "mnist.pkl.gz"), 
                   'rb') as f:
        (X, Z), (VX, VZ), (TX, TZ) = cPickle.load(f)

    if with_verification_set:
        if 'mnistv_X' not in loaded_datasets:
            loaded_datasets['mnistv_X'] = gp.as_garray(X)
            loaded_datasets['mnistv_Z'] = Z
            loaded_datasets['mnistv_VX'] = gp.as_garray(VX)
            loaded_datasets['mnistv_VZ'] = VZ
            loaded_datasets['mnistv_TX'] = gp.as_garray(TX)
            loaded_datasets['mnistv_TZ'] = TZ
        return (loaded_datasets['mnistv_X'], 
                loaded_datasets['mnistv_VX'],
                loaded_datasets['mnistv_TX'],
                loaded_datasets['mnistv_Z'],
                loaded_datasets['mnistv_VZ'],
                loaded_datasets['mnistv_TZ'])
    else:
        if 'mnist_X' not in loaded_datasets:
            loaded_datasets['mnist_X'] = gp.as_garray(np.concatenate((X,VX), 
                                                                     axis=0))
            loaded_datasets['mnist_Z'] = np.concatenate((Z,VZ), axis=0)
            loaded_datasets['mnist_TX'] = gp.as_garray(TX)       
            loaded_datasets['mnist_TZ'] = TZ
        return (loaded_datasets['mnist_X'], loaded_datasets['mnist_TX'],
                loaded_datasets['mnist_Z'], loaded_datasets['mnist_TZ'])


def load_ruslan_mnist():
    mdata = scipy.io.loadmat(os.path.join(get_base_dir(), 
                                          "datasets", "mnist.mat"))
    return (gp.as_garray(mdata['fbatchdata']), 
            gp.as_garray(mdata['test_fbatchdata']))


def load_or_dataset():
    if 'or_O' not in loaded_datasets:
        trn_data = np.load(os.path.join(get_base_dir(), "datasets", "ordata.npz"))
        tst_data = np.load(os.path.join(get_base_dir(), "datasets", "ordata_test.npz"))

        loaded_datasets['or_OX'] = gp.as_garray(trn_data['O'])
        loaded_datasets['or_OZ'] = gp.as_garray(trn_data['OZ'])
        loaded_datasets['or_TOX'] = gp.as_garray(tst_data['O'])
        loaded_datasets['or_TOZ'] = gp.as_garray(tst_data['OZ'])

    return (loaded_datasets['or_OX'], loaded_datasets['or_TOX'],
            loaded_datasets['or_OZ'], loaded_datasets['or_TOZ'])
