# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np

from math import sqrt
from common.util import flatten_samples, unflatten_samples_like

def generate_or_dataset(X, Z, samples):
    X = gp.as_numpy_array(X)
    Z = gp.as_numpy_array(Z)

    si = np.random.randint(0, X.shape[0], 
                           size=(samples, 2))

    x = X[si[:,0], :]
    y = X[si[:,1], :]
    O = or_sample(x, y)

    OZ = np.zeros((samples, 2))
    OZ[:, 0] = Z[si[:,0]]
    OZ[:, 1] = Z[si[:,1]]

    return O, OZ

def generate_sample_indices_for_or_dataset(S, n_samples):
    S = gp.as_numpy_array(S)
    return np.random.randint(0, S.shape[0], size=(n_samples, 2))

def generate_or_dataset_with_shift(S, SZ, ref_SZ, x_shift, y_shift, n_samples,
                                   sample_indices=None):
    S = gp.as_numpy_array(S)
    SZ = gp.as_numpy_array(SZ)

    if sample_indices is not None:
        si = sample_indices
    else:
        si = generate_sample_indices_for_or_dataset(S, n_samples)

    X = S[si[:,0]]
    XZ = SZ[si[:,0]]
    ref_XZ = ref_SZ[si[:,0]]
    Y = S[si[:,1]]
    YZ = SZ[si[:,1]]
    ref_YZ = ref_SZ[si[:,1]]

    O = or_sample_with_shift(X, Y, x_shift, y_shift)

    return X, XZ, ref_XZ, Y, YZ, ref_YZ, O


def save_or_dataset(filename, O, OZ):
    np.savez_compressed(filename, O=O, OZ=OZ)

def or_sample(x, y):
    return (x + y) > 0.5

def or_sample_with_shift(x, y, x_shift, y_shift):    
    if x.ndim == 2:
        x = x.reshape((1, x.shape[0], x.shape[1]))
        y = y.reshape((1, y.shape[0], y.shape[1]))
        one_sample = True
    else:
        one_sample = False

    n_samples = x.shape[0]
    height = x.shape[1]
    width = x.shape[2]
    assert 0 <= x_shift <= width and 0 <= y_shift <= height

    o = gp.zeros((n_samples, height + y_shift, width + x_shift))
    o[:,0:height,0:width] = x
    o[:,y_shift:,x_shift:] = o[:,y_shift:,x_shift:] + y
    o = o > 0.5

    if one_sample:
        o = o.reshape((o.shape[1], o.shape[2]))

    return o

def or_rest(z, x):
    z = gp.as_numpy_array(z)
    x = gp.as_numpy_array(x)
    
    y = np.zeros(z.shape)
    ym = np.ones(z.shape)
    ym[(z == 1) & (x == 1)] = 0
    y[(z == 1) & (x == 0)] = 1
    
    # If pixels that are needed to explain the picture are forced on
    # this results in pixels that cannot be turned off by the rbm.
    ym[(z == 1) & (x == 0)] = 0

    # turn off whole force:
    #ym = ym * 0
    
    return gp.as_garray(y), gp.as_garray(ym)   

def or_rest_fast(z, x):
    z = gp.as_numpy_array(z)
    x = gp.as_numpy_array(x)
    
    y = np.zeros(z.shape)
    ym = np.ones(z.shape)
    ym[(z == 1) & (x == 1)] = 0
    y[(z == 1) & (x == 0)] = 1
    
    return gp.as_garray(y), gp.as_garray(ym)   

def or_infer(rbm, vis, iters, k, beta=1):
    xi = vis
    xf = 1 - vis

    for i in range(iters):
        xs, _ = rbm.gibbs_sample(xi, k, vis_force=xf, beta=beta)
        yi, yf = or_rest(vis, xs)
        ys, _ = rbm.gibbs_sample(yi, k, vis_force=yf, beta=beta)
        xi, xf = or_rest(vis, ys)

    return xs, ys

def or_infer_with_shift(rbm, vis, x_shift, y_shift, 
                        iters, k, beta=1):
    
    n_samples = vis.shape[0]
    height = vis.shape[1] - y_shift
    width = vis.shape[2] - x_shift
          
    xi = vis.copy()[:,0:height,0:width]
    xf = 1 - xi

    yi = vis.copy()[:,y_shift:,x_shift:]
    yf = 1 - yi

    for i in range(iters):
        xs, _ = rbm.gibbs_sample(flatten_samples(xi), 
                                 k, vis_force=flatten_samples(xf), beta=beta)
        xs = unflatten_samples_like(xs, xi)
        xr, xrf = or_rest(vis[:, 0:height, 0:width], xs)

        yi_by_x = xr[:, y_shift:, x_shift:]
        yf_by_x = xrf[:, y_shift:, x_shift:]
        yi[:, 0:height-y_shift, 0:width-x_shift] = yi_by_x
        yf[:, 0:height-y_shift, 0:width-x_shift] = yf_by_x


        ys, _ = rbm.gibbs_sample(flatten_samples(yi), 
                                 k, vis_force=flatten_samples(yf), beta=beta)
        ys = unflatten_samples_like(ys, yi)
        yr, yrf = or_rest(vis[:, y_shift:, x_shift:], ys)

        xi_by_y = yr[:, 0:height-y_shift, 0:width-x_shift]
        xf_by_y = yf[:, 0:height-y_shift, 0:width-x_shift]
        xi[:, y_shift:, x_shift:] = xi_by_y
        xf[:, y_shift:, x_shift:] = xf_by_y

    return xs, ys

def cross_entropy(rbm, vis, points, x_shift, y_shift,
                  iters, k, beta=1):

    n_samples = vis.shape[0]
    height = vis.shape[1] - y_shift
    width = vis.shape[2] - x_shift
          
    xi = vis.copy()[:,0:height,0:width]
    xf = 1 - xi

    yi = vis.copy()[:,y_shift:,x_shift:]
    yf = 1 - yi

    H = 0
    for n in range(points):
        for i in range(iters):
            xs, _ = rbm.gibbs_sample(flatten_samples(xi), 
                                     k, vis_force=flatten_samples(xf), beta=beta)
            xs = unflatten_samples_like(xs, xi)
            xr, xrf = or_rest(vis[:, 0:height, 0:width], xs)

            yi_by_x = xr[:, y_shift:, x_shift:]
            yf_by_x = xrf[:, y_shift:, x_shift:]
            yi[:, 0:height-y_shift, 0:width-x_shift] = yi_by_x
            yf[:, 0:height-y_shift, 0:width-x_shift] = yf_by_x


            ys, _ = rbm.gibbs_sample(flatten_samples(yi), 
                                     k, vis_force=flatten_samples(yf), beta=beta)
            ys = unflatten_samples_like(ys, yi)
            yr, yrf = or_rest(vis[:, y_shift:, x_shift:], ys)

            xi_by_y = yr[:, 0:height-y_shift, 0:width-x_shift]
            xf_by_y = yf[:, 0:height-y_shift, 0:width-x_shift]
            xi[:, y_shift:, x_shift:] = xi_by_y
            xf[:, y_shift:, x_shift:] = xf_by_y

        H += (rbm.free_energy(flatten_samples(xs), beta=beta) + 
              rbm.free_energy(flatten_samples(ys), beta=beta))

    return H