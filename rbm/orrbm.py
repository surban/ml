# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np

from math import sqrt
from common.util import flatten_samples, unflatten_samples_like


width = 28
height = 28
base_y = 8

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
    n_samples = x.shape[0]

    o = gp.zeros((n_samples, height+2*base_y, 2*width))
    o[:, base_y:base_y+height, 0:width] = x

    if isinstance(x_shift, int) and isinstance(y_shift, int):
        assert 0 <= x_shift <= width and 0 <= y_shift <= 2*base_y
        o[:, y_shift:y_shift+height, x_shift:x_shift+width] += y
    else:
        assert len(x_shift) == n_samples and len(y_shift) == n_samples
        for s in range(n_samples):
            assert 0 <= x_shift[s] <= width and 0 <= y_shift[s] <= 2*base_y
            o[s, y_shift[s]:y_shift[s]+height, x_shift[s]:x_shift[s]+width] += y[s]
    o = o > 0.5

    return o



def or_rest(z, x):
    z = gp.as_numpy_array(z)
    x = gp.as_numpy_array(x)
    
    y = np.zeros(z.shape)
    ym = np.ones(z.shape)
    ym[(z == 1) & (x == 1)] = 0
    y[(z == 1) & (x == 0)] = 1
    
    # "no on force":
    # If pixels that are needed to explain the picture are forced on
    # this results in pixels that cannot be turned off by the rbm.
    ym[(z == 1) & (x == 0)] = 0

    # "no force":
    #ym = ym * 0

    # best is to have "no force" off and "no on force" on
    
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


def rect_intersection(a_x, a_y, a_width, a_height,
                      b_x, b_y, b_width, b_height):
    a_xs = set(range(a_x, a_x+a_width))
    a_ys = set(range(a_y, a_y+a_height))
    b_xs = set(range(b_x, b_x+b_width))
    b_ys = set(range(b_y, b_y+b_height))

    i_xs = a_xs & b_xs
    i_ys = a_ys & b_ys
    
    if len(i_xs) == 0 or len(i_ys) == 0:
        return 0, 0, 0, 0

    i_x = min(i_xs)
    i_y = min(i_ys)
    i_width = max(i_xs) - i_x
    i_height = max(i_ys) - i_y

    return i_x, i_y, i_width, i_height


def or_infer_with_shift(rbm, vis, x_shift, y_shift, iters, k, beta=1):
    assert vis.shape[1] == height+2*base_y and vis.shape[2] == 2*width
    
    infer = or_infer_with_shift_iter(rbm, vis, x_shift, y_shift, k, beta=beta)
    for i in range(iters):
        xs, ys = infer.next()

    return xs, ys

def or_infer_with_shift_iter(rbm, vis, x_shift, y_shift, k, beta=1):                                
    n_samples = vis.shape[0]
          
    x_sx = 0
    x_sy = base_y
    x_ex = x_sx + width
    x_ey = x_sy + height
    
    y_sx = x_shift
    y_sy = y_shift
    y_ex = y_sx + width
    y_ey = y_sy + height

    i_sx, i_sy, i_width, i_height = rect_intersection(x_sx, x_sy, width, height,
                                                      y_sx, y_sy, width, height)
    i_ex = i_sx + i_width
    i_ey = i_sy + i_height

    #print "intersection: ", i_sx, i_sy, i_width, i_height
    #print "fixed"
    
    xi = vis.copy()[:, x_sy:x_ey, x_sx:x_ex]
    xf = 1 - xi

    yi = vis.copy()[:, y_sy:y_ey, y_sx:y_ex]
    yf = 1 - yi

    while True:
        xs, _ = rbm.gibbs_sample(flatten_samples(xi), 
                                 k, vis_force=flatten_samples(xf), beta=beta)
        xs = unflatten_samples_like(xs, xi)

        xr, xrf = or_rest(vis[:, x_sy:x_ey, x_sx:x_ex], xs)
        yi_by_x = xr[:, i_sy-x_sy:i_ey-x_sy, i_sx-x_sx:i_ex-x_sx]
        yf_by_x = xrf[:, i_sy-x_sy:i_ey-x_sy, i_sx-x_sx:i_ex-x_sx]
        yi[:, i_sy-y_sy:i_ey-y_sy, i_sx-y_sx:i_ex-y_sx] = yi_by_x
        yf[:, i_sy-y_sy:i_ey-y_sy, i_sx-y_sx:i_ex-y_sx] = yf_by_x


        ys, _ = rbm.gibbs_sample(flatten_samples(yi), 
                                 k, vis_force=flatten_samples(yf), beta=beta)
        ys = unflatten_samples_like(ys, yi)

        yr, yrf = or_rest(vis[:, y_sy:y_ey, y_sx:y_ex], ys)
        xi_by_y = yr[:, i_sy-y_sy:i_ey-y_sy, i_sx-y_sx:i_ex-y_sx]
        xf_by_y = yrf[:, i_sy-y_sy:i_ey-y_sy, i_sx-y_sx:i_ex-y_sx] #fixed
        xi[:, i_sy-x_sy:i_ey-x_sy, i_sx-x_sx:i_ex-x_sx] = xi_by_y
        xf[:, i_sy-x_sy:i_ey-x_sy, i_sx-x_sx:i_ex-x_sx] = xf_by_y


        # BUG: yf instead of yrf, but worked anyway????

        yield xs, ys


def cross_entropy(rbm, vis, points, x_shift, y_shift,
                  iters, k, beta=1):
    assert vis.shape[1] == height+2*base_y and vis.shape[2] == 2*width

    infer = or_infer_with_shift_iter(rbm, vis, x_shift, y_shift, k, beta=beta)
    H = gp.zeros((vis.shape[0],))
    for n in range(points):
        for i in range(iters):
            xs, ys = infer.next()    

        H += (rbm.free_energy(flatten_samples(xs), beta=beta) + 
              rbm.free_energy(flatten_samples(ys), beta=beta))

    return H / points


