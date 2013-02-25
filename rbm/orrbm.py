# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np

def or_sample(x, y):
    return (x + y) > 0.5

def or_rest(z, x):
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
