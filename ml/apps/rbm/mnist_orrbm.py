# -*- coding: utf-8 -*-

import math

import numpy as np

import gnumpy as gp
import ml.rbm.orrbm


def or_performance(myrbm, svc, OX, OZ, iters, gibbs_steps, beta):
    OZ = gp.as_numpy_array(OZ)
    batch_size = 1000
    errs = 0

    for i in range(int(math.ceil(OX.shape[0] / float(batch_size)))):
        ox = OX[i*batch_size : (i+1)*batch_size, :]
        oz = OZ[i*batch_size : (i+1)*batch_size, :]

        x1, x2 = ml.rbm.orrbm.or_infer(myrbm, ox, iters, gibbs_steps, beta=beta)
        y1 = svc.predict(gp.as_numpy_array(x1))
        y2 = svc.predict(gp.as_numpy_array(x2))

        z1 = oz[:, 0]
        z2 = oz[:, 1]

        diff = (z1 - y1)**2 + (z2 - y2)**2
        errs += np.count_nonzero(diff)

    err_prob = errs / float(OX.shape[0])
    return err_prob





