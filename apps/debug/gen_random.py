# -*- coding: utf-8 -*-

import os
import sys
import Image as pil
import numpy as np
import gnumpy as gp
import scipy.io

import common.util as util
import common.dlutil as dlutil
import rbm.util as rbmutil
import mnist_rbm_config as cfg

from rbm.rbm import RestrictedBoltzmannMachine 
from rbm.util import sample_binomial
from common.util import myrand as mr

# parameters
n_rows = 1000
n_cols = 784
n_iters = 60000 / n_rows * 3


def write_bytes_to_file(file, data):
    for x in data:
        file.write(chr(int(x)))

# use gnumpy rng
gp.seed_rand(1)
with open("rng_gnumpy.dat", 'wb') as file:
    for i in range(n_iters):
        print "%d / %d\r" % (i, n_iters),
        gx = gp.rand((n_rows, n_cols))
        x = gp.as_numpy_array(gx)
        fx = np.reshape(x, -1)
        bx = np.floor(fx * 256)
        write_bytes_to_file(file, bx)

# use lcg rng
mr.seed(1)
with open("rng_lcg.dat", 'wb') as file:
    for i in range(n_iters):
        print "%d / %d\r" % (i, n_iters),
        gx = mr.rand((n_rows, n_cols))
        x = gp.as_numpy_array(gx)
        fx = np.reshape(x, -1)
        bx = np.floor(fx * 256)
        write_bytes_to_file(file, bx)






