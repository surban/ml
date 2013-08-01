import common.gpu
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curnd
import numpy as np

from nn.gpu_shift import *


def test_gpu_shift():
    x_len = 100
    s_len = x_len
    n_samples = 500000

    print "random data:"
    data = generate_random_data_gpu(x_len, n_samples, binary=True) 
    print data

    shifts, shifts_hot = generate_shifts_gpu(x_len, n_samples)
    print "shifts:"
    print shifts
    print "shifts_hot:"
    print shifts_hot

    print "shifted:"
    shifted = generate_shifted_gpu(data, shifts)
    print shifted


