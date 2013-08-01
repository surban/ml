import common.gpu
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curnd
import numpy as np

import pycuda.autoinit

def test_gpuarray_to_garray():

    x = curnd.rand((3,3), dtype=np.float32)
    x = x + 2
    gx = common.gpu.gpuarray_to_garray(x)

    print "x:"
    print x

    print "gpuarray_to_garray(x):"
    print gx

