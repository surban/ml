
import gnumpy
import cudamat
import numpy as np

def test_nothing():
    print "nothing done"

    cm_mat = cudamat.cudamat()
    cm_mat.size[0] = 1
    cm_mat.size[1] = 1
    #cm_mat.data_device = ctypes.cast(x.ptr, ctypes.POINTER(ctypes.c_float))
    cm_mat.on_host = 0
    cm_mat.on_device = 1
    cm_mat.is_trans = 0
    cm_mat.owns_data = 0 
    # note: cm_mat dosen't owe the data; x does. So x will delete it.

    # create CUDAMatrix
    px = cudamat.CUDAMatrix(cm_mat)
    px._base = None # x won't be freed if the cudamat object isn't freed.
    px.mat_on_host = False # let cudamat know that we don't have a numpy
                           # array attached.

    # create garray
    ans = gnumpy.garray(px, [3,2], ___const_garray)



def test_cudamat():
    ac = np.eye(5)
    print "------------ ****** Allocating ac"
    a = cudamat.CUDAMatrix(ac)
    print "------------ ******** Done"
    print a
    del a
