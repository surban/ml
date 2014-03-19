from __future__ import division

import ml.common.gpu
import pycuda.driver as cuda
#import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gp
import pycuda.curandom as gprng
import numpy as np
import math

from ml.common.gpu import gpuarray_to_garray

# Kernels
cuda_mod = SourceModule(r"""
#include <stdio.h>

typedef unsigned int uint32;
typedef signed int int32;

__forceinline__ __device__ uint32 elem(uint32 st_i, uint32 st_j, uint32 i, uint32 j)
{
    return i*st_i + j*st_j;
}

__global__ void shift_to_hot(uint32 *shifts, float *hot,
                             uint32 hot_st0, uint32 hot_st1,
                             uint32 shift_len, uint32 n_samples)
{
    uint32 smpl = threadIdx.x + blockIdx.x*blockDim.x;
    if (smpl >= n_samples)
        return;
    uint32 shft = shifts[smpl];
    
    for (uint32 i=0; i < shift_len; i++)
    {
        hot[elem(hot_st0, hot_st1, i, smpl)] = (i == shft) ? 1.0f : 0.0f;
    }    
}

__global__ void shift(float *data, uint32 *shifts, float *shifted,
                      uint32 st0, uint32 st1,
                      uint32 data_len, uint32 n_samples)
{
    //printf("data: %p   shifts: %p   shifted: %p\n", data, shifts, shifted);
    //printf("st0: %u    st1: %u\n", st0, st1);
    //printf("data_len: %u   n_samples: %u   threads_per_block: %u\n",
    //        data_len, n_samples, threads_per_block);

    uint32 smpl = threadIdx.x + blockIdx.x*blockDim.x;
    if (smpl >= n_samples)
        return;
    uint32 shft = shifts[smpl];

    //printf("smpl: %u\n", smpl);
    //printf("working with shift %u\n", shft);

    for (uint32 src=0; src < data_len; src++)
    {
        uint32 trgt = src + shft;
        if (trgt >= data_len)
            trgt -= data_len;

        shifted[elem(st0, st1, trgt, smpl)] = data[elem(st0, st1, src, smpl)];
    }
}
""")
gpu_shift = cuda_mod.get_function('shift')
gpu_shift_to_hot = cuda_mod.get_function('shift_to_hot')

gpu_rng = gprng.XORWOWRandomNumberGenerator()


def generate_random_data(x_len, n_samples, binary=False):
    data = 4.0 * (gpu_rng.gen_uniform((x_len, n_samples), np.float32) - 0.5)
    #data = gpu_rng.gen_uniform((x_len, n_samples), np.float32) - 0.5
    if binary:
        data = data >= gp.zeros_like(data)
    return data


def generate_shifts(s_len, n_samples):
    shifts = gpu_rng.gen_uniform((n_samples,), np.float32)
    shifts = shifts * (s_len - 0.01)
    #print "interm:", shifts
    shifts = shifts.astype(np.uint32)

    shifts_hot = gp.empty((s_len, n_samples), np.float32)
    threads_per_block = 32
    n_blocks = int(math.ceil(n_samples / threads_per_block))
    gpu_shift_to_hot(shifts, shifts_hot,
                     np.uint32(shifts_hot.strides[0]/4), 
                     np.uint32(shifts_hot.strides[1]/4),
                     np.uint32(s_len), np.uint32(n_samples),
                     block=(threads_per_block, 1,1), grid=(n_blocks, 1))

    return shifts, shifts_hot


def generate_shifted(data, shifts):
    x_len = data.shape[0]
    n_samples = data.shape[1]
    assert len(shifts.shape) == 1 and shifts.shape[0] == n_samples

    shifted = gp.empty_like(data)
    threads_per_block = 32
    n_blocks = int(math.ceil(n_samples / threads_per_block))
    gpu_shift(data, shifts, shifted,
              np.uint32(data.strides[0]/4), np.uint32(data.strides[1]/4),
              np.uint32(x_len), np.uint32(n_samples),
              block=(threads_per_block, 1,1), grid=(n_blocks, 1))

    return shifted


def generate_data(x_len, s_len, n_samples, binary=False):
    data = generate_random_data(x_len, n_samples, binary=binary)
    shifts, shifts_hot = generate_shifts(s_len, n_samples)
    shifted = generate_shifted(data, shifts)
    return (gpuarray_to_garray(data), 
            gpuarray_to_garray(shifts_hot), 
            gpuarray_to_garray(shifted))


def generate_id_data(x_len, n_samples, binary=False):
    data = generate_random_data(x_len, n_samples, binary=binary)
    return gpuarray_to_garray(data), gpuarray_to_garray(data)
