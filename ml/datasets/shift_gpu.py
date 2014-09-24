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


# random number generator
gpu_rng = gprng.XORWOWRandomNumberGenerator()

# Kernels
cuda_mod = SourceModule(r"""
#include <stdio.h>

const bool debug = false;

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

__global__ void shift_to_hot_2d(uint32 *x_shifts, uint32 *y_shifts, float *hot,
                                uint32 hot_st0, uint32 hot_st1,
                                uint32 width, uint32 height, uint32 n_samples)
{
    uint32 smpl = threadIdx.x + blockIdx.x*blockDim.x;
    if (smpl >= n_samples)
        return;

    uint32 x_shft = x_shifts[smpl];
    uint32 y_shft = y_shifts[smpl];

    for (uint32 x=0; x < width; x++)
    {
        for (uint32 y=0; y < height; y++)
        {
            uint32 pos = y * width + x;
            hot[elem(hot_st0, hot_st1, pos, smpl)] = (x == x_shft && y == y_shft) ? 1.0f : 0.0f;
        }
    }
}

__global__ void shift(float *data, uint32 *shifts, float *shifted,
                      uint32 st0, uint32 st1,
                      uint32 data_len, uint32 n_samples)
{
    if (debug)
    {
        printf("data: %p   shifts: %p   shifted: %p\n", data, shifts, shifted);
        printf("st0: %u    st1: %u\n", st0, st1);
        printf("data_len: %u   n_samples: %u\n",
                data_len, n_samples);
    }

    uint32 smpl = threadIdx.x + blockIdx.x*blockDim.x;
    if (smpl >= n_samples)
        return;
    uint32 shft = shifts[smpl];

    if (debug)
    {
        printf("smpl: %u\n", smpl);
        printf("working with shift %u\n", shft);
    }

    for (uint32 src=0; src < data_len; src++)
    {
        uint32 trgt = src + shft;
        if (trgt >= data_len)
            trgt -= data_len;

        shifted[elem(st0, st1, trgt, smpl)] = data[elem(st0, st1, src, smpl)];
    }
}

__global__ void shift_2d(float *data, uint32 *x_shifts, uint32 *y_shifts, float *shifted,
                         uint32 st0, uint32 st1,
                         uint32 width, uint32 height, uint32 n_samples)
{
    uint32 smpl = threadIdx.x + blockIdx.x*blockDim.x;
    if (smpl >= n_samples)
        return;

    uint32 x_shft = x_shifts[smpl];
    uint32 y_shft = y_shifts[smpl];

    for (uint32 x_src=0; x_src < width; x_src++)
    {
        for (uint32 y_src=0; y_src < height; y_src++)
        {
            uint32 x_trgt = x_src + x_shft;
            if (x_trgt >= width)
                x_trgt -= width;

            uint32 y_trgt = y_src + y_shft;
            if (y_trgt >= height)
                y_trgt -= height;

            uint32 src = y_src * width + x_src;
            uint32 trgt = y_trgt * width + x_trgt;

            shifted[elem(st0, st1, trgt, smpl)] = data[elem(st0, st1, src, smpl)];
        }
    }
}

__global__ void vstack(uint32 *x, uint32 *y, float *stacked,
                       uint32 stacked_st0, uint32 stacked_st1,
                       uint32 length)
{
    uint32 smpl = threadIdx.x + blockIdx.x*blockDim.x;
    if (smpl >= length)
        return;

    stacked[elem(stacked_st0, stacked_st1, 0, smpl)] = x[smpl];
    stacked[elem(stacked_st0, stacked_st1, 1, smpl)] = y[smpl];
}
""")
gpu_shift = cuda_mod.get_function('shift')
gpu_shift_2d = cuda_mod.get_function('shift_2d')
gpu_shift_to_hot = cuda_mod.get_function('shift_to_hot')
gpu_shift_to_hot_2d = cuda_mod.get_function('shift_to_hot_2d')
gpu_vstack = cuda_mod.get_function('vstack')


def generate_random_data(x_len, n_samples, binary=False):
    data = 4.0 * (gpu_rng.gen_uniform((x_len, n_samples), np.float32) - 0.5)
    #data = gpu_rng.gen_uniform((x_len, n_samples), np.float32) - 0.5
    if binary:
        data = data >= gp.zeros_like(data)
    return data


def generate_shifts(s_len, n_samples):
    shifts = gpu_rng.gen_uniform((n_samples,), np.float32) * (s_len - 0.01)
    shifts = shifts.astype(np.uint32)

    shifts_hot = gp.empty((s_len, n_samples), np.float32)
    threads_per_block = 32
    n_blocks = int(math.ceil(n_samples / threads_per_block))
    gpu_shift_to_hot(shifts, shifts_hot,
                     np.uint32(shifts_hot.strides[0]/4), 
                     np.uint32(shifts_hot.strides[1]/4),
                     np.uint32(s_len), np.uint32(n_samples),
                     block=(threads_per_block, 1, 1), grid=(n_blocks, 1))

    return shifts, shifts_hot


def generate_shifts_2d(width, height, n_samples, with_hot=False):
    x_shifts = gpu_rng.gen_uniform((n_samples,), np.float32) * (width - 0.01)
    x_shifts = x_shifts.astype(np.uint32)

    y_shifts = gpu_rng.gen_uniform((n_samples,), np.float32) * (height - 0.01)
    y_shifts = y_shifts.astype(np.uint32)

    if with_hot:
        shifts_hot = gp.empty((width * height, n_samples), np.float32)
        threads_per_block = 32
        n_blocks = int(math.ceil(n_samples / threads_per_block))
        gpu_shift_to_hot_2d(x_shifts, y_shifts, shifts_hot,
                            np.uint32(shifts_hot.strides[0]/4),
                            np.uint32(shifts_hot.strides[1]/4),
                            np.uint32(width), np.uint32(height), np.uint32(n_samples),
                            block=(threads_per_block, 1, 1), grid=(n_blocks, 1))
        return x_shifts, y_shifts, shifts_hot
    else:
        shifts = gp.empty((2, n_samples), np.float32)
        threads_per_block = 32
        n_blocks = int(math.ceil(n_samples / threads_per_block))
        gpu_vstack(y_shifts, x_shifts, shifts,
                   np.uint32(shifts.strides[0]/4), np.uint32(shifts.strides[1]/4),
                   np.uint32(n_samples),
                   block=(threads_per_block, 1, 1), grid=(n_blocks, 1))
        return x_shifts, y_shifts, shifts


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


def generate_shifted_2d(data, x_shifts, y_shifts, width, height):
    n_samples = data.shape[1]

    shifted = gp.empty_like(data)
    threads_per_block = 32
    n_blocks = int(math.ceil(n_samples / threads_per_block))
    gpu_shift_2d(data, x_shifts, y_shifts, shifted,
                 np.uint32(data.strides[0]/4), np.uint32(data.strides[1]/4),
                 np.uint32(width), np.uint32(height), np.uint32(n_samples),
                 block=(threads_per_block, 1, 1), grid=(n_blocks, 1))

    return shifted


def generate_data(x_len, s_len, n_samples, binary=False):
    data = generate_random_data(x_len, n_samples, binary=binary)
    shifts, shifts_hot = generate_shifts(s_len, n_samples)
    shifted = generate_shifted(data, shifts)
    return (gpuarray_to_garray(data), 
            gpuarray_to_garray(shifts_hot), 
            gpuarray_to_garray(shifted))


def generate_data_2d(width, height, n_samples, binary=False, with_hot=False, force_id=True):
    data = generate_random_data(width * height, n_samples, binary=binary)
    x_shifts, y_shifts, shifts = generate_shifts_2d(width, height, n_samples, with_hot=with_hot)
    if force_id:
        shifted = data
    else:
        shifted = generate_shifted_2d(data, x_shifts, y_shifts, width, height)
    return (gpuarray_to_garray(data),
            gpuarray_to_garray(shifts),
            gpuarray_to_garray(shifted))


def generate_id_data(x_len, n_samples, binary=False):
    data = generate_random_data(x_len, n_samples, binary=binary)
    return gpuarray_to_garray(data), gpuarray_to_garray(data)


def generate_id_data_2d(width, height, n_samples, binary=False):
    return generate_id_data(width*height, n_samples, binary)
