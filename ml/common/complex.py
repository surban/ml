from __future__ import division

import theano.tensor as T
import numpy as np


def clog(re, im):
    log_re = 0.5 * T.log(re**2 + im**2)
    log_im = T.arctan2(im, re)
    return log_re, log_im


def cexp(re, im):
    exp_re = T.exp(re) * T.cos(im)
    exp_im = T.exp(re) * T.sin(im)
    return exp_re, exp_im


def cdot(a_re, a_im, b_re, b_im):
    """Returns a*b where a and b are complex numbers."""
    dot_re = T.dot(a_re, b_re) - T.dot(a_im, b_im)
    dot_im = T.dot(a_im, b_re) + T.dot(a_re, b_im)
    return dot_re, dot_im


def np_cdot(a_re, a_im, b_re, b_im):
    dot_re = np.dot(a_re, b_re) - np.dot(a_im, b_im)
    dot_im = np.dot(a_im, b_re) + np.dot(a_re, b_im)
    return dot_re, dot_im


def cmul(a_re, a_im, b_re, b_im):
    mul_re = a_re*b_re - a_im*b_im
    mul_im = a_im*b_re + a_re*b_im
    return mul_re, mul_im


def dft_weights_complex(N):
    n = np.arange(N)[np.newaxis, :]
    k = np.arange(N)[:, np.newaxis]   
    w = np.exp(-2j*np.pi*k*n/N)
    return w


def dft_weights_complex_2d(N, M):
    Ngrid, Mgrid = np.meshgrid(np.arange(N), np.arange(M), indexing='ij')
    fx = Ngrid.flatten()[np.newaxis, :]
    fy = Mgrid.flatten()[np.newaxis, :]
    px = Ngrid.flatten()[:, np.newaxis]
    py = Mgrid.flatten()[:, np.newaxis]
    w = np.exp(-2j*np.pi * (px*fx/N + py*fy/M))
    return w


def dft_weights(N):
    w = dft_weights_complex(N)
    return w.real, w.imag


def dft_weights_2d(N, M):
    w = dft_weights_complex_2d(N, M)
    return w.real, w.imag


def idft_weights_complex(N):
    n = np.arange(N)[np.newaxis, :]
    k = np.arange(N)[:, np.newaxis]   
    w = np.exp(2j*np.pi*k*n/N) / N
    return w


def idft_weights_complex_2d(N, M):
    Ngrid, Mgrid = np.meshgrid(np.arange(N), np.arange(M), indexing='ij')
    fx = Ngrid.flatten()[np.newaxis, :]
    fy = Mgrid.flatten()[np.newaxis, :]
    px = Ngrid.flatten()[:, np.newaxis]
    py = Mgrid.flatten()[:, np.newaxis]
    w = np.exp(2j*np.pi * (px*fx/N + py*fy/M)) / N / M
    return w


def idft_weights(N):
    w = idft_weights_complex(N)
    return w.real, w.imag


def idft_weights_2d(N, M):
    w = idft_weights_complex_2d(N, M)
    return w.real, w.imag

