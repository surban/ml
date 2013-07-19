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


def dft_weights(N):
    n = np.arange(N)[np.newaxis, :]
    k = np.arange(N)[:, np.newaxis]   
    w = np.exp(-2j*np.pi*k*n/N)
    return w.real, w.imag

def idft_weights(N):
    n = np.arange(N)[np.newaxis, :]
    k = np.arange(N)[:, np.newaxis]   
    w = np.exp(2j*np.pi*k*n/N) / N
    return w.real, w.imag

