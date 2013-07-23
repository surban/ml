from __future__ import division

import theano
import theano.tensor as T
import numpy as np

from common.complex import cdot, clog, cexp

class FourierShiftNet(object):

    def __init__(self, 
                 x_to_xhat_re, x_to_xhat_im,
                 s_to_shat_re, s_to_shat_im,
                 Xhat_to_Yhat_re, Xhat_to_Yhat_im, 
                 Shat_to_Yhat_re, Shat_to_Yhat_im, 
                 yhat_to_y_re, yhat_to_y_im):
        self.x_to_xhat_re = x_to_xhat_re
        self.x_to_xhat_im = x_to_xhat_im
        self.s_to_shat_re = s_to_shat_re
        self.s_to_shat_im = s_to_shat_im
        self.Xhat_to_Yhat_re = Xhat_to_Yhat_re
        self.Xhat_to_Yhat_im = Xhat_to_Yhat_im
        self.Shat_to_Yhat_re = Shat_to_Yhat_re
        self.Shat_to_Yhat_im = Shat_to_Yhat_im
        self.yhat_to_y_re = yhat_to_y_re
        self.yhat_to_y_im = yhat_to_y_im

    @staticmethod
    def parameter_shapes(x_len, s_len):
        return {'x_to_xhat_re': (x_len, x_len),
                'x_to_xhat_im': (x_len, x_len),
                's_to_shat_re': (s_len, s_len),
                's_to_shat_im': (s_len, s_len),
                'Xhat_to_Yhat_re': (x_len, x_len),
                'Xhat_to_Yhat_im': (x_len, x_len),
                'Shat_to_Yhat_re': (x_len, s_len),
                'Shat_to_Yhat_im': (x_len, s_len),
                'yhat_to_y_re': (x_len, x_len),
                'yhat_to_y_im': (x_len, x_len)}


    def output(self, x, s):
        # DFT layer of x: xhat
        xhat_re, xhat_im = cdot(self.x_to_xhat_re, self.x_to_xhat_im,
                                x, T.zeros_like(x))

        # DFT layer of s: shat
        shat_re, shat_im = cdot(self.s_to_shat_re, self.s_to_shat_im,
                                s, T.zeros_like(s))

        # log layer of xhat: Xhat
        Xhat_re, Xhat_im = clog(xhat_re, xhat_im)

        # log layer of shat: Shat
        Shat_re, Shat_im = clog(shat_re, shat_im)

        # multiplication of Xhat and Shat in log space: Yhat
        Yhat1_re, Yhat1_im = cdot(self.Xhat_to_Yhat_re, self.Xhat_to_Yhat_im,
                                  Xhat_re, Xhat_im)
        Yhat2_re, Yhat2_im = cdot(self.Shat_to_Yhat_re, self.Shat_to_Yhat_im,
                                  Shat_re, Shat_im)
        Yhat_re = Yhat1_re + Yhat2_re
        Yhat_im = Yhat2_re + Yhat2_im

        # exp layer of Yhat: yhat
        yhat_re, yhat_im = cexp(Yhat_re, Yhat_im)

        # inverse DFT layer of yhat: y
        y_re, y_im = cdot(self.yhat_to_y_re, self.yhat_to_y_im,
                          yhat_re, yhat_im)

        # output is real part of y
        return y_re

    @staticmethod
    def optimal_weights(x_len, s_len):
        # DFT layer of x: xhat
        x_to_xhat = fourier_weights(x_len)
        x_to_xhat_re = np.real(x_to_xhat)
        x_to_xhat_im = np.imag(x_to_xhat)

        # DFT layer of s: shat
        s_to_shat = fourier_weights(s_len)
        s_to_shat_re = np.real(s_to_shat)
        s_to_shat_im = np.imag(s_to_shat)

        # multiplication of Yhat and Shat in log space: Yhat
        Xhat_to_Yhat = np.eye(x_len)
        Xhat_to_Yhat_re = np.real(Xhat_to_Yhat)
        Xhat_to_Yhat_im = np.imag(Xhat_to_Yhat)

        Shat_to_Yhat = np.eye(x_len, s_len)
        Shat_to_Yhat_re = np.real(Shat_to_Yhat)
        Shat_to_Yhat_im = np.imag(Shat_to_Yhat)

        # inverse DFT layer of yhat y:
        yhat_to_y = inverse_fourier_weights(x_len)
        yhat_to_y_re = np.real(yhat_to_y)
        yhat_to_y_im = np.imag(yhat_to_y)

        return (x_to_xhat_re, x_to_xhat_im,
                s_to_shat_re, s_to_shat_im,
                Xhat_to_Yhat_re, Xhat_to_Yhat_im,
                Shat_to_Yhat_re, Shat_to_Yhat_im,
                yhat_to_y_re, yhat_to_y_im)



def fourier_weights(N):
    n = np.arange(0, N)[np.newaxis, :]
    k = np.arange(0, N)[:, np.newaxis]
    W = np.exp(-2j*np.pi*k*n/N)
    return W

def inverse_fourier_weights(N):
    k = np.arange(0, N)[np.newaxis, :]
    n = np.arange(0, N)[:, np.newaxis]
    W = 1/N * np.exp(-2j*np.pi*k*n/N)
    return W


def shifted(x, s):
    assert x.ndim == 1 and s.ndim == 1, "x and s must be vectors"
    assert len(s) <= len(x), "x must be at least as long as s"
    spos = np.nonzero(s)[0]
    assert len(spos) <= 1, "zero or one entries of s must be one"
    if len(spos) == 0:
        shift = 0
    else:
        assert s[spos[0]] == 1, "entries of s must be either 0 or 1"
        shift = spos[0]+1
    y = np.roll(x, shift)
    return y


def generate_data(x_len, s_len, n_samples, binary=False):
    assert s_len <= x_len
    inputs = np.zeros((x_len, n_samples))
    shifts = np.zeros((s_len, n_samples))
    targets = np.zeros((x_len, n_samples))

    for s in range(n_samples):
        if binary:
            inputs[:,s] = np.random.randint(0, 2, (x_len,))
        else:
            inputs[:,s] = np.random.random((x_len,)) - 0.5
        #shft = np.random.randint(-1, s_len)
        shft = np.random.randint(0, s_len)
        if shft != -1:
            shifts[shft,s] = 1
        targets[:,s] = shifted(inputs[:,s], shifts[:,s])

    return inputs, shifts, targets






