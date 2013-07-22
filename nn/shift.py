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
    def parameter_shapes(x_len, max_shift):
        s_len = max_shift+1
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


def shifted(x, s):
    assert x.ndim == 1 and s.ndim == 1, "x and s must be vectors"
    assert len(s) <= len(x), "x must be at least as long as s"
    spos = np.nonzero(s)
    assert len(spos) == 1, "exactly one entry of s must be one"
    shift = spos[0]
    assert s[shift] == 1, "entries of s must be either 0 or 1"
    y = np.roll(x, shift)
    return y


def generate_data(x_len, max_shift, n_samples):
    assert max_shift <= x_len
    inputs = np.zeros((x_len, n_samples))
    shifts = np.zeros((max_shift+1, n_samples))
    targets = np.zeros((x_len, n_samples))

    for s in range(n_samples):
        inputs[:,s] = np.random.random((x_len,)) - 0.5
        shft = np.random.randint(0, max_shift)
        shifts[shft,s] = 1
        targets[:,s] = shifted(inputs[:,s], shifts[:,s])

    return inputs, shifts, targets






