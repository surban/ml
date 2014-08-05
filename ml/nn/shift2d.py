from __future__ import division

import theano
import theano.tensor as T
import numpy as np

from ml.common.complex import cdot, clog, cexp, dft_weights, idft_weights, dft_weights_2d, idft_weights_2d


def shift_amounts(shifts):
    nz = np.transpose(np.nonzero(shifts))
    amounts = np.zeros((shifts.shape[1],))
    amounts[nz[:,1]] = nz[:,0]
    return amounts


class FourierShiftNet2D(object):

    def __init__(self, width, height, arch,
                 x_to_xhat_re, x_to_xhat_im,
                 s_to_shat_re, s_to_shat_im,
                 Xhat_to_Yhat_re, Xhat_to_Yhat_im, 
                 Shat_to_Yhat_re, Shat_to_Yhat_im, 
                 yhat_to_y_re, yhat_to_y_im):
        assert arch in ['conv', 'comp']
        self.width = width
        self.height = height
        self.arch = arch
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
    def parameter_shapes(width, height, arch):
        x_len = width * height
        if arch == 'conv':
            s_len = x_len
        elif arch == 'comp':
            s_len = 2
        else:
            assert False
        return {'x_to_xhat_re': (x_len, x_len),
                'x_to_xhat_im': (x_len, x_len),
                's_to_shat_re': (x_len, s_len),
                's_to_shat_im': (x_len, s_len),
                'Xhat_to_Yhat_re': (x_len, x_len),
                'Xhat_to_Yhat_im': (x_len, x_len),
                'Shat_to_Yhat_re': (x_len, x_len),
                'Shat_to_Yhat_im': (x_len, x_len),
                'yhat_to_y_re': (x_len, x_len),
                'yhat_to_y_im': (x_len, x_len)}

    def output(self, x, s):
        # DFT layer of x: xhat
        xhat_re, xhat_im = cdot(self.x_to_xhat_re, self.x_to_xhat_im,
                                x, T.zeros_like(x))

        # DFT layer of s / shift calculation: shat
        shat_re, shat_im = cdot(self.s_to_shat_re, self.s_to_shat_im,
                                s, T.zeros_like(s))

        # log layer of xhat: Xhat
        Xhat_re, Xhat_im = clog(xhat_re, xhat_im)

        if self.arch == 'conv':
            # log layer of shat: Shat
            Shat_re, Shat_im = clog(shat_re, shat_im)
        elif self.arch == 'comp':
            # linear transfer function for shat
            Shat_re, Shat_im = shat_re, shat_im
        else:
            assert False

        # multiplication of Xhat and Shat in log space: Yhat
        Yhat1_re, Yhat1_im = cdot(self.Xhat_to_Yhat_re, self.Xhat_to_Yhat_im,
                                  Xhat_re, Xhat_im)
        Yhat2_re, Yhat2_im = cdot(self.Shat_to_Yhat_re, self.Shat_to_Yhat_im,
                                  Shat_re, Shat_im)
        Yhat_re = Yhat1_re + Yhat2_re
        Yhat_im = Yhat1_im + Yhat2_im
        # Yhat_re = Yhat1_re
        # Yhat_im = Yhat1_im

        # exp layer of Yhat: yhat
        yhat_re, yhat_im = cexp(Yhat_re, Yhat_im)

        # inverse DFT layer of yhat: y
        y_re, y_im = cdot(self.yhat_to_y_re, self.yhat_to_y_im,
                          yhat_re, yhat_im)

        return y_re, y_im

    def optimal_shift_weights(self):
        # DFT layer of x: xhat
        x_to_xhat_re,  x_to_xhat_im = dft_weights_2d(self.width, self.height)

        if self.arch == 'conv':
            # DFT layer of s: shat
            s_to_shat_re, s_to_shat_im = dft_weights_2d(self.width, self.height)
        elif self.arch == 'comp':
            # shift calculation from s: shat
            xgrid, ygrid = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing='ij')
            fx = xgrid.flatten()[:, np.newaxis]
            fy = ygrid.flatten()[:, np.newaxis]
            s_to_shat_x = -2j*np.pi * fx / self.width
            s_to_shat_y = -2j*np.pi * fy / self.height
            s_to_shat = np.hstack((s_to_shat_x, s_to_shat_y))
            s_to_shat_re, s_to_shat_im = np.real(s_to_shat), np.imag(s_to_shat)
        else:
            assert False

        # multiplication of Yhat and Shat in log space: Yhat
        Xhat_to_Yhat = np.eye(self.width * self.height)
        Xhat_to_Yhat_re = np.real(Xhat_to_Yhat)
        Xhat_to_Yhat_im = np.imag(Xhat_to_Yhat)

        Shat_to_Yhat = np.eye(self.width * self.height)
        Shat_to_Yhat_re = np.real(Shat_to_Yhat)
        Shat_to_Yhat_im = np.imag(Shat_to_Yhat)

        # inverse DFT layer of yhat y:
        yhat_to_y_re, yhat_to_y_im = idft_weights_2d(self.width, self.height)

        return (x_to_xhat_re, x_to_xhat_im,
                s_to_shat_re, s_to_shat_im,
                Xhat_to_Yhat_re, Xhat_to_Yhat_im,
                Shat_to_Yhat_re, Shat_to_Yhat_im,
                yhat_to_y_re, yhat_to_y_im)

