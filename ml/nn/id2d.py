from __future__ import division

import theano
import theano.tensor as T
import numpy as np

from ml.common.complex import cdot, clog, cexp, dft_weights, idft_weights, dft_weights_2d, idft_weights_2d


class FourierIdNet2D(object):

    def __init__(self, width, height,
                 x_to_xhat_re, x_to_xhat_im,
                 Xhat_to_Yhat_re, Xhat_to_Yhat_im,
                 yhat_to_y_re, yhat_to_y_im):
        self.width = width
        self.height = height
        self.x_to_xhat_re = x_to_xhat_re
        self.x_to_xhat_im = x_to_xhat_im
        self.Xhat_to_Yhat_re = Xhat_to_Yhat_re
        self.Xhat_to_Yhat_im = Xhat_to_Yhat_im
        self.yhat_to_y_re = yhat_to_y_re
        self.yhat_to_y_im = yhat_to_y_im

    @staticmethod
    def parameter_shapes(width, height):
        x_len = width*height
        return {'x_to_xhat_re': (x_len, x_len),
                'x_to_xhat_im': (x_len, x_len),
                'Xhat_to_Yhat_re': (x_len, x_len),
                'Xhat_to_Yhat_im': (x_len, x_len),
                'yhat_to_y_re': (x_len, x_len),
                'yhat_to_y_im': (x_len, x_len)}

    def output(self, x, output_y_im=False):
        # DFT layer of x: xhat
        xhat_re, xhat_im = cdot(self.x_to_xhat_re, self.x_to_xhat_im,
                                x, T.zeros_like(x))

        # log layer of xhat: Xhat
        Xhat_re, Xhat_im = clog(xhat_re, xhat_im)

        # multiplication of Xhat and Shat in log space: Yhat
        Yhat_re, Yhat_im = cdot(self.Xhat_to_Yhat_re, self.Xhat_to_Yhat_im,
                                Xhat_re, Xhat_im)

        # exp layer of Yhat: yhat
        yhat_re, yhat_im = cexp(Yhat_re, Yhat_im)

        # inverse DFT layer of yhat: y
        y_re, y_im = cdot(self.yhat_to_y_re, self.yhat_to_y_im,
                          yhat_re, yhat_im)

        if not output_y_im:
            # output is real part of y
            return y_re
        else:
            return y_re, y_im

    def optimal_weights(self):
        # DFT layer of x: xhat
        x_to_xhat_re,  x_to_xhat_im = dft_weights_2d(self.width, self.height)

        # multiplication of Yhat in log space: Yhat
        Xhat_to_Yhat = np.eye(self.width * self.height)
        Xhat_to_Yhat_re = np.real(Xhat_to_Yhat)
        Xhat_to_Yhat_im = np.imag(Xhat_to_Yhat)

        # inverse DFT layer of yhat: y
        yhat_to_y_re, yhat_to_y_im = idft_weights_2d(self.width, self.height)

        return (x_to_xhat_re, x_to_xhat_im,
                Xhat_to_Yhat_re, Xhat_to_Yhat_im,
                yhat_to_y_re, yhat_to_y_im)

