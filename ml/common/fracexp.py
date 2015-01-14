import numpy as np

import theano
import theano.tensor as T

from theano import gof
from theano.scalar.basic import ScalarOp
from complex import cmul

def np_clog(re, im):
    log_re = 0.5 * np.log(re**2 + im**2)
    log_im = np.arctan2(im, re)
    return log_re, log_im

def np_cabs(re, im):
    return np.sqrt(re**2 + im**2)

CP_re = 1.1
CP_im = 0.0
for i in range(200):
    CP_re, CP_im = np_clog(CP_re, CP_im)
r0 = 0.0001



class ChiOp(ScalarOp):

    nin = 2
    nout = 2

    def __init__(self):
        super(ChiOp, self).__init__(name='ChiOp')

    @staticmethod
    def output_types_preference(type_re, type_im):
        return type_re, type_im

    def impl(self, re, im):
        if np_cabs(re - CP_re, im - CP_im) <= r0:
            return re - CP_re, im - CP_im
        else:
            x_re, x_im = np_clog(re, im)
            y_re, y_im = self.impl(x_re, x_im)
            return cmul(CP_re, CP_im, y_re, y_im)


chi_scalar = ChiOp()
chi = T.Elemwise(chi_scalar)

if __name__ == '__main__':
    x_re, x_im = T.fvectors(['x_re', 'x_im'])
    chi_re, chi_im = chi(x_re, x_im)
    f = theano.function([x_re, x_im], [chi_re, chi_im])

    x_re_val = np.asarray([1, 2, 3], dtype='float32')
    x_im_val = np.asarray([10, 20, 30], dtype='float32')

    print f(x_re_val, x_im_val)

