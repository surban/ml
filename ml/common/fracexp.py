import numpy as np

import theano
import theano.tensor as T

from theano import gof
from theano.scalar.basic import ScalarOp
from complex import cmul

# log branch point
lbp = -0.5

# inner circle radius
r0 = 0.0001


def crecip(re, im):
    d = re**2 + im**2
    return re/d, -im/d

def np_clog(re, im):
    log_re = 0.5 * np.log(re**2 + im**2)
    log_im = np.arctan2(im, re)
    return log_re, log_im

def np_cexp(re, im):
    r = np.exp(re)
    exp_re = r * np.cos(im)
    exp_im = r * np.sin(im)
    return exp_re, exp_im

def np_blog(re, im):
    log_re = 0.5 * np.log(re**2 + im**2)
    log_im = np.arctan2(im, re)
    log_im[log_im < lbp] += 2 * np.pi
    return log_re, log_im

def np_cabs(re, im):
    return np.sqrt(re**2 + im**2)

# exp fixed point
CP_re = 1.1
CP_im = 0.0
for i in range(200):
    CP_re, CP_im = np_clog(CP_re, CP_im)
CPR_re, CPR_im = crecip(CP_re, CP_im)


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
            x_re, x_im = np_blog(re, im)
            y_re, y_im = self.impl(x_re, x_im)
            return cmul(CP_re, CP_im, y_re, y_im)

chi_scalar = ChiOp()
chi = T.Elemwise(chi_scalar)


class DChiOp(ScalarOp):
    nin = 2
    nout = 2

    def __init__(self):
        super(DChiOp, self).__init__(name='DChiOp')

    @staticmethod
    def output_types_preference(type_re, type_im):
        return type_re, type_im

    def impl(self, re, im):
        if np_cabs(re - CP_re, im - CP_im) <= r0:
            return 1., 0.
        else:
            r_re, r_im = crecip(re, im)
            z_re, z_im = cmul(CP_re, CP_im, r_re, r_im)
            x_re, x_im = np_blog(re, im)
            y_re, y_im = self.impl(x_re, x_im)
            return cmul(z_re, z_im, y_re, y_im)

dchi_scalar = DChiOp()
dchi = T.Elemwise(dchi_scalar)


class ChiInvOp(ScalarOp):
    nin = 2
    nout = 2

    def __init__(self):
        super(ChiInvOp, self).__init__(name='ChiInvOp')

    @staticmethod
    def output_types_preference(type_re, type_im):
        return type_re, type_im

    def impl(self, re, im):
        if np_cabs(re, im) <= r0:
            return re + CP_re, im + CP_im
        else:
            x_re, x_im = cmul(re, im, CPR_re, CPR_im)
            y_re, y_im = self.impl(x_re, x_im)
            return np_cexp(y_re, y_im)

chiinv_scalar = ChiInvOp()
chiinv = T.Elemwise(chiinv_scalar)


class DChiInvOp(ScalarOp):
    nin = 2
    nout = 2

    def __init__(self):
        super(ChiInvOp, self).__init__(name='DChiInvOp')

    @staticmethod
    def output_types_preference(type_re, type_im):
        return type_re, type_im

    def rec(self, re, im):
        if np_cabs(re, im) <= r0:
            return 1., 0., re + CP_re, im + CP_im
        else:
            x_re, x_im = cmul(re, im, CPR_re, CPR_im)
            d_re, d_im, ci_re, ci_im = self.rec(x_re, x_im)
            y_re, y_im = np_cexp(d_re, d_im)
            z_re, z_im = cmul(y_re, y_im, d_re, d_im)
            w_re, w_im = cmul(z_re, z_im, CPR_re, CPR_im)
            cix_re, cix_im = np_cexp(ci_re, ci_im)
            return w_re, w_im, cix_re, cix_im

    def impl(self, re, im):
        return self.rec(re, im)



if __name__ == '__main__':
    x_re, x_im = T.fvectors(['x_re', 'x_im'])
    chi_re, chi_im = chi(x_re, x_im)
    f = theano.function([x_re, x_im], [chi_re, chi_im])

    x_re_val = np.asarray([1, 2, 3], dtype='float32')
    x_im_val = np.asarray([10, 20, 30], dtype='float32')

    print f(x_re_val, x_im_val)

