import climin
import numpy as np
import gnumpy as gp
import theano
import theano.tensor as T
import breze.util

import common.util

from common.util import floatx
from common.gpu import function
from common.complex import *
from math import floor



w_re = T.matrix('w_re')
w_im = T.matrix('w_im')
x_re = T.matrix('x_re')
x_im = T.matrix('x_im')

w = np.random.random((3,3)) + np.random.random((3,3))*1j
x = np.random.random((3,2)) + np.random.random((3,2))*1j


def test_dft():
    print "DFT:"

    w_re, w_im = dft_weights(x.shape[0])
    iw_re, iw_im = idft_weights(x.shape[0])

    print "x:"
    print x

    x_re = x
    x_im = x*0

    dft_re, dft_im = np_cdot(w_re, w_im, x_re, x_im)
    print "DFT[x]:"
    print "Re: ", dft_re
    print "Im: ", dft_im

    idft_re, idft_im = np_cdot(iw_re, iw_im, dft_re, dft_im)
    print "DFT^-1[DFT[x]]:"
    print "Re: ", idft_re
    print "Im: ", idft_im

    assert np.linalg.norm(idft_re-x_re) < 0.001


def test_dot():
    print "=================================="
    print "Complex dot product:"

    y = np.dot(w,x)

    f_cdot = function([w_re, w_im, x_re, x_im], cdot(w_re, w_im, x_re, x_im))
    t_y_re, t_y_im = f_cdot(w.real, w.imag, x.real, x.imag)
    t_y = t_y_re + t_y_im*1j

    print "NumPy result:"
    print y

    print "Theano result:"
    print t_y
    assert np.linalg.norm(y-t_y) < 0.001


def test_log():
    print "=================================="
    print "Complex logarithm:"

    y = np.log(x)

    f_log = function([x_re, x_im], clog(x_re, x_im))
    t_y_re, t_y_im = f_log(x.real, x.imag)
    t_y = t_y_re + t_y_im*1j

    print "NumPy result:"
    print y

    print "Theano result:"
    print t_y
    assert np.linalg.norm(y-t_y) < 0.001


def test_exp():
    print "=================================="
    print "Complex exponential:"

    y = np.exp(x)

    f_exp = function([x_re, x_im], cexp(x_re, x_im))
    t_y_re, t_y_im = f_exp(x.real, x.imag)
    t_y = t_y_re + t_y_im*1j

    print "NumPy result:"
    print y

    print "Theano result:"
    print t_y
    assert np.linalg.norm(y-t_y) < 0.001




