# -*- coding: utf-8 -*-
import sys

import common.gpu

import climin
import numpy as np
import gnumpy as gp
import theano
import theano.tensor as T
import breze.util
import matplotlib.pyplot as plt
import pdb

import common.util
import nn.id
from common.complex import *
from common.util import floatx
from common.gpu import gather, post, function
from math import floor, isnan  
from scipy.linalg import block_diag

np.set_printoptions(precision=3, suppress=True)

def plot_complex_weights(re, im):
    plt.plot(re, im, 'rx')
    #plt.xlabel('Re')
    #plt.ylabel('Im')
    plt.axis('equal')
    #plt.gca().set_aspect('equal')

def plot_all_weights(ps):
    plt.subplot(3,1,1)
    plot_complex_weights(gather(ps['yhat_to_y_re']), gather(ps['yhat_to_y_im']))
    plt.title('yhat_to_y')

    plt.subplot(3,1,2)
    plot_complex_weights(gather(ps['Xhat_to_Yhat_re']), gather(ps['Xhat_to_Yhat_im']))
    plt.title('Xhat_to_Yhat')

    plt.subplot(3,1,3)
    plot_complex_weights(gather(ps['x_to_xhat_re']), gather(ps['x_to_xhat_im']))
    plt.title('x_to_xhat')

    plt.tight_layout()


if __name__ == '__main__':
    # hyperparameters
    cfg, plot_dir = common.util.standard_cfg(prepend_scriptname=False)

    # parameters
    ps = breze.util.ParameterSet(**nn.id.FourierIdNet.parameter_shapes(cfg.x_len))
    ps.data[:] = post(np.load(plot_dir + "/result.npz")['ps'])

    # plot
    plt.figure()
    plot_all_weights(ps)
    plt.savefig(plot_dir + "/weights.pdf")

