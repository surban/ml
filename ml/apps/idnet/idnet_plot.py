# -*- coding: utf-8 -*-

import ml.common.gpu
import ml.common.util

import numpy as np
import breze.util
import matplotlib.pyplot as plt

import ml.nn.id
from ml.common.complex import *
from ml.common.gpu import gather

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
    cfg, plot_dir = ml.common.util.standard_cfg(prepend_scriptname=False)

    # parameters
    ps = breze.util.ParameterSet(**ml.nn.id.FourierIdNet.parameter_shapes(cfg.x_len))
    ps.data[:] = post(np.load(plot_dir + "/result.npz")['ps'])

    # plot
    plt.figure()
    plot_all_weights(ps)
    plt.savefig(plot_dir + "/weights.pdf")

