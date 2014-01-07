# -*- coding: utf-8 -*-

import breze.util
import matplotlib.pyplot as plt

import common.gpu
import common.util
from nn.shift import FourierShiftNet
from common.complex import *
from common.util import floatx
from common.gpu import gather, post

np.set_printoptions(precision=3, suppress=True)

def plot_complex_weights(re, im):
    plt.plot(re, im, 'rx')
    #plt.xlabel('Re')
    #plt.ylabel('Im')
    plt.axis('equal')
    #plt.gca().set_aspect('equal')

def plot_all_weights(ps):
    plt.subplot(3,2,1)
    plot_complex_weights(gather(ps['yhat_to_y_re']), gather(ps['yhat_to_y_im']))
    plt.title('yhat_to_y')

    plt.subplot(3,2,3)
    plot_complex_weights(gather(ps['Xhat_to_Yhat_re']), gather(ps['Xhat_to_Yhat_im']))
    plt.title('Xhat_to_Yhat')

    plt.subplot(3,2,4)
    plot_complex_weights(gather(ps['Shat_to_Yhat_re']), gather(ps['Shat_to_Yhat_im']))
    plt.title('Shat_to_Yhat')

    plt.subplot(3,2,5)
    plot_complex_weights(gather(ps['x_to_xhat_re']), gather(ps['x_to_xhat_im']))
    plt.title('x_to_xhat')

    plt.subplot(3,2,6)
    plot_complex_weights(gather(ps['s_to_shat_re']), gather(ps['s_to_shat_im']))
    plt.title('s_to_shat')

    plt.tight_layout()


if __name__ == '__main__':
    # hyperparameters
    cfg, plot_dir = common.util.standard_cfg(prepend_scriptname=False)

    # parameters
    ps = breze.util.ParameterSet(**FourierShiftNet.parameter_shapes(cfg.x_len, cfg.s_len))
    ps.data[:] = post(np.load(plot_dir + "/result.npz")['ps'])

    # plot
    plt.figure()
    plot_all_weights(ps)
    plt.savefig(plot_dir + "/weights.pdf")

