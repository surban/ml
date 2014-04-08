import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import math
import ml.common.progress as progress

import time

max_force = 20
max_skin = 2


def multistep_gradient_dense(predictor_with_grad, force, skin, valid):
    """Calculates the gradient of the error function over multi-step prediction.
    Inputs have the form: force[step, sample], skin[step, sample], valid[step, sample]
    Output has the form:  grad[weight]
    """

    # skin[step, sample]
    # skin_p[step, sample]
    # pred_wrt_prev[sample]
    # pred_wrt_weights[weight, sample]
    # skin_wrt_weights[weight, sample]
    # error_wrt_weights[weight, sample]
    # x[0=force/1=skin, sample]

    n_steps = force.shape[0]
    n_samples = force.shape[1]

    skin_p = np.zeros((n_steps, n_samples))
    skin_p[0, :] = skin[0, :]
    prev_skin_wrt_weights = None
    total_error_wrt_weights = None

    for step in range(1, n_steps):
        # progress.status(step-1, n_steps, "multistep_gradient")

        # clamp to range to avoid under-/overflows
        skin_p[step-1, skin[step-1, :] < 0] = 0
        skin_p[step-1, skin[step-1, :] > max_skin] = max_skin

        x = np.vstack((force[step, :], skin_p[step-1, :]))
        skin_p[step, :], pred_wrt_prev_all, pred_wrt_weights = predictor_with_grad(x)
        pred_wrt_prev = pred_wrt_prev_all[1, :]
        pred_wrt_weights = pred_wrt_weights.toarray()

        # print "pred_wrt_prev: ", pred_wrt_prev.shape
        # print "pred_wrt_weights: ", pred_wrt_weights.shape

        if step == 1:
            skin_wrt_weights = pred_wrt_weights
        else:
            skin_wrt_weights = pred_wrt_weights + pred_wrt_prev * prev_skin_wrt_weights

        error_wrt_weights = (skin_p[step, :] - skin[step, :]) * skin_wrt_weights
        error_wrt_weights[~valid[step, :]] = 0

        # print "error_wrt_weights: ", error_wrt_weights.shape

        if step == 1:
            total_error_wrt_weights = error_wrt_weights
        else:
            total_error_wrt_weights += error_wrt_weights

        prev_skin_wrt_weights = skin_wrt_weights

    # progress.done()
    grad = np.sum(total_error_wrt_weights, axis=1)
    return grad


def multistep_gradient_own_sparse(tr, force, skin, valid):
    """Calculates the gradient of the error function over multi-step prediction.
    Inputs have the form: force[step, sample], skin[step, sample], valid[step, sample]
    Output has the form:  grad[weight]
    """

    # skin[step, sample]
    # skin_p[step, sample]
    # weight_idx[step, no, sample]
    # wrt_weights_val[step, no, sample]
    # wrt_prev[step, sample]
    # grad[weight, sample]
    # x[0=force/1=skin, sample]

    n_steps = force.shape[0]
    n_samples = force.shape[1]

    skin_p = np.zeros((n_steps, n_samples))
    skin_p[0, :] = skin[0, :]

    wrt_prev = np.zeros((n_steps, n_samples))
    weight_idx = np.zeros((n_steps, tr.weights_per_sample, n_samples), dtype='int')
    wrt_weights_val = np.zeros(weight_idx.shape)

    # obtain gradients for each step
    for step in range(1, n_steps):
        # clamp to range to avoid table under-/overflows
        skin_p[step-1, skin[step-1, :] < 0] = 0
        skin_p[step-1, skin[step-1, :] > max_skin] = max_skin

        # predict and obtain table gradient w.r.t. previous skin value and weights
        x = np.vstack((force[step, :], skin_p[step-1, :]))
        skin_p[step, :], pred_wrt_prev_all, pred_wrt_weights_idx, pred_wrt_weights_smpl, pred_wrt_weights_fac = \
            tr.predict_and_gradient_indices(x)

        # store
        weight_idx[step, :, :] = pred_wrt_weights_idx
        wrt_weights_val[step, :, :] = pred_wrt_weights_fac
        wrt_prev[step, :] = pred_wrt_prev_all[1, :]

    # build dense gradient representation
    grad = np.zeros((tr.n_weights, n_samples))
    smpl_idx = np.tile(np.arange(n_samples), (tr.weights_per_sample, 1))

    # sum error over all steps
    for step in range(1, n_steps):
        # backpropagate error through time
        wrt_weights_val[0:step, :, :] *= wrt_prev[step]

        # convert sparse to dense representation
        efac = skin_p[step, :] - skin[step, :]
        efac[~valid[step, :]] = 0
        err_wrt_weights_val = wrt_weights_val * efac
        for bstep in range(1, step+1):
            grad[weight_idx[bstep, :, :].flatten(), smpl_idx.flatten()] += err_wrt_weights_val[bstep, :, :].flatten()

    # sum gradient over all samples
    return np.sum(grad, axis=1)






