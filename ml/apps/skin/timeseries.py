import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import math
import random
import ml.common.progress as progress

max_force = 20
max_skin = 2


def plot_phase(data, *args):
    plt.plot(data[0, :], data[1, :], *args)
    plt.xlabel("force [N]")
    plt.ylabel("skin [V]")
    plt.xlim(0, max_force)
    plt.ylim(0, max_skin)


def plot_multicurve(force, skin, valid, *args):
    n_curves = force.shape[1]
    assert skin.shape[1] == n_curves
    assert valid.shape[1] == n_curves

    height = int(math.sqrt(n_curves))
    width = int(math.ceil(n_curves / float(height)))

    for c in range(n_curves):
        plt.subplot(height, width, c+1)
        valid_to = np.where(valid[:, c])[0][-1]
        plt.plot(force[0:valid_to, c], skin[0:valid_to, c], *args)
        plt.xlim(0, max_force)
        plt.ylim(0, max_skin)
        plt.title(str(c))

        if c % width == 0:
            plt.ylabel("skin [V]")
        if c / width == height-1:
            plt.xlabel("force [N]")


def plot_multicurve_time(force, skin, valid, skin_predicted=None, timestep=None):
    n_curves = force.shape[1]
    assert skin.shape[1] == n_curves
    assert valid.shape[1] == n_curves

    height = int(math.sqrt(n_curves))
    width = int(math.ceil(n_curves / float(height)))

    if timestep:
        max_time = force.shape[0] * timestep
    else:
        max_time = force.shape[0]

    for c in range(n_curves):
        plt.subplot(height, width, c+1)
        where_valid = np.where(valid[:, c])[0]
        if len(where_valid) == 0:
            valid_to = 0
        else:
            valid_to = where_valid[-1]

        if timestep:
            ts = np.linspace(0, valid_to * timestep, valid_to)
        else:
            ts = np.arange(valid_to)

        ax1 = plt.gca()
        ax1.plot(ts, force[0:valid_to, c], 'k')
        ax1.set_ylim(0, max_force)
        ax1.set_xlim(0, max_time)
        for tl in ax1.get_yticklabels():
            tl.set_color('k')
        if c % width == 0:
            ax1.set_ylabel("force [N]")
        if c / width == height-1:
            ax1.set_xlabel("time [s]")

        ax2 = ax1.twinx()
        ax2.plot(ts, skin[0:valid_to, c], 'b')
        if skin_predicted is not None:
            ax2.plot(ts, skin_predicted[0:valid_to, c], 'r')
        ax2.set_ylim(0, max_skin)
        ax2.set_xlim(0, max_time)
        for tl in ax2.get_yticklabels():
            tl.set_color('b')
        if (c+1) % width == 0:
            ax2.set_ylabel("skin [V]", color='b')

        plt.title(str(c))


def build_nextstep_data(ds, purpose, taxel, n_curves=None):
    """
    Returns an array of inputs and an array of targets for next step prediction.
    :type ds: ml.datasets.skin.SkinDataset
    """
    if not n_curves:
        n_curves = ds.record_count(purpose, taxel)
    else:
        assert n_curves <= ds.record_count(purpose, taxel)

    X = np.zeros((2, 0))
    Z = np.zeros((0, ))
    for i in range(n_curves):
        rec = ds.record(purpose, taxel, i)
        x = rec[:, 0:-1]
        z = rec[1, 1:]

        X = np.concatenate((X, x), axis=1)
        Z = np.concatenate((Z, z), axis=1)

    return X, Z


def build_nextstep_data_from_multicurve(force, skin_in, valid, skin_out=None):
    if skin_out is None:
        skin_out = skin_in

    n_curves = force.shape[1]
    ns_in = np.zeros((2, 0))
    ns_skin = np.zeros((0, ))

    for smpl in range(n_curves):
        last_valid = np.nonzero(valid[:, smpl])[0][-1]
        x = np.vstack((force[0:last_valid-1, smpl],
                       skin_in[0:last_valid-1, smpl]))
        z = skin_out[1:last_valid, smpl]

        ns_in = np.concatenate((ns_in, x), axis=1)
        ns_skin = np.concatenate((ns_skin, z), axis=1)

    return ns_in, ns_skin


def add_noise_to_nextstep_data(X, Z, ratio, sigma, max_skin):
    n_samples = X.shape[1]
    n_noisy = int(n_samples * ratio)

    sno = np.random.choice(n_samples, size=n_noisy, replace=False)
    noisy_X = X[:, sno]
    noisy_Z = Z[sno]

    noisy_X[1, :] += np.random.normal(loc=0, scale=sigma, size=n_noisy)
    noisy_X[1, noisy_X[1, :] < 0] = 0
    noisy_X[1, noisy_X[1, :] > max_skin] = max_skin

    return np.hstack((X, noisy_X)), np.hstack((Z, noisy_Z))


def build_multicurve(curves):
    """Combines a list of curves into a single matrix of the form:
    force[step, sample] and skin[step, sample]."""
    if not isinstance(curves, (list, tuple)):
        curves = [curves]
    maxlen = max([c.shape[1] for c in curves])

    force = np.zeros((maxlen, len(curves)))
    skin = np.zeros((maxlen, len(curves)))
    valid = np.zeros((maxlen, len(curves)), dtype=bool)
    for sample, c in enumerate(curves):
        force[0:c.shape[1], sample] = c[0, :]
        skin[0:c.shape[1], sample] = c[1, :]
        valid[0:c.shape[1], sample] = True

    return force, skin, valid


def multistep_predict(predictor, forces, valid, skin_start, skin_p=None):
    """Multi-step prediction.
    Inputs have the form: forces[step, sample], valid[step, sample], skin_start[sample]
    Output has the form:  skin_p[step, sample]
    """
    n_steps = forces.shape[0]
    n_samples = forces.shape[1]

    if skin_p is None:
        skin_p = np.zeros((n_steps, n_samples))
    skin_p[0, :] = skin_start

    for step in range(1, n_steps):
        # clamp to range to avoid table under-/overflows
        skin_p[step-1, skin_p[step-1, :] < 0] = 0
        skin_p[step-1, skin_p[step-1, :] > max_skin] = max_skin

        x = np.vstack((forces[step, :], skin_p[step-1, :]))
        skin_p[step, :] = predictor(x)

    skin_p[~valid] = 0
    return skin_p


def restarting_multistep_predict(predictor, forces, valid, skin, restart_steps):
    if restart_steps == 0:
        return multistep_predict(predictor, forces, valid, skin[0, :])

    s_p = np.zeros(skin.shape)
    for start_step in range(0, forces.shape[0], restart_steps):
        end_step = start_step + restart_steps
        multistep_predict(predictor, forces[start_step:end_step, :], valid[start_step:end_step, :], skin[start_step, :],
                          skin_p=s_p[start_step:end_step, :])
    return s_p


def multistep_error(skin_p, skin, valid, mean_err=False):
    """Calculates the error function of multiple prediction steps."""
    diff = (skin_p - skin)**2
    diff[~valid] = 0
    if not mean_err:
        return 0.5 * np.sum(diff)
    else:
        return 0.5 * np.mean(diff)


def multistep_error_per_sample(skin_p, skin, valid):
    """Calculates the error function of multiple prediction steps."""
    diff = (skin_p - skin)**2
    diff[~valid] = 0
    return 0.5 * np.sum(diff, axis=0)


def sparse_multiply(a, b):
    # workaround: scipy throws an error if trying to multiply a sparse matrix with a 1x1 sparse matrix
    if b.shape == (1, 1):
        return b[0, 0] * a
    else:
        return a.multiply(b)


def multistep_gradient(predictor_with_grad, force, skin, valid, reset_steps=0):
    """Calculates the gradient of the error function over multi-step prediction.
    Inputs have the form: force[step, sample], skin[step, sample], valid[step, sample]
    Output has the form:  grad[weight]
    predictor_with_grad(skin[step, sample]) should return a tuple of
    (skin[step+1, sample], dSkin/dPrevious, dSkin/dWeights).
    """

    # x[0=force/1=skin, sample]
    # skin[step, sample]
    # skin_p[step, sample]
    # pred_wrt_prev[sample]                     dense
    # pred_wrt_weights[weight, sample]          sparse
    # skin_wrt_weights[weight, sample]          sparse
    # prev_skin_wrt_weights[weight, sample]     sparse
    # error_wrt_weights[weight, sample]         sparse

    n_steps = force.shape[0]
    n_samples = force.shape[1]

    skin_p = np.zeros((n_steps, n_samples))
    skin_p[0, :] = skin[0, :]

    skin_wrt_weights = None
    total_error_wrt_weights = None
    grad = None

    for step in range(1, n_steps):
        # clamp to range to avoid under-/overflows
        skin_p[step-1, skin_p[step-1, :] < 0] = 0
        skin_p[step-1, skin_p[step-1, :] > max_skin] = max_skin

        # get gradient of prediction
        x = np.vstack((force[step, :], skin_p[step-1, :]))
        skin_p[step, :], pred_wrt_prev_all, pred_wrt_weights = predictor_with_grad(x)
        pred_wrt_prev = scipy.sparse.csc_matrix(pred_wrt_prev_all[1, :])
        pred_wrt_weights = pred_wrt_weights.tocsc()

        # assert not np.any(np.isnan(pred_wrt_prev.toarray()))
        # assert not np.any(np.isnan(pred_wrt_weights.toarray()))
        # print "pred_wrt_prev:", pred_wrt_prev
        # print "pred_wrt_weights:", pred_wrt_weights

        # backpropagte through time
        if step == 1 or (reset_steps != 0 and step % reset_steps == 0):
            skin_wrt_weights = pred_wrt_weights
        else:
            skin_wrt_weights = pred_wrt_weights + sparse_multiply(skin_wrt_weights, pred_wrt_prev)

        # assert not np.any(np.isnan(skin_wrt_weights.toarray()))

        # calculate gradient of error function
        efac = skin_p[step, :] - skin[step, :]
        efac[~valid[step, :]] = 0
        efac = scipy.sparse.csc_matrix(efac)
        error_wrt_weights = sparse_multiply(skin_wrt_weights, efac)

        # print "efac:", efac
        # assert not np.any(np.isnan(error_wrt_weights.toarray()))

        # sum error over steps
        if grad is None:
            grad = np.zeros((error_wrt_weights.shape[0]))
        if step == 1:  # or (reset_steps != 0 and step % reset_steps == 0):
            if total_error_wrt_weights is not None:
                grad += np.sum(total_error_wrt_weights.toarray(), axis=1)
            total_error_wrt_weights = error_wrt_weights
        else:
            total_error_wrt_weights = total_error_wrt_weights + error_wrt_weights

        # print "total_error_wrt_weights:", total_error_wrt_weights
        # assert not np.any(np.isnan(total_error_wrt_weights.toarray()))

    # sum gradient over samples
    grad += np.sum(total_error_wrt_weights.toarray(), axis=1)
    # assert not np.any(np.isnan(grad))
    return grad


def split_multicurves(o_force, o_skin, o_valid, split_steps, overlap=0):
    n_curves = o_force.shape[1]

    # pad steps if necessary
    n_o_steps = o_force.shape[0]
    n_rest = n_o_steps % split_steps
    if n_rest != 0:
        n_fill = split_steps - n_rest
        n_steps = n_o_steps + n_fill

        force = np.zeros((n_steps, n_curves))
        force[0:n_o_steps, :] = o_force[0:n_o_steps, :]
        skin = np.zeros((n_steps, n_curves))
        skin[0:n_o_steps, :] = o_skin[0:n_o_steps, :]
        valid = np.zeros((n_steps, n_curves), dtype='bool')
        valid[0:n_o_steps, :] = o_valid[0:n_o_steps, :]
    else:
        n_steps = n_o_steps
        force = o_force
        skin = o_skin
        valid = o_valid

    # calculate number of generated samples
    assert n_steps % split_steps == 0
    samples_per_curve = n_steps / split_steps
    n_samples = n_curves * samples_per_curve

    # assign curves to offsets
    offset_curves = [[] for _ in range(split_steps)]
    offset = 0
    for curve in range(n_curves):
        offset_curves[offset].append(curve)
        offset = (offset + 1) % split_steps

    # segment curves in samples
    l_force = []
    l_skin = []
    l_valid = []
    for offset in range(split_steps):
        my_curves = offset_curves[offset]
        for sample in range(samples_per_curve - 1):
            start_step = offset + sample * split_steps
            end_step = start_step + split_steps + overlap

            if end_step > n_steps:
                continue

            l_force.append(force[start_step:end_step, my_curves])
            l_skin.append(skin[start_step:end_step, my_curves])
            l_valid.append(valid[start_step:end_step, my_curves])

    # add beginning of all curves (to avoid missing steps due to offset)
    l_force.append(force[0:split_steps+overlap, :])
    l_skin.append(skin[0:split_steps+overlap, :])
    l_valid.append(valid[0:split_steps+overlap, :])

    # shuffle samples
    perm = range(len(l_force))
    random.seed(100)
    random.shuffle(perm)
    l_force = [l_force[i] for i in perm]
    l_skin = [l_skin[i] for i in perm]
    l_valid = [l_valid[i] for i in perm]

    # build output
    s_force = np.hstack(l_force)
    s_skin = np.hstack(l_skin)
    s_valid = np.hstack(l_valid)

    # remove completely invalid samples
    have_valid = np.any(s_valid, axis=0)
    s_force = s_force[:, have_valid]
    s_skin = s_skin[:, have_valid]
    s_valid = s_valid[:, have_valid]

    return s_force, s_skin, s_valid
