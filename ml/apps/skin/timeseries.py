import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.patches
import math
import random
import ml.common.progress as progress
from GPy.util import Tango

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


def plot_multicurve_error(force, force_predicted, valid, *args):
    n_curves = force.shape[1]
    assert force_predicted.shape[1] == n_curves
    assert valid.shape[1] == n_curves

    height = int(math.sqrt(n_curves))
    width = int(math.ceil(n_curves / float(height)))

    for c in range(n_curves):
        plt.subplot(height, width, c+1)
        valid_to = np.where(valid[:, c])[0][-1] + 1
        plt.plot(force[0:valid_to, c], force_predicted[0:valid_to, c], *args)
        plt.xlim(0, max_force)
        plt.ylim(0, max_force)
        plt.title(str(c))
        plt.gca().set_aspect('equal')

        if c % width == 0:
            plt.ylabel("predicted force [N]")
        else:
            plt.gca().set_yticklabels([])
        if c / width == height-1:
            plt.xlabel("force [N]")
        else:
            plt.gca().set_xticklabels([])


def get_sample(smpl, valid, *datas):
    assert valid.ndim == 2
    where_valid = np.where(valid[:, smpl])[0]
    if len(where_valid) == 0:
        valid_to = 0
    else:
        valid_to = where_valid[-1] + 1

    vdatas = [d[..., 0:valid_to, smpl] for d in datas]
    return tuple(vdatas)


def plot_mc(valid, ydata=None, conf=None, xdata=None,
            prob=None, prob_extents=None, prob_max=None,
            conf_upper=None, conf_lower=None,
            ylabel=None, xlabel=None, ymax=None, xmax=None, label=None,
            side='left', timestep=None, conf_args={}, aspect=None,
            y_tick_color=None, n_plot_samples=None, title=None, conf_pointwise=False,
            plot_numbering=True, **data_args):
    assert valid.ndim == 2
    n_samples = valid.shape[1]
    n_max_steps = valid.shape[0]

    assert side in ['left', 'right']
    if ydata is not None: assert ydata.shape[1] == n_samples and ydata.shape[0] == n_max_steps
    if xdata is not None: assert xdata.shape[1] == n_samples and xdata.shape[0] == n_max_steps
    if conf is not None: assert conf.shape[1] == n_samples and conf.shape[0] == n_max_steps
    if conf_upper is not None: assert conf_upper.shape[1] == n_samples and conf_upper.shape[0] == n_max_steps
    if conf_lower is not None: assert conf_lower.shape[1] == n_samples and conf_lower.shape[0] == n_max_steps
    if prob is not None:
        assert prob_extents is not None and prob_max is not None
        assert prob.ndim == 3 and prob.shape[3] == n_samples and prob.shape[2] == n_max_steps

    if 'alpha' not in conf_args: conf_args['alpha'] = 0.1
    if 'edgecolor' not in conf_args: conf_args['edgecolor'] = Tango.colorsHex['darkRed']
    if 'facecolor' not in conf_args: conf_args['facecolor'] = Tango.colorsHex['lightRed']

    if n_plot_samples and n_plot_samples < n_samples:
        n_samples = n_plot_samples

        valid = valid[..., 0:n_plot_samples]
        if ydata is not None: ydata = ydata[..., 0:n_plot_samples]
        if xdata is not None: xdata = xdata[..., 0:n_plot_samples]
        if conf is not None: conf = conf[..., 0:n_plot_samples]
        if conf_upper is not None: conf_upper = conf_upper[..., 0:n_plot_samples]
        if conf_lower is not None: conf_lower = conf_upper[..., 0:n_plot_samples]
        if prob is not None: prob = prob[..., 0:n_plot_samples]

    if conf_upper is None or conf_lower is None:
        conf_upper = conf
        conf_lower = conf
    else:
        assert conf is None, "both conf_upper/conf_lower and conf specified"

    height = int(math.sqrt(n_samples))
    width = int(math.ceil(n_samples / float(height)))

    if xdata is None and xmax is None:
        if timestep:
            xmax = n_max_steps * timestep
            xlabel = "time [s]"
        else:
            xmax = n_max_steps
            xlabel = "step"

    for c in range(n_samples):
        plt.subplot(height, width, c+1)
        where_valid = np.where(valid[:, c])[0]
        if len(where_valid) == 0:
            valid_to = 0
        else:
            valid_to = where_valid[-1] + 1

        if xdata is not None:
            tdata = xdata[0:valid_to, c]
        else:
            if timestep:
                tdata = np.linspace(0, valid_to * timestep, valid_to)
            else:
                tdata = np.arange(valid_to)

        ax1 = plt.gca()
        if side == 'right':
            ax = ax1.twinx()
        else:
            ax = ax1

        if conf_lower is not None:
            if conf_pointwise:
                for i in range(valid_to - 2):
                    plt.fill_between(tdata[i:i+2],
                                     ydata[i:i+2, c] + conf_upper[i:i+2, c],
                                     ydata[i:i+2, c] - conf_lower[i:i+2, c],
                                     **conf_args)
            else:
                plt.fill_between(tdata,
                                 ydata[0:valid_to, c] + conf_upper[0:valid_to, c],
                                 ydata[0:valid_to, c] - conf_lower[0:valid_to, c],
                                 **conf_args)

        if ydata is not None:
            ax.plot(tdata, ydata[0:valid_to, c], label=label, **data_args)
        if c == n_samples - 1 and label:
            plt.legend()

        if prob is not None:
            cmap = plt.get_cmap('jet')
            plt.imshow(prob[:, 0:valid_to, c], interpolation='none', origin='lower', cmap=cmap,
                       extent=(0, tdata[-1], prob_extents[0], prob_extents[1]),
                       aspect='auto', vmin=0, vmax=prob_max)
            if c == n_samples - 1:
                plt.colorbar()
            fillbox = np.asarray([[np.min(prob[:, :, c])]])
            plt.imshow(fillbox, interpolation='none', cmap=cmap,
                       extent=((tdata[-2] + tdata[-1])/2., xmax, prob_extents[0], prob_extents[1]),
                       aspect='auto')

        if xmax:
            ax.set_xlim(0, xmax)
        if ymax:
            ax.set_ylim(0, ymax)

        if y_tick_color:
            for tl in ax.get_yticklabels():
                tl.set_color(y_tick_color)

        if (side == 'left' and c % width == 0) or (side == 'right' and (c+1) % width == 0):
            if ylabel:
                ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])

        if c / width == height-1:
            if xlabel:
                ax1.set_xlabel(xlabel)
        else:
            ax1.set_xticklabels([])

        if aspect:
            ax.set_aspect(aspect)

        if plot_numbering:
            plt.title(str(c))

    if title:
        plt.suptitle(title)


def plot_multicurve_time(force, skin, valid,
                         skin_predicted=None, skin_predicted_conf=None,
                         force_predicted=None, force_predicted_conf=None,
                         force_real=None,
                         force_prob=None, force_prob_extents=None, force_prob_max=None,
                         timestep=None):
    n_curves = force.shape[1]
    if skin is not None:
        assert skin.shape[1] == n_curves
    assert valid.shape[1] == n_curves

    height = int(math.sqrt(n_curves))
    width = int(math.ceil(n_curves / float(height)))

    if timestep:
        max_time = force.shape[0] * timestep
    else:
        max_time = force.shape[0]

    tick_color = 'k'
    if force_prob is None:
        force_color = 'k'
        linewidth = 1
    else:
        force_color = 'w'
        linewidth = 3

    for c in range(n_curves):
        plt.subplot(height, width, c+1)
        where_valid = np.where(valid[:, c])[0]
        if len(where_valid) == 0:
            valid_to = 0
        else:
            valid_to = where_valid[-1] + 1

        if timestep:
            ts = np.linspace(0, valid_to * timestep, valid_to)
        else:
            ts = np.arange(valid_to)

        ax1 = plt.gca()
        if force_real is not None:
            ax1.plot(ts, force_real[0:valid_to, c], 'b', alpha=0.4, linewidth=linewidth)
        ax1.plot(ts, force[0:valid_to, c], force_color, linewidth=linewidth)
        if force_predicted is not None:
            if force_predicted_conf is not None:
                plt.fill_between(ts,
                                 force_predicted[0:valid_to, c] + force_predicted_conf[0:valid_to, c],
                                 force_predicted[0:valid_to, c] - force_predicted_conf[0:valid_to, c],
                                 edgecolor=Tango.colorsHex['darkRed'], facecolor=Tango.colorsHex['lightRed'], alpha=0.1)
                ax1.plot(ts, force_predicted[0:valid_to, c], 'r', linewidth=linewidth)
            else:
                ax1.plot(ts, force_predicted[0:valid_to, c], 'r', linewidth=linewidth)
        if force_prob is not None:
            cmap = plt.get_cmap('jet')
            plt.imshow(force_prob[:, 0:valid_to, c], interpolation='none', origin='lower', cmap=cmap,
                       extent=(0, ts[-1], force_prob_extents[0], force_prob_extents[1]),
                       aspect='auto', vmin=0, vmax=force_prob_max)
            if c == n_curves - 1:
                plt.colorbar()
            fillbox = np.asarray([[np.min(force_prob[:, :, c])]])
            plt.imshow(fillbox, interpolation='none', cmap=cmap,
                       extent=((ts[-2] + ts[-1])/2., max_time, force_prob_extents[0], force_prob_extents[1]),
                       aspect='auto')
        ax1.set_ylim(0, max_force)
        ax1.set_xlim(0, max_time)
        for tl in ax1.get_yticklabels():
            tl.set_color(tick_color)
        for tl in ax1.get_xticklabels():
            tl.set_color(tick_color)
        if c % width == 0:
            ax1.set_ylabel("force [N]")
        else:
            ax1.set_yticklabels([])
        if c / width == height-1:
            ax1.set_xlabel("time [s]")
        else:
            ax1.set_xticklabels([])

        if skin is not None:
            ax2 = ax1.twinx()
            ax2.plot(ts, skin[0:valid_to, c], 'b', linewidth=linewidth)
            if skin_predicted is not None:
                if skin_predicted_conf is not None:
                    plt.fill_between(ts,
                                     skin_predicted[0:valid_to, c] + skin_predicted_conf[0:valid_to, c],
                                     skin_predicted[0:valid_to, c] - skin_predicted_conf[0:valid_to, c],
                                     edgecolor=Tango.colorsHex['darkRed'], facecolor=Tango.colorsHex['lightRed'], alpha=0.1)
                    ax2.plot(ts, skin_predicted[0:valid_to, c], 'r', linewidth=linewidth)
                else:
                    ax2.plot(ts, skin_predicted[0:valid_to, c], 'r', linewidth=linewidth)
            ax2.set_ylim(0, max_skin)
            ax2.set_xlim(0, max_time)
            for tl in ax2.get_yticklabels():
                tl.set_color('b')
            if (c+1) % width == 0:
                ax2.set_ylabel("skin [V]", color='b')
            else:
                ax2.set_yticklabels([])

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

    rec = ds.record(purpose, taxel, 0)

    X = np.zeros((2, 0))
    Z = np.zeros((0, ))
    for i in range(n_curves):
        rec = ds.record(purpose, taxel, i)
        x = rec[:, 0:-1]
        z = rec[1, 1:]

        X = np.concatenate((X, x), axis=1)
        Z = np.concatenate((Z, z))
        #Z = np.concatenate((Z, z), axis=1)

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
    multifreq = curves[0].shape[0] > 2

    force = np.zeros((maxlen, len(curves)))
    if multifreq:
        skin = np.zeros((curves[0].shape[0] - 1, maxlen, len(curves)))
    else:
        skin = np.zeros((maxlen, len(curves)))
    valid = np.zeros((maxlen, len(curves)), dtype=bool)
    for sample, c in enumerate(curves):
        force[0:c.shape[1], sample] = c[0, :]
        if multifreq:
            skin[:, 0:c.shape[1], sample] = c[1:, :]
        else:
            skin[0:c.shape[1], sample] = c[1, :]
        valid[0:c.shape[1], sample] = True

    return force, skin, valid


def build_flat_data(curves):
    """Concatenates all given curves into the form:
    force[sample] and skin[feature, sample]."""
    if not isinstance(curves, (list, tuple)):
        curves = [curves]
    n_samples = sum([c.shape[1] for c in curves])
    n_features = curves[0].shape[0] - 1

    force = np.zeros((n_samples,))
    skin = np.zeros((n_features, n_samples))
    pos = 0
    for c in curves:
        ns = c.shape[1]
        force[pos : pos+ns] = c[0, :]
        skin[:, pos : pos+ns] = c[1:, :]
        pos += ns

    return force, skin


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


def multistep_error(prediction, truth, valid, mean_err=False):
    """Calculates the error function of multiple prediction steps."""
    diff = (prediction - truth)**2
    diff[~valid] = 0
    if not mean_err:
        return 0.5 * np.sum(diff)
    else:
        return 0.5 * np.sum(diff) / np.sum(valid)


def multistep_r2(prediction, truth, valid):
    """Calculates the R^2 value of the data set"""
    t = truth[valid]
    p = prediction[valid]

    tmean = np.mean(t)
    sstot = np.sum((t - tmean)**2)
    ssres = np.sum((t - p)**2)
    r2 = 1. - ssres / float(sstot)
    return r2


def multistep_error_per_sample(prediction, truth, valid):
    """Calculates the error function of multiple prediction steps."""
    diff = (prediction - truth)**2
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
