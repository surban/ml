import numpy as np
import matplotlib.pyplot as plt
import math


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
        valid_to = np.where(valid[:, c])[0][-1]

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


def predict_multistep(predictor, forces, valid):
    """Multi-step prediction.
    Inputs have the form: forces[step, sample], valid[step, sample]
    Output has the form:  skin[step, sample]
    """
    n_steps = forces.shape[0]
    n_samples = forces.shape[1]

    skin = np.zeros((n_steps, n_samples))
    for step in range(1, n_steps):
        x = np.vstack((forces[step, :], skin[step-1, :]))
        skin[step, :] = predictor(x)
        skin[step, skin[step, :] < 0] = 0
        skin[step, skin[step, :] > max_skin] = max_skin

    skin[~valid] = 0
    return skin


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

