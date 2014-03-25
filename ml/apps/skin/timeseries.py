import numpy as np
import matplotlib.pyplot as plt


def plot_phase(data, *args):
    plt.plot(data[0, :], data[1, :], *args)
    plt.xlabel("force [N]")
    plt.ylabel("skin [V]")
    plt.xlim(0, 25)
    plt.ylim(0, 2)


def predict(predictor, forces, valid):
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

    skin[~valid] = 0
    return skin


def build_multicurve(curves):
    """Combines a list of curves into a single matrix of the form: x[step, sample]."""
    maxlen = max([len(c) for c in curves])

    x = np.zeros((maxlen, len(curves)))
    valid = np.zeros((maxlen, len(curves)), dtype=bool)
    for sample, c in enumerate(curves):
        x[0:len(c), sample] = c
        valid[0:len(c), sample] = True

    return x, valid

