
import numpy as np
import itertools


class TableRegression(object):

    def __init__(self, mins, maxs, steps):
        if not (len(mins) == len(maxs) == len(steps)):
            raise ValueError("arguments must have same dimensions")
        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.steps = np.asarray(steps)
        self.dims = len(self.mins)

        low_high = [0, 1]
        self.low_high_selector = np.array(list(itertools.product(*itertools.repeat(low_high, self.dims)))).T

    def pos_for(self, x):
        if np.any(x < self.mins[:, np.newaxis]) or np.any(x >= self.maxs[:, np.newaxis]):
            raise ValueError("supplied value is out of table range")
        rel = x - self.mins[:, np.newaxis]
        rel_steps = rel / self.steps[:, np.newaxis]
        low = np.int(rel_steps)
        high = low + 1
        high_fac = rel_steps - low
        low_fac = 1 - high_fac

        low_high = np.concatenate((low[:, :, np.newaxis], high[:, :, np.newaxis]), axis=2)





class TwoDimensionalTable(object):

    def __init__(self, x_range, x_step, y_range, y_step):
        self.x_support = np.arange(x_range[0], x_range[1] + 0.1*x_step, x_step)
        self.y_support = np.arange(y_range[0], y_range[1] + 0.1*y_step, y_step)
        self.table = np.zeros((self.y_support.shape[0], self.x_support.shape[0]))

    def selection_vector(self, x, support):
        """
        :type x: np.ndarray
        :rtype: np.ndarray
        """
        sv = np.zeros(support.shape)

        i = np.searchsorted(support, x)
        if i == 0 or i >= len(support):
            raise ValueError("Value %g out of support [%g, %g]." % (x, support[0], support[-1]))
        low = support[i-1]
        high = support[i]
        pos = (x-low) / (high-low)

        sv[i-1] = 1-pos
        sv[i] = pos
        return sv

    def selection_matrix(self, xs, support):
        sv = np.zeros((support.size, xs.size))
        for j in range(xs.size):
            sv[:, j] = self.selection_vector(xs[j], support)
        return sv

    def predict(self, xs, ys):
        xsel = self.selection_matrix(xs, self.x_support)
        ysel = self.selection_matrix(ys, self.y_support)
        return np.dot(ysel.T, np.dot(self.table, xsel))

    def train(self, xs, ys, targets):
        xsel = self.selection_matrix(xs, self.x_support)
        xsel_inv = np.linalg.pinv(xsel)
        ysel = self.selection_matrix(ys, self.y_support)
        ysel_inv = np.linalg.pinv(ysel)
        self.table = np.dot(ysel_inv.T, np.dot(targets[np.newaxis, :], xsel_inv))

