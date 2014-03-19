
import numpy as np


class TwoDimensionalTable(object):

    def __init__(self, x_range, y_range, step):
        self.x_support = np.arange(x_range[0], x_range[1] + 0.1*step, step)
        self.y_support = np.arange(y_range[0], y_range[1] + 0.1*step, step)
        self.table = np.zeros((self.y_support.shape[0], self.x_support.shape[0]))

    def selection_vector(self, x, support):
        """
        :type x: np.ndarray
        :rtype: np.ndarray
        """
        sv = np.zeros(support.shape)

        i = np.searchsorted(support, x)
        low = support[i]
        high = support[i+1]
        pos = (x-low) / (high-low)

        sv[i] = 1-pos
        sv[i+1] = pos
        return sv

    def selection_matrix(self, xs, support):
        sv = np.zeros((support.shape[0], xs.shape[1]))
        for j in range(xs.shape[1]):
            sv[:, j] = self.selection_vector(xs[:, j], support)
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
        self.table = np.dot(ysel_inv.T, np.dot(targets, xsel_inv))

