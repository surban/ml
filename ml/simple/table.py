
import numpy as np
import itertools


class TableRegression(object):

    def __init__(self, mins, steps, maxs):
        if not (len(mins) == len(maxs) == len(steps)):
            raise ValueError("arguments must have same dimensions")
        self.mins = np.asarray(mins)
        self.steps = np.asarray(steps)
        self.elems = ((maxs - self.mins) / self.steps).astype('int')
        self.maxs = self.mins + self.elems * self.steps
        self.dims = len(self.mins)

        self.strides = np.ones((self.dims,), dtype='int')
        for d in range(1, self.dims):
            self.strides[d] = self.strides[d-1] * self.elems[d-1]
        self.fv_size = int(np.product(self.elems))

        self.low_high_selector = \
            np.array(list(itertools.product(*itertools.repeat([0, 1], self.dims))), dtype='int').T[:, np.newaxis, :]

        self.weights = np.zeros((self.fv_size,))

    def table_fvec(self, x):
        x = np.asarray(x)
        viol = (x < self.mins[:, np.newaxis]) | (x >= self.maxs[:, np.newaxis])
        if np.any(viol):
            raise ValueError("supplied value(s) are out of table range at " +
                             str(np.transpose(np.nonzero(viol))))

        rel = x - self.mins[:, np.newaxis]
        rel_steps = rel / self.steps[:, np.newaxis]
        low = rel_steps.astype('int')

        #print "low: ", low

        lh = low[:, :, np.newaxis] + self.low_high_selector
        lh_idx = np.sum(lh * self.strides[:, np.newaxis, np.newaxis], axis=0)
        lh_idx_flat = np.reshape(lh_idx, (-1,))

        #print "lh: ", lh
        #print "lh_idx: ", lh_idx
        #print "lh_idx_flat: ", lh_idx_flat

        smpl_idx = np.repeat(np.arange(lh_idx.shape[0])[:, np.newaxis], lh_idx.shape[1], axis=1)
        smpl_idx_flat = np.reshape(smpl_idx, (-1,))

        #print "smpl_idx: ", smpl_idx
        #print "smpl_idx_flat: ", smpl_idx_flat

        dists = rel_steps[:, :, np.newaxis] - lh
        fac_comp = 1 - np.abs(dists)
        fac = np.product(fac_comp, axis=0)
        fac_flat = np.reshape(fac, (-1,))

        x = np.zeros((self.fv_size, x.shape[1]))
        x[lh_idx_flat, smpl_idx_flat] = fac_flat

        return x

    def predict(self, x):
        return np.dot(self.weights, self.table_fvec(x))

    def train(self, x, t):
        self.weights = np.dot(t, np.linalg.pinv(self.table_fvec(x)))

