
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg


class TableRegression(object):

    def __init__(self, mins, steps, maxs, smooth=False):
        if not (len(mins) == len(maxs) == len(steps)):
            raise ValueError("arguments must have same dimensions")
        self.smooth = smooth

        self._steps = np.asarray(steps)
        self._mins = np.asarray(mins) - self._steps
        self._elems = ((maxs - self._mins) / self._steps + 3).astype('int')
        self._maxs = self._mins + self._elems * self._steps

        self._min_limits = self._mins + self._steps
        self._max_limits = self._mins + (self._elems - 2) * self._steps

        self._dims = len(self._mins)
        self._strides = np.ones((self._dims,), dtype='int')
        for d in range(1, self._dims):
            self._strides[d] = self._strides[d-1] * self._elems[d-1]
        self._n_weights = int(np.product(self._elems))

        self._low_high_selector = \
            np.array(list(itertools.product(*itertools.repeat([0, 1], self._dims))), dtype='int').T[:, np.newaxis, :]

        self._low_high_smooth_selector = \
            np.array(list(itertools.product(*itertools.repeat([-1, 0, 1, 2], self._dims))), dtype='int').T[:, np.newaxis, :]
        self._low_high_fac_selector = self._low_high_smooth_selector + 2
        self._low_high_fac_selector[self._low_high_fac_selector > 2] -= 4

        #print "_low_high_fac_selector: ", self._low_high_fac_selector

        self._weights = np.zeros((self._n_weights,))

    @property
    def max_values(self):
        return self._max_limits[:]

    def _table_idx(self, x):
        if self.smooth:
            return self._table_idx_smooth(x)
        else:
            return self._table_idx_nearest(x)

    def _table_idx_nearest(self, x):
        x = np.asarray(x)
        viol = (x < self._min_limits[:, np.newaxis]) | (x > self._max_limits[:, np.newaxis])
        if np.any(viol):
            raise ValueError("supplied value(s) are out of table range at " +
                             str(np.transpose(np.nonzero(viol))))

        rel = x - self._mins[:, np.newaxis]
        rel_steps = rel / self._steps[:, np.newaxis]
        low = rel_steps.astype('int')

        #print "low: ", low

        lh = low[:, :, np.newaxis] + self._low_high_selector
        lh_idx = np.sum(lh * self._strides[:, np.newaxis, np.newaxis], axis=0)
        lh_idx_flat = np.reshape(lh_idx, (-1,))

        #print "lh: ", lh
        #print "lh_idx: ", lh_idx
        #print "lh_idx_flat: ", lh_idx_flat

        smpl_idx = np.repeat(np.arange(lh_idx.shape[0])[:, np.newaxis], lh_idx.shape[1], axis=1)
        smpl_idx_flat = np.reshape(smpl_idx, (-1,))

        #print "smpl_idx: ", smpl_idx
        #print "smpl_idx_flat: ", smpl_idx_flat

        dists = rel_steps[:, :, np.newaxis] - lh
        fac_comp = 1 - np.fabs(dists)
        fac = np.product(fac_comp, axis=0)
        fac_flat = np.reshape(fac, (-1,))

        return lh_idx_flat, smpl_idx_flat, fac_flat

    def _table_idx_smooth(self, x):
        x = np.asarray(x)
        viol = (x < self._min_limits[:, np.newaxis]) | (x > self._max_limits[:, np.newaxis])
        if np.any(viol):
            raise ValueError("supplied value(s) are out of table range (min: %s, max: %s) at %s" %
                             (str(self._min_limits), str(self._max_limits), str(np.transpose(np.nonzero(viol)))))

        rel = x - self._mins[:, np.newaxis]
        rel_steps = rel / self._steps[:, np.newaxis]
        low = rel_steps.astype('int')
        base_dists = rel_steps - low

        # print "low: ", low
        # print "base_dists: ", base_dists

        lh = low[:, :, np.newaxis] + self._low_high_smooth_selector
        lh_idx = np.sum(lh * self._strides[:, np.newaxis, np.newaxis], axis=0)
        lh_idx_flat = np.reshape(lh_idx, (-1,))

        # print "lh: ", lh
        # print "lh_idx: ", lh_idx
        # print "lh_idx_flat: ", lh_idx_flat

        smpl_idx = np.repeat(np.arange(lh_idx.shape[0])[:, np.newaxis], lh_idx.shape[1], axis=1)
        smpl_idx_flat = np.reshape(smpl_idx, (-1,))

        # print "smpl_idx: ", smpl_idx
        # print "smpl_idx_flat: ", smpl_idx_flat

        fac_comp = np.fabs(base_dists[:, :, np.newaxis] - self._low_high_fac_selector) / 4.0
        fac = np.product(fac_comp, axis=0)
        fac_flat = np.reshape(fac, (-1,))

        return lh_idx_flat, smpl_idx_flat, fac_flat

    def _table_fvec_dense(self, x):
        lh_idx_flat, smpl_idx_flat, fac_flat = self._table_idx(x)
        x = np.zeros((self._n_weights, x.shape[1]))
        x[lh_idx_flat, smpl_idx_flat] = fac_flat
        return x

    def _table_fvec_sparse(self, x):
        lh_idx_flat, smpl_idx_flat, fac_flat = self._table_idx(x)
        x = scipy.sparse.lil_matrix((self._n_weights, x.shape[1]))
        x[lh_idx_flat, smpl_idx_flat] = fac_flat
        return x.tocsc()

    def predict_dense(self, x):
        return np.dot(self._weights, self._table_fvec_dense(x))

    def predict(self, x):
        fvec = self._table_fvec_sparse(x)
        return fvec.T.dot(self._weights).T

    def train_dense(self, x, t):
        self._weights = np.dot(t, np.linalg.pinv(self._table_fvec_dense(x)))

    def train(self, x, t):
        fvec = self._table_fvec_sparse(x)
        res, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = scipy.sparse.linalg.lsqr(fvec.T, t.T)
        print "istop: %d   itn: %d" % (istop, itn)
        self._weights = res.T

    def train_lsmr(self, x, t):
        fvec = self._table_fvec_sparse(x)
        res, istop, itn, normr, normar, norma, conda, normx = scipy.sparse.linalg.lsmr(fvec.T, t.T)
        print "istop: %d   itn: %d" % (istop, itn)
        self._weights = res.T

    @property
    def weight_matrix(self):
        return np.reshape(self._weights, tuple(self._elems), 'F')

    def plot2d(self, rng=None):
        if self._dims != 2:
            raise ValueError("plot only suitable for TableRegressions with two dimensions")

        if not rng:
            rng = np.amax(np.fabs(self.weight_matrix))

        img = plt.imshow(self.weight_matrix.T, origin='lower', interpolation='nearest', cmap='PuOr',
                         extent=(self._mins[0], self._maxs[0], self._mins[1], self._maxs[1]),
                         aspect='auto')
        img.set_clim(-rng, rng)
        plt.colorbar()
        plt.xlabel("dim 0")
        plt.ylabel("dim 1")

        return img

    def plot2d_relative(self, rel_dim=0, rng=None):
        if self._dims != 2:
            raise ValueError("plot only suitable for TableRegressions with two dimensions")

        myval = np.arange(self._mins[rel_dim], self._maxs[rel_dim], self._steps[rel_dim])
        if rel_dim == 0:
            myval = myval[:, np.newaxis]
        elif rel_dim == 1:
            myval = myval[np.newaxis, :]
        data = self.weight_matrix - myval

        if not rng:
            rng = np.amax(np.fabs(data))

        img = plt.imshow(data.T, origin='lower', interpolation='nearest', cmap='PuOr',
                         extent=(self._mins[0], self._maxs[0], self._mins[1], self._maxs[1]),
                         aspect='auto')
        img.set_clim(-rng, rng)
        plt.colorbar()
        plt.xlabel("dim 0")
        plt.ylabel("dim 1")

        return img


if __name__ == '__main__':
    print "nonsmooth:"
    tr = TableRegression([0,    0],
                         [0.1,  0.1],
                         [1,    1])
    fv = tr._table_fvec_dense(np.asarray([[0.23, 0.46],
                                          [0.11, 0.09]]))
    print np.sum(fv, axis=0)

    print "smooth:"
    tr = TableRegression([0], [0.1], [0.5], smooth=True)
    fv = tr._table_fvec_dense(np.asarray([[0.21, 0.49]]))
    print fv
    print "sum: ", np.sum(fv, axis=0)

    print "smooth:"
    tr = TableRegression([0,    0],
                         [0.1,  0.1],
                         [0.4,  0.4],
                         smooth=True)
    fv = tr._table_fvec_dense(np.asarray([[0.21],
                                          [0.29]]))
    print fv.reshape(tr._elems)
    print "sum: ", np.sum(fv, axis=0)

