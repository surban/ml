
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from abc import ABCMeta, abstractmethod

from ml.common.test import check_gradient


import time

class TableRegression(object):
    __metaclass__ = ABCMeta

    def __init__(self, mins, steps, maxs):
        if not (len(mins) == len(maxs) == len(steps)):
            raise ValueError("arguments must have same dimensions")

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
        self.weights = np.zeros((self._n_weights,))

        self._low_high_selector = None
        self.sparse = True
        self._init_selector()
        self._idxs_per_sample = self._low_high_selector.shape[2]


    @abstractmethod
    def _init_selector(self):
        pass

    @abstractmethod
    def _weight_idx_and_fac(self, x):
        pass

    @property
    def max_values(self):
        return self._max_limits[:]

    @property
    def weight_matrix(self):
        return np.reshape(self.weights, tuple(self._elems), 'F')

    @property
    def weights_per_sample(self):
        return self._idxs_per_sample

    @property
    def n_weights(self):
        return self.weights.size

    def _lh_idx(self, x):
        x = np.asarray(x)
        viol = (x < self._min_limits[:, np.newaxis]) | (x > self._max_limits[:, np.newaxis])
        if np.any(viol):
            raise ValueError("supplied value(s) are out of table range (min: %s, max: %s) at %s" %
                             (str(self._min_limits), str(self._max_limits), str(np.transpose(np.nonzero(viol)))))

        rel = x - self._mins[:, np.newaxis]
        rel_steps = rel / self._steps[:, np.newaxis]
        low = rel_steps.astype('int')

        lh = low[:, :, np.newaxis] + self._low_high_selector
        lh_idx = np.sum(lh * self._strides[:, np.newaxis, np.newaxis], axis=0)
        lh_idx_flat = np.reshape(lh_idx, (-1,))

        smpl_idx = np.repeat(np.arange(lh_idx.shape[0])[:, np.newaxis], lh_idx.shape[1], axis=1)
        smpl_idx_flat = np.reshape(smpl_idx, (-1,))

        #print "_table_idx:"
        #print "low: ", low
        #print "lh: ", lh
        #print "lh_idx: ", lh_idx
        #print "lh_idx_flat: ", lh_idx_flat
        #print "smpl_idx: ", smpl_idx
        #print "smpl_idx_flat: ", smpl_idx_flat
        #print

        return low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat

    def _fvec(self, x):
        lh_idx_flat, smpl_idx_flat, fac_flat = self._weight_idx_and_fac(x)
        return self._fvec_from_idx(x, lh_idx_flat, smpl_idx_flat, fac_flat)

    def _fvec_from_idx(self, x, lh_idx_flat, smpl_idx_flat, fac_flat):
        if self.sparse:
            x = scipy.sparse.lil_matrix((self._n_weights, x.shape[1]))
            x[lh_idx_flat, smpl_idx_flat] = fac_flat
            x = x.tocsr()
        else:
            x = np.zeros((self._n_weights, x.shape[1]))
            x[lh_idx_flat, smpl_idx_flat] = fac_flat
        return x

    def _fast_weight_mul(self, idx_flat, fac_flat):
        wsel = self.weights[idx_flat]

        wsel = np.reshape(wsel, (-1, self._idxs_per_sample))
        fac = np.reshape(fac_flat, (-1, self._idxs_per_sample))

        wf = wsel * fac
        return np.sum(wf, axis=1)

    def _predict_from_fvec(self, fvec):
        x = fvec.T.dot(self.weights).T
        return x

    def predict_slow(self, x):
        return self._predict_from_fvec(self._fvec(x))

    def predict(self, x):
        lh_idx_flat, smpl_idx_flat, fac_flat = self._weight_idx_and_fac(x)
        return self._fast_weight_mul(lh_idx_flat, fac_flat)

    def train(self, x, t):
        fvec = self._fvec(x)
        res, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = scipy.sparse.linalg.lsqr(fvec.T, t.T)
        print "istop: %d   itn: %d" % (istop, itn)
        self.weights = res.T

    def train_lsmr(self, x, t):
        fvec = self._fvec(x)
        res, istop, itn, normr, normar, norma, conda, normx = scipy.sparse.linalg.lsmr(fvec.T, t.T)
        print "istop: %d   itn: %d" % (istop, itn)
        self.weights = res.T

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


class NearestTableRegression(TableRegression):
    def _init_selector(self):
        self._low_high_selector = \
            np.array(list(itertools.product(*itertools.repeat([0, 1], self._dims))), dtype='int').T[:, np.newaxis, :]

    def _weight_idx_and_fac(self, x):
        low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat = self._lh_idx(x)

        dists = rel_steps[:, :, np.newaxis] - lh
        fac_comp = 1 - np.fabs(dists)
        fac = np.product(fac_comp, axis=0)
        fac_flat = np.reshape(fac, (-1,))

        return lh_idx_flat, smpl_idx_flat, fac_flat


class SmoothTableRegression(TableRegression):
    def _init_selector(self):
        self._low_high_selector = \
            np.array(list(itertools.product(*itertools.repeat([-1, 0, 1, 2], self._dims))), dtype='int').T[:, np.newaxis, :]
        self._low_high_fac_selector = self._low_high_selector + 2
        self._low_high_fac_selector[self._low_high_fac_selector > 2] -= 4
        self.sparse_weight_gradient = True

        # print "_low_high_fac_selector: ", self._low_high_fac_selector

    def _intermidiates(self, x):
        low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat = self._lh_idx(x)

        base_dists = rel_steps - low
        fac_comp = (base_dists[:, :, np.newaxis] - self._low_high_fac_selector) / 4.0

        fac = np.product(np.fabs(fac_comp), axis=0)
        fac_flat = np.reshape(fac, (-1,))

        return low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat, fac_comp, fac_flat

    def _weight_idx_and_fac(self, x):
        low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat, fac_comp, fac_flat = self._intermidiates(x)
        return lh_idx_flat, smpl_idx_flat, fac_flat

    def _x_gradient_from_intermidiates(self, x, low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat, fac_comp, fac_flat):
        # used variables:
        # fac_comp       [distance factor product dim, sample, weight number]
        # x_grad_fac_val [dim of derivative, distance factor product dim, sample, weight number]
        # x_grad_fac     [dim of derivative, distance factor product dim, sample, weight number]
        # x_grad_fac_prod[dim of derivative, sample, weight number]

        x_grad_fac_val = np.tile(fac_comp[np.newaxis, :, :, :], (self._dims, 1, 1, 1))
        x_grad_fac = np.fabs(x_grad_fac_val)
        for d in range(self._dims):
            tmp = np.sign(x_grad_fac_val[d, d, :, :])
            tmp[tmp == 0] = 1
            x_grad_fac[d, d, :, :] = tmp / 4.0 / self._steps[d]
        x_grad_fac_prod = np.product(x_grad_fac, axis=1)

        x_grad_weights = np.reshape(self.weights[lh_idx_flat], lh_idx.shape)
        x_grad = np.sum(x_grad_weights[np.newaxis, :, :] * x_grad_fac_prod, axis=2)

        # print "gradient:"
        # print "fac_comp: ", fac_comp
        # print "x_grad_fac: ", x_grad_fac
        # print "x_grad_prod: ", x_grad_prod
        # print "grad_weights: ", x_grad_weights[np.newaxis, :, :]
        # print

        return x_grad

    def _w_gradient_from_intermidiates(self, x, lh_idx_flat, smpl_idx_flat, fac_flat):
        if self.sparse_weight_gradient:
            w_grad = scipy.sparse.coo_matrix((fac_flat, (lh_idx_flat, smpl_idx_flat)),
                                             shape=(self._n_weights, x.shape[1]))
        else:
            w_grad = np.zeros((self._n_weights, x.shape[1]))
            w_grad[lh_idx_flat, smpl_idx_flat] = fac_flat
        return w_grad

    def gradient(self, x):
        """Returns (gradient w.r.t input, gradient w.r.t. weights)"""
        low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat, fac_comp, fac_flat = self._intermidiates(x)
        x_grad = self._x_gradient_from_intermidiates(x, low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat,
                                                     fac_comp, fac_flat)
        w_grad = self._w_gradient_from_intermidiates(x, lh_idx_flat, smpl_idx_flat, fac_flat)
        return x_grad, w_grad

    def predict_and_gradient(self, x):
        """Returns (prediction, gradient w.r.t input, gradient w.r.t. weights)"""
        low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat, fac_comp, fac_flat = self._intermidiates(x)

        pred = self._fast_weight_mul(lh_idx_flat, fac_flat)
        x_grad = self._x_gradient_from_intermidiates(x, low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat,
                                                     fac_comp, fac_flat)
        w_grad = self._w_gradient_from_intermidiates(x, lh_idx_flat, smpl_idx_flat, fac_flat)

        return pred, x_grad, w_grad

    def predict_and_gradient_indices(self, x):
        low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat, fac_comp, fac_flat = self._intermidiates(x)

        pred = self._fast_weight_mul(lh_idx_flat, fac_flat)
        x_grad = self._x_gradient_from_intermidiates(x, low, rel_steps, lh, lh_idx, lh_idx_flat, smpl_idx_flat,
                                                   fac_comp, fac_flat)

        w_grad_idx = lh_idx.T
        w_grad_smpl_idx = np.reshape(smpl_idx_flat, (-1, self._idxs_per_sample)).T
        w_grad_fac = np.reshape(fac_flat, (-1, self._idxs_per_sample)).T

        return pred, x_grad, w_grad_idx, w_grad_smpl_idx, w_grad_fac


########################################################################################################################
#                                             TEST CODE                                                                #
########################################################################################################################


def w_func_wrapper(tr, x, w):
    tr._weights = w[:, 0]
    return tr.predict(x)


def w_grad_wrapper(tr, x, w):
    tr._weights = w[:, 0]
    _, w_grad = tr.gradient(x)
    return w_grad.toarray()


def test_gradients():
    print "smooth gradient:"
    tr = SmoothTableRegression([0], [0.1], [0.5])
    tr.weights = np.random.random(size=tr.weights.shape)
    x = np.asarray([[0.22]])
    x_grad, w_grad = tr.gradient(x)
    print "x grad: "
    print x_grad
    print "w grad: "
    print w_grad.toarray()

    print "smooth gradient check x 1d"
    tr = SmoothTableRegression([0], [0.1], [0.4])
    tr.weights = np.random.random(size=tr.weights.shape)
    # tr._weights = np.arange(tr._weights.size)
    x = np.asarray([[0.23, 0.17]])
    # check_gradient(tr.predict, lambda x: tr.gradient(x)[0], x)
    print "smooth gradient check w 1d"
    check_gradient(lambda w: w_func_wrapper(tr, x, w), lambda w: w_grad_wrapper(tr, x, w),
                   tr.weights[:, np.newaxis])

    print "smooth gradient check x 2d"
    tr = SmoothTableRegression([0,    0],
                               [0.1,  0.1],
                               [0.4,  0.4])
    # np.random.seed(1)
    tr.weights = np.random.random(size=tr.weights.shape)
    # for i in range(tr._elems[0]):
    #     for j in range(tr._elems[1]):
    #         tr._weights[i*tr._strides[0] + j*tr._strides[1]] = i*10 + j
    #tr._weights = np.ones(tr._weights.shape)
    x = np.asarray([[0.21, 0.18, 0.10],
                    [0.23, 0.30, 0.11]])
    check_gradient(tr.predict, lambda x: tr.gradient(x)[0], x)
                   #direction=np.asarray([[0], [1]]))
    print "smooth gradient check w 2d"
    check_gradient(lambda w: w_func_wrapper(tr, x, w), lambda w: w_grad_wrapper(tr, x, w),
                   tr.weights[:, np.newaxis])


def test_fvec():
    print "nonsmooth 1d fvec:"
    tr = NearestTableRegression([0,    0],
                                [0.1,  0.1],
                                [1,    1])
    fv = tr._fvec(np.asarray([[0.23, 0.46],
                              [0.11, 0.09]]))
    fv = fv.toarray()
    print np.sum(fv, axis=0)

    print "smooth 1d fvec:"
    tr = SmoothTableRegression([0], [0.1], [0.5])
    fv = tr._fvec(np.asarray([[0.21, 0.49]]))
    fv = fv.toarray()
    print fv
    print "sum: ", np.sum(fv, axis=0)

    print "smooth 2d fvec:"
    tr = SmoothTableRegression([0,    0],
                               [0.1,  0.1],
                               [0.4,  0.4])
    fv = tr._fvec(np.asarray([[0.21],
                              [0.29]]))
    fv = fv.toarray()
    print fv.reshape(tr._elems)
    print "sum: ", np.sum(fv, axis=0)
    print


if __name__ == '__main__':
    test_fvec()
    test_gradients()

    for i in range(10):
        test_gradients()


