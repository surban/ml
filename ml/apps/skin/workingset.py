from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from ml.apps.skin.timeseries import *
from ml.simple.table import *
from ml.datasets.skin import SkinDataset
from ml.common.util import ParameterHistory
from ml.common.progress import status, done
from ml.common.test import check_gradient, check_directional_gradient


class SkinWorkingset(object):

    all_prts = ['trn', 'val', 'tst']
    range_epsilon = 0.1

    def __init__(self, ds_name, taxel=None, curve_limit=None,
                 discrete_force_states=100, discrete_skin_states=100):

        self.discrete_force_states = discrete_force_states
        self.discrete_skin_states = discrete_skin_states

        self.ds = SkinDataset(ds_name)
        self.ds.print_statistics()
        self.curve_limit = curve_limit

        self._taxel = None
        self.taxel = taxel

    @property
    def taxel(self):
        return self._taxel

    @taxel.setter
    def taxel(self, value):
        self._taxel = value
        print "Using taxel ", self._taxel
        print
        self._build_data()
        self._build_discrete_data()

    def _build_data(self):
        self.ns_in = {}
        self.ns_skin = {}
        self.curves = {}
        self.force = {}
        self.skin = {}
        self.valid = {}

        if self._taxel is None:
            return

        for prt in self.all_prts:
            self.ns_in[prt], self.ns_skin[prt] = build_nextstep_data(self.ds, prt, self._taxel,
                                                                     n_curves=self.curve_limit)
            print "%s: next step:          %d steps" % (prt, self.ns_in[prt].shape[1])

            self.curves[prt] = self.ds.record(prt, self._taxel)
            if self.curve_limit is not None:
                self.curves[prt] = self.curves[prt][0:self.curve_limit]
            self.force[prt], self.skin[prt], self.valid[prt] = build_multicurve(self.curves[prt])
            print "%s: curves:             %d" % (prt, len(self.curves[prt]))

        self.force_min = min([np.min(self.force[prt]) for prt in self.all_prts]) - self.range_epsilon
        self.force_max = max([np.max(self.force[prt]) for prt in self.all_prts]) + self.range_epsilon
        self.skin_min = min([np.min(self.skin[prt]) for prt in self.all_prts]) - self.range_epsilon
        self.skin_max = max([np.max(self.skin[prt]) for prt in self.all_prts]) + self.range_epsilon

        print
        print "minimum force:   %f" % self.force_min
        print "maximum force:   %f" % self.force_max
        print "minimum skin:    %f" % self.skin_min
        print "maximum skin:    %f" % self.skin_max

    def _build_discrete_data(self):
        self.discrete_force = {}
        self.discrete_skin = {}

        if self._taxel is None:
            return

        self.force_step = (self.force_max - self.force_min) / float(self.discrete_force_states)
        self.skin_step = (self.skin_max - self.skin_min) / float(self.discrete_skin_states)

        for prt in self.all_prts:
            self.discrete_force[prt] = np.asarray((self.force[prt] - self.force_min) / float(self.force_step),
                                                  dtype='int')
            self.discrete_skin[prt] = np.asarray((self.skin[prt] - self.skin_min) / float(self.skin_step),
                                                 dtype='int')

        print
        print "force step:   %f" % self.force_step
        print "skin step:    %f" % self.skin_step

    def from_discrete_force(self, discrete_force):
        return discrete_force * self.force_step + self.force_min

    def from_discrete_skin(self, discrete_skin):
        return discrete_skin * self.skin_step + self.skin_min

    def n_curves(self, prt):
        return len(self.curves[prt])

    def error(self, tr):
        err = {}
        for prt in self.all_prts:
            ns_skin_p = tr.predict(self.ns_in[prt])
            err['ns_'+prt] = 0.5 * np.mean((self.ns_skin[prt] - ns_skin_p)**2)

            skin_p = multistep_predict(tr.predict, self.force[prt], self.valid[prt], self.skin[prt][0, :])
            err['ms_'+prt] = multistep_error(skin_p, self.skin[prt], self.valid[prt], mean_err=True)
            mseps = multistep_error_per_sample(skin_p, self.skin[prt], self.valid[prt])
            err['failed_'+prt] = np.nonzero(mseps > 100)[0]
        return err

    def restarting_predict_error(self, tr, restart_steps):
        err = {}
        err['restart_steps'] = restart_steps
        for prt in self.all_prts:
            ns_skin_p = tr.predict(self.ns_in[prt])
            err['ns_'+prt] = 0.5 * np.mean((self.ns_skin[prt] - ns_skin_p)**2)

            skin_p = restarting_multistep_predict(tr.predict, self.force[prt], self.valid[prt], self.skin[prt],
                                                  restart_steps)
            err['ms_'+prt] = multistep_error(skin_p, self.skin[prt], self.valid[prt], mean_err=True)
            mseps = multistep_error_per_sample(skin_p, self.skin[prt], self.valid[prt])
            err['failed_'+prt] = np.nonzero(mseps > 100)[0]
        return err

    def print_error(self, err):
        if 'restart_steps' in err:
            print "restart steps: %d" % err['restart_steps']
        for prt in self.all_prts:
            print "%s:   next step: %.7f;  all steps: %.5f;  failed curves: %d" % (prt, err['ns_'+prt], err['ms_'+prt],
                                                                                   len(err['failed_'+prt]))

    def trim_to_valid(self, f, s, v):
        max_v = np.nonzero(v)
        last_valid = max_v[0][-1]
        return f[0:last_valid], s[0:last_valid]

    def get_discrete_fs_curve(self, prt, smpl):
        f = self.discrete_force[prt][:, smpl]
        s = self.discrete_skin[prt][:, smpl]
        v = self.valid[prt][:, smpl]
        return self.trim_to_valid(f, s, v)

    def get_fs_curve(self, prt, smpl):
        f = self.force[prt][:, smpl]
        s = self.skin[prt][:, smpl]
        v = self.valid[prt][:, smpl]
        return self.trim_to_valid(f, s, v)

    def predict_multicurve(self, predictor, force, skin, valid):
        skin_p = np.zeros(skin.shape)
        for smpl in range(force.shape[1]):
            status(smpl, force.shape[1], "Predicting")
            f, s = self.trim_to_valid(force[:, smpl], skin[:, smpl], valid[:, smpl])
            s_p = predictor(f)
            skin_p[0:s_p.shape[0], smpl] = s_p
        done()
        return skin_p

    def predict_multicurve_parallel(self, predictor, force, skin, valid):
        pool = multiprocessing.Pool()
        results = []

        skin_p = np.zeros(skin.shape)
        for smpl in range(force.shape[1]):
            f, s = self.trim_to_valid(force[:, smpl], skin[:, smpl], valid[:, smpl])
            results.append(pool.apply_async(predictor), f)
        for smpl in range(len(results)):
            s_p = results[smpl].get()
            skin_p[0:s_p.shape[0], smpl] = s_p
        return skin_p


class SkinNextstepWorkingset(object):

    def __init__(self, ws, tr, ns_in, ns_skin, batch_size=None):
        self.ws = ws
        self.tr = tr
        self.batch_size = batch_size

        self.ns_in = ns_in
        self.ns_skin = ns_skin

        if batch_size is not None:
            self.n_samples = self.ns_in.shape[1]
            self.n_batches = int(ceil(self.n_samples / self.batch_size))
            self.batch = 0

        if batch_size is not None:
            print "Number of samples:           ", self.n_samples
            print "Number of batches:           ", self.n_batches
            print "Batch size:                  ", self.batch_size

    def next_batch(self):
        self.batch = (self.batch + 1) % self.n_batches

    def get_batch(self, batch=None):
        if self.batch_size is None:
            return self.ns_in, self.ns_skin
        else:
            if batch is None:
                batch = self.batch
            ns_in = self.ns_in[:, batch*self.batch_size:(batch+1)*self.batch_size]
            ns_skin = self.ns_skin[batch*self.batch_size:(batch+1)*self.batch_size]
            return ns_in, ns_skin

    def error_wrapper(self, dummy):
        ns_in, ns_skin = self.get_batch()
        ns_skin_p = self.tr.predict(ns_in)
        return 0.5 * np.sum((ns_skin_p - ns_skin)**2)

    def gradient_wrapper(self, dummy):
        ns_in, ns_skin = self.get_batch()
        dEdF = self.tr.predict(ns_in) - ns_skin
        x_grad, w_grad = self.tr.gradient(ns_in)
        return w_grad.dot(dEdF)


class SkinMultistepWorkingset(object):

    def __init__(self, ws, tr, force, skin, valid, batch_size):
        self.ws = ws
        self.tr = tr
        self.batch_size = batch_size

        self.force = force
        self.skin = skin
        self.valid = valid

        self.n_samples = self.force.shape[1]
        self.n_batches = int(ceil(self.n_samples / float(self.batch_size)))
        self.batch = 0

        print "Number of multistep samples: ", self.n_samples
        print "Number of batches:           ", self.n_batches
        print "Batch size:                  ", self.batch_size

    def next_batch(self):
        self.batch = (self.batch + 1) % self.n_batches

    def get_fvs(self, batch=None):
        if batch is None:
            batch = self.batch
        f = self.force[:, batch*self.batch_size:(batch+1)*self.batch_size]
        v = self.valid[:, batch*self.batch_size:(batch+1)*self.batch_size]
        s = self.skin[:, batch*self.batch_size:(batch+1)*self.batch_size]
        return f, v, s

    def error_wrapper(self, dummy):
        f, v, s = self.get_fvs()
        s_p = multistep_predict(self.tr.predict, f, v, s[0, :])
        return multistep_error(s_p, s, v)

    def gradient_wrapper(self, dummy):
        f, v, s = self.get_fvs()
        return multistep_gradient(self.tr.predict_and_gradient, f, s, v)


class SkinMultistepAlternativeWorkingset(object):

    def __init__(self, ws, tr, force, skin, valid, steps, fallback_gradient_norm=None, random_sample_shift=False):
        self.ws = ws
        self.tr = tr
        self.steps = steps
        self.current_steps = steps

        self.fallback_gradient_norm = fallback_gradient_norm

        if random_sample_shift:
            shifts = np.random.randint(0, force.shape[0], size=force.shape[1])
            self.force = np.zeros(force.shape)
            self.skin = np.zeros(skin.shape)
            self.valid = np.zeros(valid.shape, dtype=valid.dtype)
            for smpl in range(force.shape[1]):
                self.force[:, smpl] = np.roll(force[:, smpl], shifts[smpl])
                self.skin[:, smpl] = np.roll(skin[:, smpl], shifts[smpl])
                self.valid[:, smpl] = np.roll(valid[:, smpl], shifts[smpl])
        else:
            self.force = force
            self.skin = skin
            self.valid = valid

        print "Steps:                       ", self.steps
        self.start_step = 0

    def next_batch(self):
        self.start_step = (self.start_step + self.steps / 2) % (self.force.shape[0] - 10)

    def get_fvs(self, start_step=None):
        if start_step is None:
            start_step = self.start_step
        f = self.force[start_step:start_step+self.steps, :]
        v = self.valid[start_step:start_step+self.steps, :]
        s = self.skin[start_step:start_step+self.steps, :]
        return f, v, s

    def error_wrapper(self, dummy):
        f, v, s = self.get_fvs()
        s_p = multistep_predict(self.tr.predict, f, v, s[0, :])
        return multistep_error(s_p, s, v)

    def gradient_wrapper(self, dummy):
        if self.fallback_gradient_norm is None:
            f, v, s = self.get_fvs()
            grad = multistep_gradient(self.tr.predict_and_gradient, f, s, v) / float(f.shape[1])
            return grad
        else:
            while True:
                f, v, s = self.get_fvs()
                f = f[0:self.current_steps, :]
                v = v[0:self.current_steps, :]
                s = s[0:self.current_steps, :]
                grad = multistep_gradient(self.tr.predict_and_gradient, f, s, v) / float(f.shape[1])
                if self.current_steps == 1:
                    break
                if np.linalg.norm(grad) < self.fallback_gradient_norm:
                    self.current_steps = min(self.steps, self.current_steps*2)
                    break
                else:
                    self.current_steps = max(1, int(self.current_steps/2))
                    # print "falling back to %d steps because |grad| = %g > %g" % (self.current_steps, np.linalg.norm(grad),
                    #                                                              self.fallback_gradient_norm)
            return grad


class SkinMultistepFullWorkingset(object):

    def __init__(self, ws, tr, force, skin, valid):
        self.ws = ws
        self.tr = tr

        self.force = force
        self.skin = skin
        self.valid = valid

    def get_fvs(self, start_step=None):
        f = self.force
        v = self.valid
        s = self.skin
        return f, v, s

    def error_wrapper(self, dummy):
        f,v,s = self.get_fvs()
        s_p = multistep_predict(self.tr.predict, f, v, s[0, :])
        return multistep_error(s_p, s, v)

    def gradient_wrapper(self, dummy):
        f,v,s = self.get_fvs()
        return multistep_gradient(self.tr.predict_and_gradient, f, s, v)
