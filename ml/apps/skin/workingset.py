from math import ceil
import numpy as np
import matplotlib.pyplot as plt

from ml.apps.skin.timeseries import *
from ml.simple.table import *
from ml.datasets.skin import SkinDataset
from ml.common.util import ParameterHistory
from ml.common.test import check_gradient, check_directional_gradient
from climin.gd import GradientDescent
from climin.bfgs import Bfgs, Lbfgs
from climin.rmsprop import RmsProp


class SkinWorkingset(object):

    force_min = 0
    force_step = 0.1
    force_max = 25
    skin_min = 0
    skin_step = 0.02
    skin_max = 2

    def __init__(self, ds_name, taxel):
        self.ds = SkinDataset(ds_name)
        self.ds.print_statistics()
        self.taxel = taxel
        print "Using taxel ", self.taxel
        print
        self.build_data()

    def build_data(self):
        self.ns_in = {}
        self.ns_skin = {}
        self.curves = {}
        self.force = {}
        self.skin = {}
        self.valid = {}

        for prt in ['trn', 'val', 'tst']:
            self.ns_in[prt], self.ns_skin[prt] = build_nextstep_data(self.ds, prt, self.taxel)
            print "%s: next step:          %d steps" % (prt, self.ns_in[prt].shape[1])

            self.curves[prt] = self.ds.record(prt, self.taxel)
            self.force[prt], self.skin[prt], self.valid[prt] = build_multicurve(self.curves[prt])
            print "%s: curves:             %d" % (prt, len(self.curves[prt]))

    def error(self, tr):
        err = {}
        for prt in ['trn', 'val', 'tst']:
            ns_skin_p = tr.predict(self.ns_in[prt])
            err['ns_'+prt] = 0.5 * np.mean((self.ns_skin[prt] - ns_skin_p)**2)

            skin_p = multistep_predict(tr.predict, self.force[prt], self.valid[prt], self.skin[prt][0, :])
            err['ms_'+prt] = multistep_error(skin_p, self.skin[prt], self.valid[prt], mean_err=True)
            mseps = multistep_error_per_sample(skin_p, self.skin[prt], self.valid[prt])
            err['failed_'+prt] = np.nonzero(mseps > 100)[0]

        return err


class SkinMultistepWorkingset(object):

    def __init__(self, ws, tr, force, skin, valid, batch_size):
        self.ws = ws
        self.tr = tr
        self.batch_size = batch_size

        self.force = force
        self.skin = skin
        self.valid = valid

        self.n_samples = self.force.shape[1]
        self.n_batches = int(ceil(self.n_samples / self.batch_size))
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
        f,v,s = self.get_fvs()
        s_p = multistep_predict(self.tr.predict, f, v, s[0, :])
        return multistep_error(s_p, s, v)

    def gradient_wrapper(self, dummy):
        f,v,s = self.get_fvs()
        return multistep_gradient(self.tr.predict_and_gradient, f, s, v)


class SkinMultistepAlternativeWorkingset(object):

    def __init__(self, ws, tr, force, skin, valid, steps):
        self.ws = ws
        self.tr = tr
        self.steps = steps

        self.force = force
        self.skin = skin
        self.valid = valid

        print "Steps:                       ", self.steps
        self.start_step = 0

    def next_batch(self):
        self.start_step = (self.start_step + self.steps + 1) % (self.force.shape[0] - 10)

    def get_fvs(self, start_step=None):
        if start_step is None:
            start_step = self.start_step
        f = self.force[start_step:start_step+self.steps, :]
        v = self.valid[start_step:start_step+self.steps, :]
        s = self.skin[start_step:start_step+self.steps, :]
        return f, v, s

    def error_wrapper(self, dummy):
        f,v,s = self.get_fvs()
        s_p = multistep_predict(self.tr.predict, f, v, s[0, :])
        return multistep_error(s_p, s, v)

    def gradient_wrapper(self, dummy):
        f,v,s = self.get_fvs()
        return multistep_gradient(self.tr.predict_and_gradient, f, s, v)


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
