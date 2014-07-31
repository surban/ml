import GPy as gpy
from GPy.util import Tango
from matplotlib import pyplot as plt, pyplot
import numpy as np
from ml.common.gifcreator import GIFCreator


class VarGP(object):

    def __init__(self, inp, tar,
                 use_data_variance=True,
                 kernel=None, normalize_x=True, normalize_y=True,
                 optimize=True, optimize_restarts=1,
                 var_kernel=None, var_optimize_restarts=1, var_normalize_x=False, var_normalize_y=True,
                 gp_parameters={},
                 cutoff_stds=None, std_adjust=1.0, std_power=1.0, min_std=0.01):

        # inp[feature, smpl]
        # tar[smpl]

        assert inp.ndim == 2
        assert tar.ndim == 1

        if kernel is None:
            kernel = lambda: gpy.kern.rbf(inp.shape[0])
        if var_kernel is None:
            var_kernel = kernel

        self.mngful = None
        self.cutoff_stds = cutoff_stds
        self.std_adjust = std_adjust
        self.std_power = std_power
        self.use_data_variance = use_data_variance
        self.min_std = min_std
        self.kernel = kernel
        self.var_kernel = var_kernel
        self.var_normalize_x = var_normalize_x
        self.var_normalize_y = var_normalize_y
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.gp_parameters = gp_parameters
        self.n_features = None

        self.gp = None
        self.gp_var_adjust = None

        self.train(inp, tar, optimize, optimize_restarts, var_optimize_restarts)

    def copy_gp_parameters(self, src, dest):
        for par in src._get_param_names():
            dest[par] = src[par]

    def train(self, inp, tar, optimize=False, optimize_restarts=1, var_optimize_restarts=1):
        # merge duplicates
        inp_mrgd, sta_mrgd, sta_var = self._merge_duplicates(inp, tar)

        # handle singleton data (workaround for non-positive definite matrices in GPy)
        if inp_mrgd.shape[1] == 1:
            inp_mrgd = np.resize(inp_mrgd, (inp_mrgd.shape[0], 2))
            sta_mrgd = np.resize(sta_mrgd, (2,))
            inp_mrgd[:, 1] = inp_mrgd[:, 0] + 1
            sta_mrgd[1] = sta_mrgd[0]
            singleton_fix = True
        else:
            singleton_fix = False

        n_samples = inp_mrgd.shape[1]
        self.n_features = inp_mrgd.shape[0]

        old_gp = self.gp
        old_gp_var_adjust = self.gp_var_adjust

        # fit GP on data
        self.gp = gpy.models.GPRegression(inp_mrgd.T, sta_mrgd[:, np.newaxis], self.kernel(),
                                          normalize_X=self.normalize_x, normalize_Y=self.normalize_y)
        self.gp.unconstrain('')
        for k, v in self.gp_parameters.iteritems():
            if v is not None:
                self.gp.constrain_fixed(k, v)
        self.gp.ensure_default_constraints()
        if old_gp is not None:
            self.copy_gp_parameters(old_gp, self.gp)
        if optimize:
            if optimize_restarts == 1:
                self.gp.optimize()
            else:
                self.gp.optimize_restarts(num_restarts=optimize_restarts)

        # estimate variance from predictions of first GP
        sta_mu, sta_var, _, _ = self.gp.predict(inp_mrgd.T, full_cov=False)
        sta_mrgd_var = np.zeros(n_samples)
        for n in range(n_samples):
            smpl_inp = np.all(np.isclose(inp, inp_mrgd[:, n:n+1]), axis=0)
            sta_mrgd_var[n] = np.mean((tar[smpl_inp] - sta_mu[n])**2)
        if singleton_fix:
            sta_mrgd_var = np.zeros(shape=sta_mrgd.shape)
        sta_mrgd_var -= sta_var[:, 0]

        # fit GP on variance
        self.gp_var_adjust = gpy.models.GPRegression(inp_mrgd.T, sta_mrgd_var[:, np.newaxis], self.var_kernel(),
                                                     normalize_X=self.var_normalize_x, normalize_Y=self.var_normalize_y)
        self.gp_var_adjust.unconstrain('')
        self.gp_var_adjust.ensure_default_constraints()
        if old_gp_var_adjust is not None:
            self.copy_gp_parameters(old_gp_var_adjust, self.gp_var_adjust)
        if optimize:
            if var_optimize_restarts == 1:
                self.gp_var_adjust.optimize()
            else:
                self.gp_var_adjust.optimize_restarts(num_restarts=var_optimize_restarts)

    def __str__(self):
        return str(self.gp)

    def _merge_duplicates(self, inp, tar):
        # inp[feature, smpl]
        # tar[smpl]

        assert tar.ndim == 1
        n_features = inp.shape[0]
        n_samples = inp.shape[1]

        xn = {}
        for smpl in range(n_samples):
            k = tuple(inp[:, smpl])
            if k in xn:
                xn[k].append(tar[smpl])
            else:
                xn[k] = [tar[smpl]]

        merged_inp = np.zeros((n_features, len(xn.keys())), dtype=inp.dtype)
        merged_tar = np.zeros(len(xn.keys()))
        merged_tar_var = np.zeros(len(xn.keys()))
        for smpl, (k, v) in enumerate(xn.iteritems()):
            merged_inp[:, smpl] = np.asarray(k)
            merged_tar[smpl] = np.mean(v)
            merged_tar_var[smpl] = np.var(v, ddof=1)

        return merged_inp, merged_tar, merged_tar_var

    def limit_meaningful_predictions(self, mngful_dist, inp, n_inp_values):
        # inp[feature, smpl]

        assert inp.ndim == 2
        assert inp.shape[0] == self.n_features
        assert self.n_features == 1, "limit_meaningful_predictions() requires one-dimensional input space"

        # determine meaningful predictions
        self.mngful = np.zeros(n_inp_values, dtype='bool')
        for i in range(mngful_dist):
            self.mngful[np.minimum(inp[0, :] + i, n_inp_values - 1)] = True
            self.mngful[np.maximum(inp[0, :] - i, 0)] = True

    def predict(self, inp, with_var_adjust_var=False):
        assert inp.ndim == 2
        assert inp.shape[0] == self.n_features

        tar, var_gp, _, _ = self.gp.predict(inp.T, full_cov=False)
        tar = tar[:, 0]
        var_gp = var_gp[:, 0]

        var_adjust, var_adjust_var, _, _ = self.gp_var_adjust.predict(inp.T, full_cov=False)
        var_adjust = var_adjust[:, 0]
        var_adjust_var = var_adjust_var[:, 0]

        if self.use_data_variance:
            var = var_gp + var_adjust
        else:
            var = var_gp
        var[var < 0] = 0
        std = np.sqrt(var)
        std = self.std_adjust * np.power(std, self.std_power)

        if with_var_adjust_var:
            return tar, std, var_gp, var_adjust, var_adjust_var
        else:
            return tar, std, var_gp, var_adjust

    def pdf_for_all_inputs(self, n_inp_values, n_tar_values):
        if self.n_features == 1:
            if isinstance(n_inp_values, tuple) or isinstance(n_inp_values, list):
                n_inp_values = n_inp_values[0]
            inp = np.arange(n_inp_values)[np.newaxis, :]
            return self.pdf(inp, n_tar_values)
        else:
            assert len(n_inp_values) == self.n_features
            inp_dims = [np.arange(n) for n in n_inp_values]
            inps = np.meshgrid(*inp_dims, indexing='ij')
            flat_inps = [inp.flatten() for inp in inps]
            inp_vec = np.vstack(tuple(flat_inps))
            flat_pdf = self.pdf(inp_vec, n_tar_values)
            pdf_shape = (flat_pdf.shape[0],) + inps[0].shape
            pdf = np.reshape(flat_pdf, pdf_shape)
            return pdf

    def pdf(self, inp, n_tar_values):
        # inp[feature, smpl]
        # pdf[tar, smpl]
        # tar[smpl]
        # std[smpl]

        assert inp.ndim == 2
        assert inp.shape[0] == self.n_features

        n_samples = inp.shape[1]

        # calculate predictions and associated variances
        tar, std, _, _ = self.predict(inp)
        std[std < self.min_std] = self.min_std

        # discretize output distribution
        pdf = np.zeros((n_tar_values, n_samples))
        tar_values = np.arange(n_tar_values)
        for smpl in range(n_samples):
            if self.mngful is not None:
                inp_val = inp[0, smpl]
                if not self.mngful[inp_val]:
                    continue

            ipdf = normal_pdf(tar_values[np.newaxis, :], np.atleast_1d(tar[smpl]), np.atleast_2d(std[smpl])**2)
            if self.cutoff_stds is not None:
                ipdf[np.abs(tar_values - tar[smpl]) > self.cutoff_stds * std[smpl]] = 0
            nfac = np.sum(ipdf)
            if nfac == 0:
                nfac = 1
            ipdf /= nfac
            pdf[:, smpl] = ipdf

        return pdf

    def plot(self, trn_inp, trn_tar, inp=None, n_inp_values=None, n_tar_values=None, rng=None, hmarker=None, ylim=None):
        assert self.n_features == 1, "plot() requires one-dimensional input space"

        if inp is None:
            assert n_inp_values, "plot() requires either inp or n_inp_values"
            inp = np.arange(n_inp_values)[np.newaxis, :]

        assert inp.ndim == 2 and inp.shape[0] == 1, "plot() requires one-dimensional input space"

        if rng is None:
            rng = [np.min(inp), np.max(inp)]

        tar, std, var_gp, var_data, var_data_var = self.predict(inp, with_var_adjust_var=True)

        # plot base GP
        plt.subplot(2, 2, 1)
        plt.hold(True)
        self.gp.plot(plot_limits=rng, which_data_rows=[], ax=plt.gca())
        if n_tar_values:
            plt.ylim(0, n_tar_values)
            plt.plot((n_tar_values - 5) * self.mngful, 'g')
        plt.plot(trn_inp[0, :], trn_tar, 'r.')
        if hmarker is not None:
            plt.axhline(hmarker, color='r')
        if ylim:
            plt.ylim(ylim)

        # plot GP with data variance
        plt.subplot(2, 2, 2)
        plt.hold(True)
        if n_tar_values:
            plt.ylim(0, n_tar_values)
        conf_gp = 2 * np.sqrt(var_gp)
        conf_total = 2 * std
        # cutoff = cutoff_stds * pred_std
        # plt.fill_between(pred_inp, pred_tar + cutoff, pred_tar - cutoff,
        #                  edgecolor=Tango.colorsHex['darkRed'], facecolor=Tango.colorsHex['lightRed'], alpha=0.4)
        plt.fill_between(inp[0, :], tar + conf_total, tar - conf_total,
                         edgecolor=Tango.colorsHex['darkBlue'], facecolor=Tango.colorsHex['lightBlue'], alpha=0.4)
        # plt.fill_between(inp[0, :], tar + conf_gp, tar - conf_gp,
        #                  edgecolor=Tango.colorsHex['darkPurple'], facecolor=Tango.colorsHex['lightPurple'], alpha=0.4)
        plt.plot(inp[0, :], tar, 'k')
        plt.plot(trn_inp[0, :], trn_tar, 'r.')
        #plt.errorbar(pred_inp, pred_tar, 2*np.sqrt(sta_mrgd_var), fmt=None)
        if hmarker is not None:
            plt.axhline(hmarker, color='r')
        if ylim:
            plt.ylim(ylim)

        # plot variance GP
        plt.subplot(2, 2, 3)
        plt.hold(True)

        sta_mu, sta_var, _, _ = self.gp.predict(trn_inp.T, full_cov=False)
        sta_mrgd_var = np.zeros(trn_inp.shape[1])
        for n in range(trn_inp.shape[1]):
            sta_mrgd_var[n] = np.mean((trn_tar[n] - sta_mu[n])**2)
        sta_mrgd_var -= sta_var[:, 0]

        if n_tar_values:
            plt.ylim(0, n_tar_values)
        var_data_conf = 2 * np.sqrt(var_data_var)
        # plt.fill_between(inp[0, :], var_data + var_data_conf, var_data - var_data_conf,
        #                  edgecolor=Tango.colorsHex['darkBlue'], facecolor=Tango.colorsHex['lightBlue'], alpha=0.4)
        plt.plot(inp[0, :], var_data, 'k')
        plt.plot(trn_inp[0, :], sta_mrgd_var, 'r.')

        if n_inp_values and n_tar_values:
            # plot output distribution
            pdf = self.pdf_for_all_inputs(n_inp_values, n_tar_values)
            plt.subplot(2, 2, 4)
            plt.imshow(pdf, origin='lower', aspect=1, interpolation='none')
            plt.colorbar()

    def prediction_grid_2d(self, xmin, xmax, xstep, ymin, ymax, ystep):
        assert self.n_features == 2, "plot_2d() requires two-dimensional input space"

        # construct input data
        xrng = np.arange(xmin, xmax, xstep)
        yrng = np.arange(ymin, ymax, ystep)
        xgrid, ygrid = np.meshgrid(xrng, yrng, indexing='ij')
        xflat = xgrid.flatten()
        yflat = ygrid.flatten()
        inp = np.vstack((xflat, yflat))

        # calculate predictions
        tar, std, var_gp, var_adjust = self.predict(inp)
        std_gp = np.sqrt(var_gp)

        # shape output
        tar_grid = np.reshape(tar, xgrid.shape)
        std_grid = np.reshape(std, xgrid.shape)
        std_gp_grid = np.reshape(std_gp, xgrid.shape)

        return tar_grid, std_grid, std_gp_grid

    def plot_2d(self, xmin, xmax, xstep, ymin, ymax, ystep, tar_min=None, tar_max=None,
                cmap='jet', limit_to_std=None):

        tar_grid, std_grid, std_gp_grid = self.prediction_grid_2d(xmin, xmax, xstep, ymin, ymax, ystep)

        if limit_to_std:
            tar_grid[std_gp_grid > limit_to_std] = 0

        # plot
        plt.subplot(2, 2, 1)
        plt.imshow(tar_grid.T, interpolation='none', origin='lower', cmap=cmap,
                   extent=(xmin, xmax, ymin, ymax),
                   aspect='auto', vmin=tar_min, vmax=tar_max)
        plt.colorbar()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title("mean")

        plt.subplot(2, 2, 3)
        plt.imshow(std_grid.T, interpolation='none', origin='lower', cmap=cmap,
                   extent=(xmin, xmax, ymin, ymax),
                   aspect='auto', vmin=0)
        plt.colorbar()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title("total std. deviation")

        plt.subplot(2, 2, 4)
        plt.imshow(std_gp_grid.T, interpolation='none', origin='lower', cmap=cmap,
                   extent=(xmin, xmax, ymin, ymax),
                   aspect='auto', vmin=0)
        plt.colorbar()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title("GP std. deviation")

        plt.subplot(2,2, 1)

    def animate_2d(self, filename,
                   xmin, xmax, xstep, ymin, ymax, ystep, tar_min, tar_max, n_steps=100, std_devs=2, duration=0.1,
                   cmap='jet'):
        tar_grid, std_grid, std_gp_grid = self.prediction_grid_2d(xmin, xmax, xstep, ymin, ymax, ystep)

        steps = range(n_steps)
        steps.extend(range(n_steps - 1, -1, -1))

        gc = GIFCreator()

        for step in steps:
            pos = step / float(n_steps - 1)

            min_grid = tar_grid - std_devs * std_grid
            max_grid = tar_grid + std_devs * std_grid
            step_grid = (1. - pos) * min_grid + pos * max_grid

            plt.clf()
            plt.imshow(step_grid.T, interpolation='none', origin='lower', cmap=cmap,
                       extent=(xmin, xmax, ymin, ymax),
                       aspect='auto', vmin=tar_min, vmax=tar_max)
            plt.colorbar()
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            plt.savefig(gc.add_image())

        gc.create(filename, duration=duration)


def normal_pdf(x, mu, sigma):
    assert x.ndim == 2
    assert mu.ndim == 1 and sigma.ndim == 2
    k = x.shape[0]
    assert k == mu.shape[0]
    assert k == sigma.shape[0] == sigma.shape[1]

    nf = (2.0*np.pi)**(-k/2.0) * np.linalg.det(sigma)**(-0.5)
    prec = np.linalg.inv(sigma)
    e = -0.5 * np.sum((x - mu[:, np.newaxis]) * np.dot(prec, (x - mu[:, np.newaxis])), axis=0)
    pdf = nf * np.exp(e)

    return pdf