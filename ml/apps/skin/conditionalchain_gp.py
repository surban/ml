import ml.common.plot
import numpy as np
import matplotlib.pyplot as plt

from ml.apps.skin.workingset import *
from ml.simple.conditional_chain import *
from ml.common.util import standard_cfg


cfg, plot_dir = standard_cfg(prepend_scriptname=False)

# load dataset
ws = SkinWorkingset(cfg.dataset_name,
                    discrete_force_states=cfg.discrete_force_states,
                    discrete_skin_states=cfg.discrete_skin_states)

all_mse_val = []

for taxel in ws.ds.available_taxels():
    ws.taxel = taxel
    taxel_str = "%d%d" % (taxel[0], taxel[1])

    # train
    sc = ConditionalChain(ws.discrete_skin_states, ws.discrete_force_states)
    sc.train_gp_var(ws.discrete_skin['trn'], ws.discrete_force['trn'], ws.valid['trn'],
                    cutoff_stds=cfg.cutoff_stds, mngful_dist=int(cfg.mngful_dist), std_adjust=cfg.std_adjust,
                    gp_kernel=eval('lambda: gpy.kern.%s(1)' % cfg.kernel),
                    gp_normalize_x=False, gp_normalize_y=cfg.normalize_y, gp_optimize=True,
                    gp_parameters={})

    # predict
    force, skin, valid = ws.discrete_force['tst'], ws.discrete_skin['tst'], ws.valid['tst']
    skin_p = ws.predict_multicurve(lambda f: sc.most_probable_states(f)[0], force, skin, valid)

    # calculate error
    mse = multistep_error(ws.from_discrete_skin(skin_p), ws.from_discrete_skin(skin), valid, mean_err=True)
    all_mse_val.append(mse)
    with open(plot_dir + "/%s_err.txt" % taxel_str, 'w') as of:
        of.write("%f\n" % mse)

    # plot
    n_plot_smpls = 100
    plt.figure(figsize=(40, 20))
    plot_multicurve_time(ws.from_discrete_force(force[:, 0:n_plot_smpls]),
                         ws.from_discrete_skin(skin[:, 0:n_plot_smpls]),
                         valid[:, 0:n_plot_smpls],
                         ws.from_discrete_skin(skin_p[:, 0:n_plot_smpls]),
                         timestep=ws.ds.timestep)
    plt.savefig(plot_dir + "/%s.pdf" % taxel_str)

# average error
all_mse = np.mean(all_mse_val)
with open(plot_dir + "err.txt", 'w') as of:
    of.write("%f\n" % all_mse)


