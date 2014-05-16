tmpl = """
# SUBMIT: runner=python -m ml.apps.skin.conditionalchain_gp
# SBATCH --mem=2048
# SBATCH --time=1:00:00
dataset_name = "bidir_small"
discrete_force_states = $discrete_force_states$
discrete_skin_states = $discrete_skin_states$
kernel = "$kernel$"
std_adjust = $std_adjust$
cutoff_stds = $cutoff_stds$
mngful_dist = $mngful_dist$
normalize_y = $normalize_y$
"""

import submit
submit.remove_index_dirs()
submit.gridsearch(name="$CFG_INDEX$/cfg.py",
                  template=tmpl,
                  parameter_ranges={"discrete_force_states":    [50, 100, 300],
                                    "discrete_skin_states":     [50, 100, 300],
                                    "kernel":                   ['rbf', 'Matern32', 'Matern52'],
                                    "std_adjust":               "0.6:0.3:2.5",
                                    "cutoff_stds":              "0.6:0.3:2.5",
                                    "mngful_dist":              "1:4:16",
                                    "normalize_y":              ["True", "False"],
                                    })

