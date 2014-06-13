
# SUBMIT: runner=python -m ml.apps.skin.conditionalchain_gp
# SBATCH --mem=3096
# SBATCH --time=1:00:00
dataset_name = "bidir_small"
discrete_force_states = 100
discrete_skin_states = 50
kernel = "Matern52"
std_adjust = 1.8
cutoff_stds = 0.9
mngful_dist = 1.0
normalize_y = False
