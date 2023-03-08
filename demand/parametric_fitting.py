import numpy as np
from scipy import stats as stats


def _get_normal_paras(qty):
    dist_paras = stats.norm.fit(qty)
    return dist_paras


def _get_gamma_paras(qty):
    dist_paras = stats.gamma.fit(qty)
    return dist_paras


def _get_poisson_paras(qty):
    dist_paras = np.mean(qty)
    return dist_paras


def _get_lognormal_paras(qty):
    dist_paras = stats.lognorm.fit(qty)
    return dist_paras


_parametric_fit_map = {
    'normal': _get_normal_paras,
    'gamma': _get_gamma_paras,
    'poisson': _get_poisson_paras,
    'lognormal': _get_lognormal_paras
}
