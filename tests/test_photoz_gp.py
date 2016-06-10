
import pytest
import numpy as np

from delight.priors import Rayleigh, Schechter, Kumaraswamy
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
from delight.utils import random_X_bztl,\
    random_filtercoefs, random_linecoefs, random_hyperparams

from scipy.misc import derivative
from copy import deepcopy

NREPEAT = 4
size = 10
numBands = 5
numLines = 3
numCoefs = 3
relative_accuracy = 0.01


@pytest.fixture(params=[False])
def create_p_z_t(request):
    if request.param is False:
        return None
    else:
        alpha0, alpha1 = np.random.uniform(0., 2., size=2)
        return Rayleigh(alpha0, alpha1)


@pytest.fixture(params=[False])
def create_p_ell_t(request):
    if request.param is False:
        return None
    else:
        ellStar, alpha0, alpha1 = np.random.uniform(0., 2., size=2)
        return Schechter(ellStar, alpha0, alpha1)


@pytest.fixture(params=[False])
def create_p_t(request):
    if request.param is False:
        return None
    else:
        alpha0, alpha1 = np.random.uniform(0., 2., size=2)
        return Kumaraswamy(alpha0, alpha1)


@pytest.fixture()
def create_gp(create_p_ell_t, create_p_z_t, create_p_t):

    X = random_X_bztl(size, numBands=numBands)
    bands, redshifts, types, luminosities = np.split(X, 4, axis=1)

    fcoefs_amp, fcoefs_mu, fcoefs_sig \
        = random_filtercoefs(numBands, numCoefs)
    lines_mu, lines_sig = random_linecoefs(numLines)
    var_T, alpha_C, alpha_L, alpha_T = random_hyperparams()

    kern = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                         lines_mu, lines_sig, var_T,
                         alpha_C, alpha_L, alpha_T)

    mean_function = Photoz_mean_function()

    noisy_fluxes = np.random.uniform(low=0., high=1., size=size)
    flux_variances = np.random.uniform(low=0., high=1., size=size)

    prior_ell_t = create_p_ell_t
    assert(prior_ell_t is None or isinstance(prior_ell_t, Schechter))
    prior_z_t = create_p_z_t
    assert(prior_z_t is None or isinstance(prior_z_t, Rayleigh))
    prior_t = create_p_t
    assert(prior_t is None or isinstance(prior_t, Kumaraswamy))

    gp = PhotozGP(
        bands, redshifts, luminosities, types,
        noisy_fluxes, flux_variances,
        kern, mean_function,
        prior_z_t=prior_z_t,
        prior_ell_t=prior_ell_t,
        prior_t=prior_t,
        X_inducing=None,
        fix_inducing_to_mean_prediction=True
        )
    gp.set_types(types)
    return gp


def test_gradients(create_gp):
    gp = create_gp
    assert(isinstance(gp, PhotozGP))

    v1 = gp.kern.alpha_L.gradient

    def f_alpha_L(v):
        gp2 = deepcopy(gp)
        gp2.kern.set_alpha_L(v)
        return gp2._log_marginal_likelihood
    v2 = derivative(f_alpha_L, gp.kern.alpha_L.values,
                    dx=0.01*gp.kern.alpha_L.values)
    if np.abs(v1) > 1e-14 and np.abs(v2) > 1e-14:
        assert abs(v1/v2-1) < relative_accuracy

    v1 = gp.kern.alpha_C.gradient

    def f_alpha_C(v):
        gp2 = deepcopy(gp)
        gp2.kern.set_alpha_C(v)
        return gp2._log_marginal_likelihood
    v2 = derivative(f_alpha_C, gp.kern.alpha_C.values,
                    dx=0.01*gp.kern.alpha_C.values)
    assert abs(v1/v2-1) < relative_accuracy

    v1 = gp.kern.var_T.gradient

    def f_var_T(v):
        gp2 = deepcopy(gp)
        gp2.kern.set_var_T(v)
        return gp2._log_marginal_likelihood
    v2 = derivative(f_var_T, gp.kern.var_T.values,
                    dx=0.01*gp.kern.var_T.values)
    assert abs(v1/v2-1) < relative_accuracy

    v1 = gp.kern.alpha_T.gradient

    def f_alpha_T(v):
        gp2 = deepcopy(gp)
        gp2.kern.set_alpha_T(v)
        return gp2._log_marginal_likelihood
    v2 = derivative(f_alpha_T, gp.kern.alpha_T.values,
                    dx=0.01*gp.kern.alpha_T.values)
    assert abs(v1/v2-1) < relative_accuracy

    for dim in range(size):

        v1 = gp.types.gradient[dim]

        def f_t(t):
            gp2 = deepcopy(gp)
            v = gp2.types.values
            v[dim] = t
            gp2.set_types(v)
            return gp2._log_marginal_likelihood
        v2 = derivative(f_t, gp.types.values[dim],
                        dx=0.01*gp.types.values[dim])
        assert abs(v1/v2-1) < relative_accuracy

        v1 = gp.luminosities.gradient[dim]

        def f_ell(t):
            gp2 = deepcopy(gp)
            v = gp2.luminosities.values
            v[dim] = t
            gp2.set_luminosities(v)
            return gp2._log_marginal_likelihood
        v2 = derivative(f_ell, gp.luminosities.values[dim],
                        dx=0.01*gp.luminosities.values[dim])
        assert abs(v1/v2-1)

        #  TODO: add redshifts
