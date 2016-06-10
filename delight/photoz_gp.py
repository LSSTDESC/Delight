
import numpy as np
import GPy

from GPy.core.model import Model
from GPy.likelihoods.gaussian import HeteroscedasticGaussian
from GPy.inference.latent_function_inference import exact_gaussian_inference
from GPy.core.parameterization.param import Param
from paramz import ObsAr

from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel


class PhotozGP(Model):
    """
    Photo-z Gaussian process, with physical kernel and mean function.
    Default: all parameters are variable except bands and likelihood/noise.
    """
    def __init__(self,
                 bands, redshifts, luminosities, types,
                 noisy_fluxes, flux_variances,
                 kern, mean_function,
                 prior_z_t=None,
                 prior_ell_t=None,
                 prior_t=None,
                 X_inducing=None,
                 fix_inducing_to_mean_prediction=True,
                 name='photozgp'):

        super(PhotozGP, self).__init__(name)

        assert bands.shape[1] == 1
        assert redshifts.shape[1] == 1
        assert luminosities.shape[1] == 1
        assert types.shape[1] == 1
        assert bands.shape[0] == bands.shape[0] and\
            bands.shape[0] == redshifts.shape[0] and\
            bands.shape[0] == luminosities.shape[0] and\
            bands.shape[0] == types.shape[0]

        self.bands = ObsAr(bands)

        self.redshifts = Param('redshifts', redshifts)
        self.link_parameter(self.redshifts)

        self.luminosities = Param('luminosities', luminosities)
        self.link_parameter(self.luminosities)

        self.types = Param('types', types)
        self.link_parameter(self.types)

        self.X = np.hstack((self.bands.values, self.redshifts.values,
                            self.luminosities.values, self.types.values))
        assert self.X.shape[1] == 4
        self.num_data, self.input_dim = self.X.shape

        assert noisy_fluxes.ndim == 1
        self.Y = ObsAr(noisy_fluxes[:, None])
        Ny, self.output_dim = self.Y.shape

        if isinstance(mean_function, Photoz_mean_function) or\
                isinstance(kern, Photoz_kernel):
            assert isinstance(mean_function, Photoz_mean_function)
            assert isinstance(kern, Photoz_kernel)
            assert mean_function.g_AB == kern.g_AB
            assert mean_function.DL_z == kern.DL_z
        self.mean_function = mean_function
        #  No need to link mean_function because it has no parameters
        self.kern = kern
        self.link_parameter(self.kern)

        self.Y_metadata = {
            'output_index': np.arange(Ny)[:, None],
            'variance': flux_variances
        }

        self.likelihood = HeteroscedasticGaussian(self.Y_metadata)

        self.inference_method =\
            exact_gaussian_inference.ExactGaussianInference()

        self.posterior = None

        self.prior_z_t = prior_z_t
        if prior_z_t is not None:
            self.link_parameter(self.prior_z_t)
        self.prior_ell_t = prior_ell_t
        if prior_ell_t is not None:
            self.link_parameter(self.prior_ell_t)
        self.prior_t = prior_t
        if prior_t is not None:
            self.link_parameter(self.prior_t)

        self.X_inducing = None
        if X_inducing is not None:
            assert X_inducing.shape[1] == self.input_dim
            self.X_inducing = X_inducing
            self.Y_inducing = np.zeros((X_inducing.shape[0], 1))
            if not fix_inducing_to_mean_prediction:  # also sample Y_inducing!
                # Otherwise Y_inducing will be set to mean GP prediction
                self.link_parameter(self.Y_inducing)

    def set_bands(self, bands):
        """Set bands"""
        assert bands.shape[1] == 1
        assert bands.shape[0] == self.redshifts.shape[0] and\
            bands.shape[0] == self.types.shape[0] and\
            bands.shape[0] == self.luminosities.shape[0]
        self.update_model(False)
        self.bands = ObsAr(bands)
        self.update_model(True)

    def set_redshifts(self, redshifts):
        """Set redshifts"""
        assert redshifts.shape[1] == 1
        assert redshifts.shape[0] == self.bands.shape[0] and\
            redshifts.shape[0] == self.types.shape[0] and\
            redshifts.shape[0] == self.luminosities.shape[0]
        self.update_model(False)
        index = self.redshifts._parent_index_
        self.unlink_parameter(self.redshifts)
        self.redshifts = Param('redshifts', redshifts)
        self.link_parameter(self.redshifts, index=index)
        self.update_model(True)

    def set_luminosities(self, luminosities):
        """Set luminosities"""
        assert luminosities.shape[1] == 1
        assert luminosities.shape[0] == self.bands.shape[0] and\
            luminosities.shape[0] == self.types.shape[0] and\
            luminosities.shape[0] == self.redshifts.shape[0]
        self.update_model(False)
        index = self.luminosities._parent_index_
        self.unlink_parameter(self.luminosities)
        self.luminosities = Param('luminosities', luminosities)
        self.link_parameter(self.luminosities, index=index)
        self.update_model(True)

    def set_types(self, types):
        """Set types"""
        assert types.shape[1] == 1
        assert types.shape[0] == self.bands.shape[0] and\
            types.shape[0] == self.redshifts.shape[0] and\
            types.shape[0] == self.luminosities.shape[0]
        self.update_model(False)
        index = self.types._parent_index_
        self.unlink_parameter(self.types)
        self.types = Param('types', types)
        self.link_parameter(self.types, index=index)
        self.update_model(True)

    def parameters_changed(self):
        """If parameters changed, compute gradients"""
        self.X = np.hstack((self.bands.values, self.redshifts.values,
                            self.luminosities.values, self.types.values))
        assert self.X.shape[1] == 4

        self.gp_posterior, self.gp_log_marginal_likelihood, self.gp_grad_dict\
            = self.inference_method.inference(self.kern, self.X,
                                              self.likelihood,
                                              self.Y, self.mean_function,
                                              self.Y_metadata)

        self._log_marginal_likelihood = self.gp_log_marginal_likelihood

        if self.X_inducing is not None:
            if fix_inducing_to_mean_prediction:
                mu, var = self._raw_predict(X_inducing, full_cov=False)
                self.Y_inducing = mu
            else:
                self._log_marginal_likelihood +=\
                    self.log_predictive_density(X_inducing, Y_inducing)
                raise NotImplementedError("Uncertain inducing not implemented")
                #  self.Y_inducing.gradient =  #  TODO : update gradients

        self.mean_function.update_gradients(self.gp_grad_dict['dL_dm'], self.X)

        self.kern.update_gradients_full(self.gp_grad_dict['dL_dK'], self.X)

        gradX = self.mean_function.gradients_X(self.gp_grad_dict['dL_dm'].T,
                                               self.X)\
            + self.kern.gradients_X_diag(self.gp_grad_dict['dL_dK'],
                                         self.X)

        if not self.redshifts.is_fixed:
            self.redshifts.gradient[:] = 0
        if not self.luminosities.is_fixed:
            self.luminosities.gradient[:] = 0
        if not self.types.is_fixed:
            self.types.gradient[:] = 0

        if not self.redshifts.is_fixed:
            self.redshifts.gradient[:, 0] += 2*gradX[:, 1]
            if self.prior_z_t is not None:
                self._log_marginal_likelihood +=\
                    self.prior_z_t.lnprob(self.redshifts, self.types)
                self.prior_z_t.update_gradients(self.redshifts, self.types)
                self.redshifts.gradient +=\
                    self.prior_z_t.grad_z(self.redshifts, self.types)
                if not self.types.is_fixed:
                    self.types.gradient +=\
                        self.prior_z_t.grad_t(self.redshifts, self.types)

        if not self.luminosities.is_fixed:
            self.luminosities.gradient[:, 0] += 2*gradX[:, 2]
            if self.prior_ell_t is not None:
                self._log_marginal_likelihood +=\
                    self.prior_ell_t.lnprob(self.luminosities, self.types)
                self.prior_ell_t.update_gradients(self.luminosities,
                                                  self.types)
                self.luminosities.gradient +=\
                    self.prior_ell_t.grad_ell(self.luminosities, self.types)
                if not self.types.is_fixed:
                    self.types.gradient +=\
                        self.prior_ell_t.grad_t(self.luminosities, self.types)

        if not self.types.is_fixed:
            self.types.gradient[:, 0] += 2*gradX[:, 3]
            if self.prior_t is not None:
                self._log_marginal_likelihood +=\
                    self.prior_t.lnprob(self.types)
                self.prior_t.update_gradients(self.types)
                self.types.gradient +=\
                    self.prior_t.grad_t(self.types)

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def _raw_predict(self, Xnew, full_cov=False):
        """
        For making predictions, without normalization or likelihood
        """
        mu, var = self.posterior._raw_predict(
            kern=self.kern, Xnew=Xnew, pred_var=self.X, full_cov=full_cov)
        mu += self.mean_function.f(Xnew)
        return mu, var

    def log_predictive_density(self, x_test, y_test, Y_metadata=None):
        """
        Calculation of the log predictive density

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param x_test: test locations (x_{*})
        :type x_test: (Nx1) array
        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param Y_metadata: metadata associated with the test points
        """
        mu_star, var_star = self._raw_predict(x_test)
        return self.likelihood.log_predictive_density(y_test, mu_star,
                                                      var_star,
                                                      Y_metadata=Y_metadata)
