import warnings
from typing import List, Union
import numpy as np
from scipy import stats
from scipy.linalg import cholesky
from scipy.linalg.lapack import dtrtri
import statsmodels.api as sm
from ..utils import maxprod, get_convex_min
from .coordinate_descent_glm import cd_ols
from .coordinate_descent_gaussian import cd_gaussian
from ..base import BaseRegressor


class NCVREG(BaseRegressor):
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 sigma: np.ndarray = None,
                 lmbd: List = None,
                 family: str = 'gaussian',
                 penalty: str = 'mcp',
                 gamma: float = 3,
                 alpha: float = 1,
                 nlambda: int = 100,
                 eps: float = 1e-4,
                 max_iter: int = 1000,
                 convex: bool = True,
                 penalty_factor: np.ndarray = None,
                 criterion: str = 'aic'):
        """
        Fit a Regularized regression model using MCP or SCAD Penalty

        Parameters
        ----------
        X : np.ndarray,
            The design matrix or predictors.

        y : np.ndarray,
            The response variable.

        sigma : np.ndarray, optional,
            The covariance matrix of the response variable. Default is None.

        lmbd : list, optional,
            A user specified list of lambda values. If not provided, the function will generate a sequence of lambda
            values based on the length of nlambda.

        family : str, optional,
            Response variable prior distribution. Options are 'gaussian', 'binomial', 'poisson'.

        penalty : str, optional,
            Penalty to be used. Options are 'mcp', 'scad', 'l1', 'l2', 'elasticnet'.

        gamma : float, optional,
            The gamma parameter for the MCP and SCAD penalties. Default is 3.

        alpha : float, optional,
            The tuning parameter for Mnet estimation, which controls the relative contribution of the MCP/SCAD penalty,
            and the L2 penalty. Default is 1. 'alpha = 1' corresponds to the MCP/SCAD penalty only. 'alpha = 0' corresponds
            to the L2 or ridge regression only.

        nlambda : int, optional,
            The number of lambda values to generate. Default is 100.

        eps : float, optional,
            Convergence threshold. Default is 1e-4.

        max_iter : int, optional,
            Maximum number of iterations. Default is 10000.

        convex : bool, optional,
            Calculate the index for which objective function ceases to be locally convex. Default is True.

        penalty_factor : np.ndarray, optional,
            Penalty factors for each predictor. Default will be 1 for all predictors.

        criterion : str, optional,
            Model selection criteria. Default is Akaike Information Criterion (AIC).
        """
        super().__init__()
        self.X = X
        self.y = y
        self.sigma = sigma
        self.lmbd = lmbd
        self.family = family
        self.criterion = criterion
        self.penalty = penalty
        self.alpha = alpha
        self.nlambda = nlambda
        self.eps = eps
        self.max_iter = max_iter
        self.convex = convex
        self.gamma = 3.7 if self.penalty == 'scad' else gamma
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.lambda_min = 0.001 if self.n > self.p else 0.05
        self.dfmax = self.p + 1
        self.penalty_factor = penalty_factor if penalty_factor is not None else np.repeat(1, self.p)

        # Parameter Checks
        if self.family not in ['gaussian', 'binomial', 'poisson']:
            raise ValueError('Invalid family')
        if self.penalty not in ['mcp', 'scad', 'l1', 'l2']:
            raise ValueError('Invalid penalty')

        if not isinstance(self.X, np.ndarray):
            try:
                self.X = np.array(self.X)
            except Exception:
                raise ValueError('X must be an np.ndarray or coercible to one')

        if not isinstance(self.y, np.ndarray):
            try:
                self.y = np.array(self.y)
            except Exception:
                raise ValueError('y must be an np.ndarray or coercible to one')

        if self.y.ndim > 1:
            raise ValueError('y must be a 1D array')

        try:
            self.X = self.X.astype(float)
        except Exception:
            raise ValueError('X cannot be coerced to a float array')

        try:
            self.y = self.y.astype(float)
        except Exception:
            raise ValueError('y cannot be coerced to a float array')

        if self.gamma <= 1 and self.penalty == 'mcp':
            raise ValueError('gamma must be greater than 1 for MCP penalty')
        if self.gamma <= 2 and self.penalty == 'scad':
            raise ValueError('gamma must be greater than 2 for SCAD penalty')
        if self.nlambda < 2:
            raise ValueError('nlambda must be at least 2')
        if self.alpha <= 0:
            raise ValueError('alpha must be greater than 0. Choose a small positive number')
        if len(self.penalty_factor) != self.p:
            raise ValueError('Dimensions do not match. penalty_factor must be of length p')
        if not isinstance(self.penalty_factor, np.ndarray):
            try:
                self.penalty_factor = np.array(self.penalty_factor)
            except Exception as e:
                raise TypeError('penalty_factor must be an np.ndarray or coercible to one. %s' % e)
        if self.family == 'binomial' and len(np.unique(self.y)) > 2:
            raise ValueError('y must be binary for binomial family')
        # Check that the only values for y in a binomial family are 0 and 1
        if self.family == 'binomial' and not all(np.isin(self.y, [0, 1])):
            raise ValueError('y must be binary for binomial family')
        if len(self.y) != self.n:
            raise ValueError('Dimensions do not match. y must be of length n')
        if np.isnan(self.y).any() or np.isnan(self.X).any():
            raise ValueError('X and y must not contain NaNs')

        # Set the degrees of freedom for the model
        rank = np.linalg.matrix_rank(self.X)
        self.df_model = float(rank)

    def _standardize(self):
        # Standardize the data
        self.X_std = stats.zscore(self.X, axis=1)

        if self.family == 'gaussian':
            self.y_std = self.y - np.mean(self.y)
        else:
            self.y_std = self.y

    def _get_sigma(self, sigma):
        """
        Returns sigma (matrix, nobs by nobs), the covariance of the error/noise terms, for GLS and the inverse of its
        Cholesky decomposition.  Handles dimensions and checks integrity.
        If sigma is None, returns None, None. Otherwise returns sigma,
        cholsigmainv.
        """
        if sigma is None:
            return None, None
        sigma = np.asarray(sigma).squeeze()
        if sigma.ndim == 0:
            sigma = np.repeat(sigma, self.n)
        if sigma.ndim == 1:
            if sigma.shape != (self.n,):
                raise ValueError("Sigma must be a scalar, 1d of length %s or a 2d "
                                 "array of shape %s x %s" % (self.n, self.n, self.n))
            cholsigmainv = 1 / np.sqrt(sigma)
        else:
            if sigma.shape != (self.n, self.n):
                raise ValueError("Sigma must be a scalar, 1d of length %s or a 2d "
                                 "array of shape %s x %s" % (self.n, self.n, self.n))
            cholsigmainv, info = dtrtri(cholesky(sigma, lower=True),
                                        lower=True, overwrite_c=True)
            if info > 0:
                raise np.linalg.LinAlgError('Cholesky decomposition of sigma '
                                            'yields a singular matrix')
            elif info < 0:
                raise ValueError('Invalid input to dtrtri (info = %d)' % info)
        return sigma, cholsigmainv

    def _get_lambda(self):
        idx = np.where(self.penalty_factor != 0)[0]

        # Fit OLS model to get lambda_max
        if self.family == 'gaussian':
            fam = sm.families.Gaussian()
        elif self.family == 'binomial':
            fam = sm.families.Binomial()
        else:
            fam = sm.families.Poisson()

        if len(idx) != self.p:
            fit = sm.GLM(self.y, self.X_std[:, idx], family=fam).fit()
        else:
            fit = sm.GLM(self.y, np.ones(self.n), family=fam).fit()

        # Find Z-max
        if self.family == 'gaussian':
            zmax = maxprod(self.X_std, fit.resid_deviance, idx, self.penalty_factor) / self.n
        else:
            zmax = maxprod(self.X_std, fit.resid_working * fit.weights, idx, self.penalty_factor) / self.n

        lambda_max = zmax / self.alpha

        if self.lambda_min == 0:
            self.lmbd = np.exp(np.linspace(np.log(lambda_max), np.log(0.001 * lambda_max), self.nlambda - 1)).append(0)
        else:
            self.lmbd = np.exp(np.linspace(np.log(lambda_max), np.log(self.lambda_min * lambda_max), self.nlambda))

        if len(idx) != self.p:
            self.lmbd = self.lmbd * 1.000001

        self.lmbd = np.sort(self.lmbd)

    def _fit_gaussian(self):
        # Initialize
        res = cd_gaussian(self.X_std, self.y_std, self.penalty, self.lmbd, self.eps,
                          self.max_iter, self.gamma, self.penalty_factor, self.alpha,
                          self.dfmax)
        return res

    def _fit_glm(self):
        # Initialize
        res = cd_ols(self.X_std, self.y, self.family, self.penalty, self.lmbd,
                     self.eps, self.max_iter, self.gamma, self.penalty_factor,
                     self.alpha, self.dfmax)
        return res

    def _get_coef(self, lmbd: Union[float, list]=None, which=None, drop=True, **kwargs):
        if which == None:
            which = len(self.lmbd)

        if isinstance(lmbd, float):
            lmbd = [lmbd]

        if lmbd:
            if max(lmbd) > max(self.lmbd) or min(lmbd) < min(self.lmbd):
                raise ValueError('Supplied lambda values are outside the range of the fitted model')
            idx = np.interp(self.lmbd, self.lmbd, lmbd)
            l = int(np.floor(idx))
            r = int(np.ceil(idx))
            w = idx % 1
            beta = ((1 - w) * self.b[:, l]) + (w * self.b[:, r])
        else:
            beta = self.b[:, :which]
        if drop:
            return np.squeeze(beta)
        else:
            return beta

    def _loglike(self, params):
        """
        Compute the value of the log-likelihood function based on family and parameters

        Parameters
        ----------
        params : arraylike,
            model parameters.

        Returns
        -------
        float
            value of the log-likelihood function.
        """

        nobs2 = self.n / 2
        yhat = np.squeeze(np.matmul(self.X_std, params))
        resid = self.y_std - yhat
        SSR = np.sum(resid**2)
        llf = -np.log(SSR) * nobs2 # concentrated likelihood e.g. -(n/2)log((y-yhat)'(y-yhat))
        llf -= (1 + np.log(2*np.pi / self.n)) * nobs2 # likelihood constant
        if np.any(self.sigma): # get covariance matrix of the parameters
            if self.sigma.ndim == 2:
                det = np.linalg.slogdet(self.sigma)
                llf -= det[1] / 2
            else:
                llf -= np.sum(np.log(self.sigma)) / 2
        return llf

    def fit(self):
        """
        Fit the model

        Parameters
        ----------

        Returns
        -------

        """

        # Standardize the data
        self._standardize()
        self.sigma, self.cholsigmainv = self._get_sigma(self.sigma)

        # Get lambda
        if self.lmbd is None:
            self._get_lambda()
        else:
            self.nlambda = len(self.lmbd)
            if self.nlambda == 1:
                warnings.warn('Only one lambda value provided. Will use this value for fitting')
            else:
                # Sort lambda values from smallest to largest
                self.lmbd = np.sort(self.lmbd)

        # Fit the model
        if self.family == 'gaussian':
            self.model = self._fit_gaussian()
            self.a = np.repeat(np.mean(self.y), self.nlambda)
            self.b = self.model['beta'].transpose()
            self.loss = self.model['loss']
            self.eta = np.array(self.model['Eta']).transpose() + np.mean(self.y)
            self.iter_idx = self.model['iter_idx']
        else:
            self.model = self._fit_glm()
            self.a = self.model['alpha']
            self.b = self.model['beta'].transpose()
            self.loss = self.model['loss']
            self.eta = self.model['Eta']
            self.iter_idx = self.model['iter_idx']

        # Eliminate saturated lambda values
        idxs = np.where(~np.isnan(self.iter_idx))[0]
        self.a = self.a[idxs]
        self.b = self.b[:, idxs]
        self.iter = self.iter_idx[idxs]
        self.lmbd = self.lmbd[idxs]
        self.loss = self.loss[idxs]
        self.eta = self.eta[idxs]

        # Identify local convexity
        if self.convex:
            self.convex_min = get_convex_min(self.b, self.X_std, self.penalty, self.gamma, self.lmbd*(1-self.alpha), self.family, self.penalty_factor, self.a)
        else:
            self.convex_min = None

        self.fitted = True

    def _get_crit(self, crit):
        # Get fitted criterion
        loglike = self._loglike(self.b)
        aic = 2 * self.df_model - 2 * loglike
        if crit == 'aic':
            return aic
        elif crit == 'bic':
            bic = np.log(self.n) * self.df_model - 2 * loglike
            return bic
        elif crit == 'aicc':
            aicc = aic - 2 * self.p * (self.p + 1) / (self.n - self.p - 1)
            return aicc
        else:
            raise ValueError('Invalid criterion specified %s' % crit)

    def aic(self):
        return self._get_crit('aic')

    def bic(self):
        return self._get_crit('bic')

    def aicc(self):
        return self._get_crit('aicc')

    def predict(self, X: np.ndarray = None, ptype: str = 'link'):
        """
        Predict the response variable from a new set of predictors or return the coefficients of the fitted model.

        Parameters
        ----------
        X : np.ndarray,
            New set of predictors

        ptype : str, optional,
            Type of prediction. Options are 'response', 'coefficients', 'link', 'class', 'vars', 'nvars'.
            Default is 'link'.

        Returns
        -------
        np.ndarray,
            Predicted response variable or coefficients

        """
        # Predict the response
        if not self.fitted:
            raise ValueError('Model has not been fitted')

        if ptype not in ['response', 'coefficients', 'link', 'class', 'vars', 'nvars']:
            raise ValueError('Invalid Prediction type')

        X_std = stats.zscore(X, axis=1)

        beta = self._get_coef()

        if ptype == "coefficients":
            return beta

        if ptype == "nvars":
            return lambda x: np.sum(beta != 0)
        if ptype == "vars":
            return lambda x: np.where(beta != 0)

        self.eta = np.matmul(X_std, beta) + self.a

        if ptype == "link" or self.family == "gaussian":
            return np.squeeze(self.eta)
        elif self.family == "binomial":
            resp = np.exp(self.eta) / (1 + np.exp(self.eta))
        elif self.family == "poisson":
            resp = np.exp(self.eta)
        else:
            raise ValueError("Unknown family specified for response prediction")

        if ptype == "response":
            return np.squeeze(resp)
        if ptype == "class":
            if self.family == "binomial":
                return np.squeeze(1*self.eta>0)
            else:
                raise ValueError("Only binomial family is supported for class prediction")


    def fit_predict(self):
        # Fit the model and predict the response
        self.fit()
        return self.predict()
