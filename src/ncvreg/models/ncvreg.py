import warnings
from typing import List
import numpy as np
from scipy import stats
import statsmodels.api as sm
from itertools import repeat
from src.ncvreg.utils import maxprod, get_convex_min
from src.ncvreg.models.coordinate_descent_glm import cd_ols
from src.ncvreg.models.coordinate_descent_gaussian import cd_gaussian
from src.ncvreg.base import BaseRegressor


class NCVREG(BaseRegressor):
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 lmbd: List = None,
                 family: str = 'gaussian',
                 penalty: str = 'mcp',
                 gamma: float = 3,
                 alpha: float = 1,
                 nlambda: int = 100,
                 eps: float = 1e-4,
                 max_iter: int = 1000,
                 convex: bool = True):
        """
        Fit a Regularized regression model using MCP or SCAD Penalty

        Parameters
        ----------
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
        """
        super().__init__()
        self.X = X
        self.y = y
        self.lmbd = lmbd
        self.family = family
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
        self.penalty_factor = np.repeat(1, self.p)

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
        if self.family == 'binomial' and len(np.unique(self.y)) > 2:
            raise ValueError('y must be binary for binomial family')
        # Check that the only values for y in a binomial family are 0 and 1
        if self.family == 'binomial' and not all(np.isin(self.y, [0, 1])):
            raise ValueError('y must be binary for binomial family')
        if len(self.y) != self.n:
            raise ValueError('Dimensions do not match. y must be of length n')
        if np.isnan(self.y).any() or np.isnan(self.X).any():
            raise ValueError('X and y must not contain NaNs')

    def _standardize(self):
        """
        Standardize the data matrix by column
        :return:
        """
        self.X_std = stats.zscore(self.X, axis=1)

        if self.family == 'gaussian':
            self.y_std = self.y - np.mean(self.y)
        else:
            self.y_std = self.y

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
            fit = sm.GLM(self.y, self.X[:, idx], family=fam).fit()
        else:
            fit = sm.GLM(self.y, np.ones(self.n), family=fam).fit()

        # Find Z-max
        if self.family == 'gaussian':
            zmax = maxprod(self.X, fit.resid_deviance, idx, self.penalty_factor) / self.n
        else:
            zmax = maxprod(self.X, fit.resid_working * fit.weights, idx, self.penalty_factor) / self.n

        lambda_max = zmax / self.alpha

        if self.lambda_min == 0:
            self.lmbd = np.exp(np.linspace(np.log(lambda_max), np.log(0.001 * lambda_max), self.nlambda - 1)).append(0)
        else:
            self.lmbd = np.exp(np.linspace(np.log(lambda_max), np.log(self.lambda_min * lambda_max), self.nlambda))

        if len(idx) != self.p:
            self.lmbd = self.lmbd * 1.000001

    def _fit_gaussian(self):
        # Initialize
        res = cd_gaussian(self.X, self.y, self.penalty, self.lmbd, self.eps,
                          self.max_iter, self.gamma, self.penalty_factor, self.alpha,
                          self.dfmax)
        return res

    def _fit_glm(self):
        # Initialize
        res = cd_ols(self.X, self.y, self.family, self.penalty, self.lmbd,
                     self.eps, self.max_iter, self.gamma, self.penalty_factor,
                     self.alpha, self.dfmax)
        return res

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

        # Get lambda
        if self.lmbd is None:
            self._get_lambda()
        else:
            self.nlambda = len(self.lmbd)
            if self.nlambda == 1:
                warnings.warn('Only one lambda value provided. Will use this value for fitting')
            else:
                self.lmbd = -np.sort(-self.lmbd)

        # Fit the model
        if self.family == 'gaussian':
            self.model = self._fit_gaussian()
            self.a = np.repeat(np.mean(self.y), self.nlambda)
            self.b = self.model['beta']
            self.loss = self.model['loss']
            self.eta = np.array(self.model['Eta']).transpose() + np.mean(self.y)
            self.iter_idx = self.model['iter_idx']
        else:
            self.model = self._fit_glm()
            self.a = self.model['alpha']
            self.b = self.model['beta']
            self.loss = self.model['loss']
            self.eta = self.model['Eta']
            self.iter_idx = self.model['iter_idx']

        # Eliminate saturated lambda values
        idxs = np.where(~np.isnan(self.iter_idx))[0]
        self.a = self.a[idxs]
        self.b = self.b[idxs]
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

    def predict(self):
        # Predict the response
        if self.family == 'gaussian':
            pass
        else:
            pass

    def fit_predict(self):
        # Fit the model and predict the response
        self.fit()
        return self.predict()
