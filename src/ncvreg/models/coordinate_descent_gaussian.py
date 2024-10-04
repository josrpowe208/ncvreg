import warnings
from itertools import repeat
from os import lstat

import scipy as sp
import numpy as np
from src.ncvreg.utils import mcp_loss, scad_loss, lasso_loss, weighted_sum, w_cross

def cd_gaussian(X, y, penalty, lmbd, eps, max_iter, gamma, multiplier, alpha, dfmax):
    n, p = X.shape
    L = len(lmbd)
    total_iter = 0
    beta = np.zeros((L, p))
    loss = np.zeros(L)
    Eta = np.zeros((L, n))
    r = y
    a, e1, e2 = np.zeros(p)
    z = np.cross(X, r, n, p)/n

    lstart = 0

    # Coordinate Descent Path
    for l in range(lstart, L):
        if l == 0:
            lmax = 0
            for j in range(p):
                if np.abs(z[j]) > lmax:
                    lmax = np.abs(z[j])
            # Define penalty cutoff
            if penalty == "lasso":
                cutoff = 2*lmbd[l] - lmax
            elif penalty == "mcp":
                cutoff = lmbd[l] + gamma / (gamma - 1)*(lmbd[l] - lmax)
            elif penalty == "scad":
                cutoff = lmbd[l] + gamma / (gamma - 2) * (lmbd[l] - lmax)
            else:
                raise ValueError("Penalty must be one of 'lasso', 'mcp', or 'scad'")

            for j in range(p):
                if np.abs(z[j]) > (cutoff * alpha * multiplier[j]):
                    e2[j] = 1
        else:
            nv = 0
            for j in range(p):
                a[j] = beta[(l - 1)*p+j]

            for j in range(p):
                if a[j] != 0:
                    nv += 1

            if (nv > dfmax) or total_iter == max_iter:
                break

            if penalty == "lasso":
                cutoff = 2*lmbd[l] - lmbd[l-1]
            elif penalty == "mcp":
                cutoff = lmbd[l] + gamma / (gamma - 1)*(lmbd[l] - lmbd[l-1])
            elif penalty == "scad":
                cutoff = lmbd[l] + gamma / (gamma - 2) * (lmbd[l] - lmbd[l - 1])
            else:
                raise ValueError("Penalty must be one of 'lasso', 'mcp', or 'scad'")

            for j in range(p):
                if np.abs(z[j]) > (cutoff * alpha * multiplier[j]):
                    e2[j] = 1

        # While loop
        while total_iter < max_iter:
            while total_iter < max_iter:
                while total_iter < max_iter:
                    total_iter += 1
                    max_change = 0
                    for j in range(p):
                        if e1[j]:
                            z[j] = np.cross(X, r, n, j) / n + a[j]

                            # Update beta
                            l1 = lmbd[l] * alpha * multiplier[j]
                            l2 = lmbd[l] * (1 - alpha) * multiplier[j]
                            if penalty == "mcp":
                                beta[l*p+j] = mcp_loss(z[j], l1, l2, gamma, 1)
                            elif penalty == "scad":
                                beta[l*p+j] = scad_loss(z[j], l1, l2, gamma, 1)
                            elif penalty == "lasso":
                                beta[l*p+j] = lasso_loss(z[j], l1, l2, 1)
                            else:
                                raise ValueError("Penalty must be one of 'lasso', 'mcp', or 'scad'")

                            # Update r
                            shift = beta[l*p+j] - a[j]
                            if shift != 0:
                                for i in range(n):
                                    r[i] -= shift * X[j*n+i]
                                if np.abs(shift) > max_change:
                                    max_change = np.abs(shift)

                    for j in range(p):
                        a[j] = beta[l*p+j]
                        if max_change < eps*sdy:
                            break

                # Scan for violations in the strong set
                violations = 0
                for j in range(p):
                    if e1[j] == 0 and e2[j] == 1:
                        z[j] = np.cross(X, r, n, j) / n
                        l1 = lmbd[l] * alpha * multiplier[j]
                        l2 = lmbd[l] * (1 - alpha) * multiplier[j]
                        if penalty == "mcp":
                            beta[l * p + j] = mcp_loss(z[j], l1, l2, gamma, 1)
                        elif penalty == "scad":
                            beta[l * p + j] = scad_loss(z[j], l1, l2, gamma, 1)
                        elif penalty == "lasso":
                            beta[l * p + j] = lasso_loss(z[j], l1, l2, 1)
                        else:
                            raise ValueError("Penalty must be one of 'lasso', 'mcp', or 'scad'")

                        if beta[l * p + j] != 0:
                            e1[j] = e2[j] = 1
                            for i in range(n):
                                r[i] -= beta[l * p + j] * X[j * n + i]
                            a[j] = beta[l * p + j]
                            violations += 1
                if violations == 0:
                    break

            # Scan for violations in the rest
            violations = 0