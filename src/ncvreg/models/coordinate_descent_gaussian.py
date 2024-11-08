import warnings
from itertools import repeat
from os import lstat

import scipy as sp
import numpy as np
from src.ncvreg.utils import mcp_loss, scad_loss, lasso_loss, gauss_loss

def cd_gaussian(X, y, penalty, lmbd, eps, max_iter, gamma, multiplier, alpha, dfmax):
    n, p = X.shape
    L = len(lmbd)
    total_iter = 0
    beta = np.zeros((L, p))
    loss = np.zeros(L)
    Eta = np.zeros((L, n))
    iter_idx = np.zeros(L)
    r = y
    a = np.zeros(p)
    z = np.zeros(p)
    e1 = np.zeros(p)
    e2 = np.zeros(p)

    for i in range(p):
        z[i] = np.dot(X[:, i], r) / n

    lstart = 0

    # Initialize residual sum of squares
    rss = gauss_loss(r, n)
    sdy = np.sqrt(rss/n)

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
                a[j] = beta[l-1, j]

            for j in range(p):
                if a[j] != 0:
                    nv += 1

            if (nv > dfmax) or total_iter == max_iter:
                for ll in range(l, L):
                    iter_idx[ll] = np.nan
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
                    iter_idx[l] += 1
                    total_iter += 1
                    max_change = 0
                    for j in range(p):
                        if e1[j]:
                            z[j] = np.dot(X[:, j], r) / n + a[j]

                            # Update beta
                            l1 = lmbd[l] * alpha * multiplier[j]
                            l2 = lmbd[l] * (1 - alpha) * multiplier[j]
                            if penalty == "mcp":
                                beta[l, j] = mcp_loss(z[j], l1, l2, gamma, 1)
                            elif penalty == "scad":
                                beta[l, j] = scad_loss(z[j], l1, l2, gamma, 1)
                            elif penalty == "lasso":
                                beta[l, j] = lasso_loss(z[j], l1, l2, 1)
                            else:
                                raise ValueError("Penalty must be one of 'lasso', 'mcp', or 'scad'")

                            # Update r
                            shift = beta[l, j] - a[j]
                            if shift != 0:
                                for i in range(n):
                                    r[i] -= shift * X[i, j]
                                if np.abs(shift) > max_change:
                                    max_change = np.abs(shift)

                    for j in range(p):
                        a[j] = beta[l, j]
                        if max_change < eps*sdy:
                            break

                # Scan for violations in the strong set
                violations = 0
                for j in range(p):
                    if e1[j] == 0 and e2[j] == 1:
                        z[j] = np.dot(X[:, j], r) / n
                        l1 = lmbd[l] * alpha * multiplier[j]
                        l2 = lmbd[l] * (1 - alpha) * multiplier[j]
                        if penalty == "mcp":
                            beta[l, j] = mcp_loss(z[j], l1, l2, gamma, 1)
                        elif penalty == "scad":
                            beta[l, j] = scad_loss(z[j], l1, l2, gamma, 1)
                        elif penalty == "lasso":
                            beta[l, j] = lasso_loss(z[j], l1, l2, 1)
                        else:
                            raise ValueError("Penalty must be one of 'lasso', 'mcp', or 'scad'")

                        if beta[l, j] != 0:
                            e1[j] = e2[j] = 1
                            for i in range(n):
                                r[i] -= beta[l, j] * X[i, j]
                            a[j] = beta[l, j]
                            violations += 1
                if violations == 0:
                    break

            # Scan for violations in the rest
            violations = 0
            for j in range(p):
                if e2[j] == 0:
                    z[j] = np.dot(X[:, j], r) / n

                    # Update beta_j
                    l1 = lmbd[l] * alpha * multiplier[j]
                    l2 = lmbd[l] * (1 - alpha) * multiplier[j]
                    if penalty == "mcp":
                        beta[l, j] = mcp_loss(z[j], l1, l2, gamma, 1)
                    elif penalty == "scad":
                        beta[l, j] = scad_loss(z[j], l1, l2, gamma, 1)
                    elif penalty == "lasso":
                        beta[l, j] = lasso_loss(z[j], l1, l2, 1)
                    else:
                        raise ValueError("Penalty must be one of 'lasso', 'mcp', or 'scad'")

                    if beta[l, j] != 0:
                        e1[j] = e2[j] = 1
                        for i in range(n):
                            r[i] -= beta[l, j] * X[i, j]
                        a[j] = beta[l, j]
                        violations += 1

            if violations == 0:
                break

        # Update loss
        loss[l] = gauss_loss(r, n)
        for i in range(n):
            Eta[l, i] = y[i] - r[i]

    res = {"a": a,
           "r": r,
           "e1": e1,
           "e2": e2,
           "z": z,
           "beta": beta,
           "loss": loss,
           "Eta": Eta,
           "iter_idx": iter_idx}
    return res