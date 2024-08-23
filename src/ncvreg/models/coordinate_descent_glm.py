import warnings
import scipy as sp
import numpy as np
from src.ncvreg.utils import mcp_loss, scad_loss, lasso_loss

def cd_ols(X, y, family, penalty, lmbd, eps, max_iter, gamma, multiplier, alpha, dfmax):
    n, p = X.shape
    L = np.zeros(lmbd)
    total_iter = 0
    beta0 = np.zeros(L)
    beta = np.zeros((L, p))
    dev = np.zeros(L)
    Eta = np.zeros((L, n))
    eta, r, w, s = np.zeros(n)
    z, a, e1, e2 = np.zeros(p)
    a0 = 0

    # Initialization
    ybar = sum(y, n) / n
    nullDev = 0

    if family == "binomial":
        a0, beta0[0] = np.log(ybar / (1 - ybar))
        for i in range(n):
            nullDev -= 2*y[i]*np.log(ybar) + 2*(1 - y[i])*np.log(1 - ybar)
    elif family == "poisson":
        a0, beta0[0] = np.log(ybar)
        for i in range(n):
            if y[i] > 0:
                nullDev += 2*(y[i]*np.log(y[i]/ybar) + ybar - y[i])
            else:
                nullDev += 2*ybar

    for i in range(n):
        s[i] = y[i] - ybar
        eta[i] = a0
    for j in range(p):
        z[j] = np.cross(X, s, n, j) / n

    # Coordinate descent path
    for l in range(L):
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
            a0 = beta0[l - 1]
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

        # While loops
        while total_iter < max_iter:
            while total_iter < max_iter:
                while total_iter < max_iter:
                    total_iter += 1

                    # Approximate L
                    dev[l] = 0
                    if family == "binomial":
                        v = 0.25
                        for i in range(n):
                            mu = sp.stats.binom.pmf(eta[i])
                            w[i] = max(mu * (1 - mu), 1e-4)
                            s[i] = y[i] - mu
                            r[i] = s[i] / w[i]
                            if y[i] == 1:
                                dev[l] = dev[l] - np.log(mu)
                            elif y[i] == 0:
                                dev[l] = dev[l] - np.log(1 - mu)
                            else:
                                dev[l] = dev[l] - y[i] * np.log(mu) - (1 - y[i]) * np.log(1 - mu)
                    elif family == "poisson":
                        for i in range(n):
                            mu = np.exp(eta[i])
                            w[i] = mu
                            s[i] = y[i] - mu
                            r[i] = s[i] / w[i]
                            if y[i] != 0:
                                dev[l] += y[i]*np.log(y[i]/mu)
                    else:
                        raise ValueError("Family must be one of 'binomial' or 'poisson'")

                    # Check for saturation
                    if dev[l] / nullDev < 0.01:
                        warnings.warn("Model saturated; exiting early")
                        break

                    # Update intercept
                    xwr = np.cross(w, r, n, 0)
                    xwx = np.sum(w, n)
                    beta0[l] = a0 + xwr / xwx
                    for i in range(n):
                        si = beta0[l] - a0
                        r[i] -= si
                        eta[i] += si
                    max_change = np.abs(si)*xwx/n

                    # Covariate loop
                    for j in range(p):
                        if e1[j]:
                            # Calculate u and v
                            xwr = np.wcross(X, r, w, n, j)
                            xwx = np.weighted_sum(X, w, n, j)
                            u = (xwr / n) + (xwx / n)*a[j]
                            v = xwx / n

                            # Update beta_j
                            l1 = lmbd[l] * alpha * multiplier[j]
                            l2 = lmbd[l] * (1 - alpha) * multiplier[j]
                            if penalty == "lasso":
                                beta[l*p+j] = lasso_loss(u, l1, l2, v)
                            elif penalty == "mcp":
                                beta[l*p+j] = mcp_loss(u, l1, l2, gamma, v)
                            elif penalty == "scad":
                                beta[l*p+j] = scad_loss(u, l1, l2, gamma, v)
                            else:
                                raise ValueError("Penalty must be one of 'lasso', 'mcp', or 'scad'")

                            # Update r
                            shift = beta[l*p+j] - a[j]
                            if shift != 0:
                                for i in range(n):
                                    si = shift*X[j*n+i]
                                    r[i] -= si
                                    eta[i] += si
                                if np.abs(shift) * np.sqrt(v) > max_change:
                                    max_change = np.abs(shift) * np.sqrt(v)

                    # Check for convergence
                    a0 = beta0[l]
                    for j in range(p):
                        a[j] = beta[l*p+j]
                    if max_change < eps:
                        break

                # Scan for violations in the strong set
                violations = 0
                for j in range(p):
                    if e1[j] == 0 and e2[j] == 1:
                        z[j] = np.cross(X, s, n, j) / n
                        l1 = lmbd[l] * alpha * multiplier[j]
                        if np.abs(z[j]) > l1:
                            e1[j], e2[j] = 1
                            violations += 1
                if violations == 0:
                    break

            # Scan for violations in the rest
            violations = 0
            for j in range(p):
                if e1[j] == 0 and e2[j] == 1:
                    z[j] = np.cross(X, s, n, j) / n
                    l1 = lmbd[l] * alpha * multiplier[j]
                    if np.abs(z[j]) > l1:
                        e1[j], e2[j] = 1
                        violations += 1
            if violations == 0:
                for i in range(n):
                    Eta[n*l+i] = eta[i]
                break
    res = {"beta0": beta0, "beta": beta, "dev": dev, "Eta": Eta, "iter": total_iter}
    return res
