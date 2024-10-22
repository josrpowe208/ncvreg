import numpy as np

def maxprod(X, residuals, idx, penalty_factor):
    """
    Compute the maximum product of the residuals and the design matrix

    :param X: np.ndarray
        Design matrix
    :param residuals: np.ndarray
        Residuals
    :param idx: np.ndarray
        Index of non-zero penalty factors
    :param penalty_factor: np.ndarray
        Penalty factors
    :return: float
        Maximum product
    """
    n, p = X.shape
    zmax = 0
    for i in range(p):
        zz = np.cross(X, residuals, n, idx[i]-1) / penalty_factor[idx[i]-1]
        if np.abs(zz) > zmax:
            zmax = np.abs(zz)
    return zmax


def get_convex_min(b, X, penalty, gamma, l2, family, pf, a, Delta):
    """
    Identify local convexity
    :param b:
    :param X:
    :param penalty:
    :param gamma:
    :param l2:
    :param family:
    :param pf:
    :param a:
    :param Delta:
    :return:
    """
    n, p = X.shape
    l = len(b)

    if penalty == "mcp":
        k = 1/gamma
    elif penalty == "scad":
        k = 1/(gamma - 1)
    elif penalty == "l1":
        return None
    else:
        raise ValueError("Penalty must be one of 'mcp', 'scad', or 'l1'")

    if l == 0:
        return None

    for i in range(l):
        if i == 0:
            A_1 = np.ones(p)
        else:
            A_1 = np.abs(b[:, i-1])

        if i == l-1:
            L_2 = l2[i]
            U = A_1
        else:
            A_2 = np.abs(b[:, i+1])
            U = np.minimum(A_1, A_2)
            L_2 = l2[i+1]

        if not sum(U) == 0:
            continue

        Xu = X[:, -U]
        p_hat = k*(pf[-U] - L_2*pf[-U])

        if family == "gaussian":
            if any(A_1 == A_2):
                eigen_min = min(np.eigvals(np.cross(Xu)/n - np.diag(p_hat)))
        elif family == "binomial":
            if (i == l):
                eta = a[i] + np.matmul(X, b[:, i])
            else:
                eta = a[i + 1] + np.matmul(X, b[:, i + 1])
            pi = np.exp(eta) / (1 + np.exp(eta))
            w = pi * (1 - pi)
            w[eta > np.log(0.9999/0.0001)] = 0.0001
            w[eta < np.log(0.0001/0.9999)] = 0.0001
            Xu = np.sqrt(w) * Xu
            XwXn = np.cross(Xu)/n
            eigen_min = min(np.eigvals(XwXn - np.diag(np.array(1, np.diag(XwXn)*p_hat))))
        elif family == "poisson":
            if (i == l):
                eta = a[i] + np.matmul(X, b[:, i])
            else:
                eta = a[i + 1] + np.matmul(X, b[:, i + 1])
            mu = np.exp(eta)
            Xu = np.sqrt(mu) * Xu
            XwXn = np.cross(Xu) / n
            eigen_min = min(np.eigvals(XwXn - np.diag(np.array(1, np.diag(XwXn) * p_hat))))
        elif family == "cox":
            if (i == l):
                eta = np.matmul(X, b[:, i])
            else:
                eta = np.matmul(X, b[:, i + 1])
            haz = drop(np.exp(eta))
            rsk = reverse(np.cumsum(reverse(haz)))
            h = haz*np.cumsum(Delta/rsk)
            XwXn = np.cross(np.sqrt(h) * Xu)/n
            eigen_min = min(np.eigvals(XwXn - np.diag(np.diag(XwXn)*p_hat, XwXn.shape[0], XwXn.shape[1])))
        else:
            raise ValueError("Family must be one of 'gaussian', 'binomial', 'poisson', or 'cox'")

        if eigen_min < 0:
            return i

    return None

def mcp_loss(z, l1, l2, gamma, v):
    if z > 0:
        s = 1
    elif z < 0:
        s = -1
    else:
        s = 0

    if np.abs(z) <= l1:
        return 0
    elif np.abs(z) <= (gamma * l1 * (1 + l2)):
        return (s * (np.abs(z) - l1)/(v * (1 + l2 - 1/gamma)))
    else:
        return (z / (v * (1 + l2)))


def scad_loss(z, l1, l2, gamma, v):
    if z > 0:
        s = 1
    elif z < 0:
        s = -1
    else:
        s = 0

    if np.abs(z) <= l1:
        return 0
    elif np.abs(z) <= (l1 * (1 + l2) + l1):
        return (s * (np.abs(z) - l1) / (v * (1 + l2)))
    elif np.abs(z) <= (gamma * l1 * (1 + l2)):
        return (s * (np.abs(z) - gamma * l1 / (gamma - 1)) / (v * (1 - 1 / (gamma-1) + l2)))
    else:
        return (z / (v * (1 + l2)))

def lasso_loss(z, l1, l2, v):
    if z > 0:
        s = 1
    elif z < 0:
        s = -1
    else:
        s = 0

    if np.abs(z) <= l1:
        return 0
    else:
        return (s * (np.abs(z) - l1) / (v * (1 + l2)))

def w_cross(X, y, w, n, j):
    nn = n*j
    val = 0
    for i in range(n):
        val += X[nn+i]*y[i]*w[i]
    return val

def weighted_sum(X, w, n, j):
    nn = n*j
    val = 0
    for i in range(n):
        val += np.power(X[nn+i], 2)*w[i]
    return val

def gauss_loss(r, n):
    l = 0
    for i in range(n):
        l += np.power(r[i], 2)
    return l

