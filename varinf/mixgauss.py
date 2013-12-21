import numpy as np
from scipy.special import psi


class NormalWishartParams(object):
    def __init__(self, u, b, W, v):
        self.d = len(u)
        if W.shape != (self.d, self.d):
            raise TypeError('u and W must have the same dimension!')
        self.u = u
        self.b = b
        self.W = W
        self.v = v

def infer_q(X, p_pi, p_nw, q_z, q_pi, q_nw):
    N = X.shape[0]
    D = X.shape[1]
    K = len(p_pi)
    W_inv = np.linalg.inv(p_nw.W)

    for it in range(maxit):
        # E-step: update q(z)
        log_pi = psi(q_pi) - psi(np.sum(q_pi))
        log_lambda = np.empty(K)
        for k in range(K):
            p = sum(psi(0.5 * (q_nw[k].v - np.arange(D) + 1)))
            log_lambda[k] = p + np.log(2) * D + np.log(np.linalg.det(q_nw[k].W))
        for n in range(N):
            for k in range(K):
                t = (np.transpose(X[n] - q_nw[k].u)).dot(q_nw[k].W).dot(X[n] - q_nw[k].u)
                q_z[n][k] = 0.5 * (-D / q_nw[k].b - q_nw[k].v * t)
            q_z[n] += log_pi + 0.5 * log_lambda
            c = np.max(q_z[n])
            q_z[n] = np.exp(q_z[n] - c - np.log(np.sum(exp(np.z[n] - c))))

        # M-step: update q(pi) and q(uk,lk)
        nk = np.sum(q_z, 0)
        XK = np.diag(nk).dot.np.transpose(q_z).dot(X)
        SK = list()
        for k in range(K):
            SK[k] = np.zeros((D, D))
            for n in range(N):
                x = X[n] - XK[k]
                SK[k] += q_z[n][k] * np.outer(x, x)
            SK[k] /= nk[k]

        q_pi = p_pi + nk
        s = p_nw.b * nk / (p_nw.b + bk)
        for k in range(K):
            q_nw[k].b = p_nw.b + nk[k]
            q_nw[k].u = (p_nw.b * p_nw.u - nk[k] * Xk[k]) / q_nw[k].b
            t = XK[k] - p_nw.u
            q_nw[k].W = W_inv + nk[k] * SK[k] + t[k] * np.outer(t, t)
            q_nw[k].W = np.linalg.inv(q_nw[k].W)
            q_nw[k].v = p_nw.v + nk[k]

