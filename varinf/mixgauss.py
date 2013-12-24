import numpy as np
from scipy.special import psi, gamma, gammaln, multigammaln
from copy import copy


class NormalWishartParams(object):
    def __init__(self, u, b, W, v):
        self.d = len(u)
        if W.shape != (self.d, self.d):
            raise TypeError('u and W must have the same dimension!')
        self.u = u
        self.b = b
        self.W = W
        self.v = v

    def __str__(self):
        return 'u={:s}\nb={:s}\nW={:s}\nv={:s}'.format(self.u.__str__(),
                                                       self.b.__str__(),
                                                       self.W.__str__(),
                                                       self.v.__str__())

def infer_q(X, p_pi, p_nw, q_pi0=None, q_nw0=None, maxit=100):
    N = X.shape[0]
    D = X.shape[1]
    K = len(p_pi)
    W_inv = np.linalg.inv(p_nw.W)

    q_z = np.empty((N, K))
    if q_pi0 is None:
        q_pi0 = copy(p_pi)
    if q_nw0 is None:
        q_nw0 = [copy(p_nw) for k in range(K)]
    q_pi = copy(q_pi0)
    q_nw = copy(q_nw0)
    for it in range(maxit):
        # E-step: update q(z)
        log_pi = psi(q_pi) - psi(np.sum(q_pi))
        log_lambda = np.empty(K)
        for k in range(K):
            p = sum(psi(0.5*(q_nw[k].v-np.arange(D))))
            log_lambda[k] = p + np.log(2) * D + np.log(np.linalg.det(q_nw[k].W))
        for n in range(N):
            for k in range(K):
                t = (np.transpose(X[n] - q_nw[k].u)).dot(q_nw[k].W).dot(X[n] - q_nw[k].u)
                q_z[n][k] = 0.5 * (-D / q_nw[k].b - q_nw[k].v * t)
            q_z[n] += log_pi + 0.5 * log_lambda
            c = np.max(q_z[n])
            q_z[n] = np.exp(q_z[n] - c - np.log(np.sum(np.exp(q_z[n] - c))))

        # M-step: update q(pi) and q(uk,lk)
        nk = np.sum(q_z, 0)
        XK = np.diag(nk).dot(np.transpose(q_z).dot(X)) / sum(nk)
        SK = []
        for k in range(K):
            SK.append(np.zeros((D, D)))
            for n in range(N):
                x = X[n] - XK[k]
                SK[k] += q_z[n][k] * np.outer(x, x)
            SK[k] /= nk[k]

        q_pi = p_pi + nk
        s = p_nw.b * nk / (p_nw.b + nk)
        for k in range(K):
            q_nw[k].b = p_nw.b + nk[k]
            q_nw[k].u = (p_nw.b * p_nw.u - nk[k] * XK[k]) / q_nw[k].b
            t = XK[k] - p_nw.u
            q_nw[k].W = W_inv + nk[k] * SK[k] + s[k] * np.outer(t, t)
            q_nw[k].W = np.linalg.inv(q_nw[k].W)
            q_nw[k].v = p_nw.v + nk[k]

        # Compute variational bound
        l = 0.0
        t = D*np.log(2*np.pi)
        s = 0.0
        for k in range(K):
            s += nk[k] * (log_lambda[k] - D/q_nw[k].b - q_nw[k].v*np.trace(SK[k].dot(q_nw[k].W)) \
                          -q_nw[k].v*(XK[k]-q_nw[k].u).dot(q_nw[k].W).dot(XK[k]-q_nw[k].u) - t)
        l += 0.5 * s
        l += np.sum(q_z.dot(log_pi))
        l += logC(p_pi) + (p_pi - 1).dot(log_pi)
        t = D*np.log(p_nw.b/(2*np.pi))
        s = 0.0
        for k in range(K):
            s += t + log_lambda[k] - D*p_nw.b/q_nw[k].b \
                    - p_nw.b*q_nw[k].v*(q_nw[k].u-p_nw.u).dot(q_nw[k].W).dot(q_nw[k].u-p_nw.u) \
                    - q_nw[k].v*np.trace(W_inv.dot(q_nw[k].W))
        l += 0.5*s + 0.5*(p_nw.v-D-1)*np.sum(log_lambda)
        l += np.sum(q_z*np.log(q_z))
        l += (q_pi - 1).dot(log_lambda) + logC(q_pi)
        for k in range(K):
            l += 0.5*log_lambda[k] + 0.5*D*np.log(q_nw[k].b/(2*np.pi)) - 0.5*D - wishart_H(q_nw[k].W, q_nw[k].v)

        print it, l

    return q_z, q_pi, q_nw


def logC(a):
    return gammaln(np.sum(a)) - np.sum(gammaln(a))

def logB(W, v):
    D = W.shape[0]
    return -0.5*D*v*np.log(2.0) - 0.5*v*np.log(np.linalg.det(W)) - multigammaln(0.5*v, D)

def wishart_Elog(W, v):
    D = W.shape[0]
    return np.sum(psi(0.5*(v-np.arange(D)))) + D*np.log(2) + np.log(np.linalg.det(W))

def wishart_H(W, v):
    D = W.shape[0]
    return -logB(W, v) - 0.5*(v-D-1)*wishart_Elog(W, v) + 0.5*v*D



def prior_uninformative(D=1, K=1):
    p_pi = np.ones(K) * 1.0 / K
    p_nw = NormalWishartParams(u=np.random.normal(0.0, 1.0, D),
                                b=1e-3+np.random.gamma(0.001, 1.0/0.001),
                                W=np.diag(np.random.normal(1.0, 0.01, D)),
                                v=(D+1)+np.random.gamma(0.001, 0.001))
    return (p_pi, p_nw)

