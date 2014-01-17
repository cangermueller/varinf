""" Bayesian Variation Inference for a Mixture of Gaussians."""

import numpy as np
from scipy.special import psi, gammaln, multigammaln
import copy


class NormalWishartParams(object):

    """ Parameters of Normal Wishart Distribution. """

    def __init__(self, u, b, W, v):
        self.d = len(u)
        if W.shape != (self.d, self.d):
            raise TypeError('u and W must have the same dimension!')
        self.u = u
        self.b = b
        self.W = W
        self.v = v

    def __str__(self, precision=3):
        p = 'u={:s}\nb={:.{p}f}\nW={:s}\nv={:.{p}f}'.format(np.array_str(self.u, precision=precision),
                                                            self.b,
                                                            np.array_str(self.W, precision=precision),
                                                            self.v,
                                                            p=precision)
        e = 'E[u]={:s}\nE[l]={:s}\nE[l\']={:s}'.format(np.array_str(self.u, precision=precision),
                                                       np.array_str(self.W*self.v, precision=precision),
                                                       np.array_str(np.linalg.inv(self.W*self.v), precision=precision))
        return p + '\n\n' + e

    def copy(self):
        return copy.deepcopy(self)


class QParams(object):
    """ Parameters of approximate distribution Q. """

    def __init__(self, z, pi, nw):
        self.z = z
        self.pi = pi
        self.nw = nw

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self, z=False, precision=3):
        s = ''
        if z:
            s += '# q(z):\n{:s}\n'.format(np.array_str(self.z, precision=precision))
            s += '# q(pi):\n{:s}'.format(np.array_str(self.pi, precision=precision))
            for i, nw in enumerate(self.nw):
                s += '\n\n# q(nw[{:d}]):\n{:s}'.format(i, nw.__str__(precision=precision))
        return s


def prior_uninformative(D=1, K=1):
    p_pi = np.ones(K)
    p_nw = NormalWishartParams(u=np.ones(D), b=1.0, W=np.eye(D), v=D)
    return (p_pi, p_nw)


def q0_random(N, D, K):
    nw = []
    for k in range(K):
        u = np.random.multivariate_normal(np.zeros(D), np.eye(D))
        nw.append(NormalWishartParams(u=u, b=1.0, W=np.eye(D), v=D))
    return QParams(z=np.empty((N, K)),
                   pi=np.ones(K) / float(K),
                   nw=nw)

def infer_q(X, p_pi, p_nw, q0, maxit=100, verbose=True):
    N = X.shape[0]
    D = X.shape[1]
    K = len(p_pi)
    W_inv = np.linalg.inv(p_nw.W)
    q = q0.copy()
    q_it = [q0.copy()]
    j_it = [-np.inf]

    if verbose:
        print '%4s %10s' % ('It', 'J')
    for it in range(maxit):
        # E-step: update q(z)
        log_pi = psi(q.pi) - psi(np.sum(q.pi))
        log_lambda = np.empty(K)
        for k in range(K):
            p = sum(psi(0.5*(q.nw[k].v-np.arange(D))))
            log_lambda[k] = p + np.log(2) * D + np.log(np.linalg.det(q.nw[k].W))
        for n in range(N):
            for k in range(K):
                t = (np.transpose(X[n] - q.nw[k].u)).dot(q.nw[k].W).dot(X[n] - q.nw[k].u)
                q.z[n][k] = 0.5 * (-D / q.nw[k].b - q.nw[k].v * t)
            q.z[n] += log_pi + 0.5 * log_lambda
            c = np.max(q.z[n])
            q.z[n] = np.exp(q.z[n] - c - np.log(np.sum(np.exp(q.z[n] - c))))


        # M-step: update q(pi) and q(uk,lk)
        nk = np.sum(q.z, 0) + 1e-10
        XK = np.diag(1/nk).dot(np.transpose(q.z).dot(X))
        SK = []
        for k in range(K):
            SK.append(np.zeros((D, D)))
            for n in range(N):
                x = X[n] - XK[k]
                SK[k] += q.z[n][k] * np.outer(x, x)
            SK[k] /= nk[k]

        q.pi = p_pi + nk
        s = p_nw.b * nk / (p_nw.b + nk)
        for k in range(K):
            q.nw[k].b = p_nw.b + nk[k]
            q.nw[k].u = (p_nw.b * p_nw.u + nk[k] * XK[k]) / q.nw[k].b
            t = XK[k] - p_nw.u
            q.nw[k].W = W_inv + nk[k] * SK[k] + s[k] * np.outer(t, t)
            q.nw[k].W = np.linalg.inv(q.nw[k].W)
            q.nw[k].v = p_nw.v + nk[k]

        q_it.append(q.copy())

        # Variational bound
        # E[ln p(X|Z,u,L)]
        j = 0.0
        t = D*np.log(2*np.pi)
        s = 0.0
        for k in range(K):
            s += nk[k] * (log_lambda[k] - D/q.nw[k].b - q.nw[k].v*np.trace(SK[k].dot(q.nw[k].W)) \
                          -q.nw[k].v*(XK[k]-q.nw[k].u).dot(q.nw[k].W).dot(XK[k]-q.nw[k].u) - t)
        j += 0.5 * s
        # E[ln p(Z|pi)]
        j += log_pi.dot(nk)
        # E[ln p(pi)]
        j += logC(p_pi) + (p_pi - 1).dot(log_pi)
        # E[ln p(u, L)]
        t = D*np.log(p_nw.b/(2*np.pi))
        s = 0.0
        for k in range(K):
            s += t + log_lambda[k] - D*p_nw.b/q.nw[k].b \
                    - p_nw.b*q.nw[k].v*(q.nw[k].u-p_nw.u).dot(q.nw[k].W).dot(q.nw[k].u-p_nw.u) \
                    - q.nw[k].v*np.trace(W_inv.dot(q.nw[k].W))
        j += 0.5*s + 0.5*(p_nw.v-D-1)*np.sum(log_lambda) + K*logB(p_nw.W, p_nw.v)
        # E[ln q(Z)]
        j -= np.sum(q.z*np.log(q.z + 1e-10))
        # E[ln q(pi)]
        j -= (q.pi - 1).dot(log_lambda) + logC(q.pi)
        # E[ln q(u, L)]
        for k in range(K):
            j -= 0.5*log_lambda[k] + 0.5*D*np.log(q.nw[k].b/(2*np.pi)) - 0.5*D - wishart_H(q.nw[k].W, q.nw[k].v)

        j_it.append(j)

        if verbose:
            print '{:4d} {:10.3f}'.format(it + 1, j)

    return (q_it, j_it)


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
