import numpy as np
from scipy.special import gamma
import scipy.stats as stats


class Params(object):
    def __init__(self, u=0.0, k=1.0, a=2.0, b=1.0):
        self.u = float(u)
        self.k = float(k)
        self.a = float(a)
        self.b = float(b)

    def __str__(self, precision=3):
        return 'u={u:-{l}.{p}f} k={k:-{l}.{p}f} a={a:-{l}.{p}f} b={b:-{l}.{p}f}'.format(u=self.u,
                                                         k=self.k,
                                                         a=self.a,
                                                         b=self.b,
                                                         p=precision,
                                                         l=precision+4)


class PPosterior(object):
    def __init__(self, params=Params()):
        self.params = params
        self.pdf_gamma = stats.gamma(a=params.a, scale=1/params.b)

    def pdf_scalar(self, u, l):
        pdf_norm = stats.norm(loc=self.params.u,
                              scale=(self.params.k*l)**(-0.5))
        return pdf_norm.pdf(u) * self.pdf_gamma.pdf(l)

    def pdf_list(self, u, l):
        return [[self.pdf_scalar(us, ls) for us in u] for ls in l]

    def pdf(self, u, l):
        if hasattr(u, '__iter__'):
            return self.pdf_list(u, l)
        else:
            return self.pdf_scalar(u, l)


class QPosterior(PPosterior):
    def __init__(self, params=Params()):
        self.params = params
        self.pdf_norm = stats.norm(loc=params.u, scale=(params.k)**(-0.5))
        self.pdf_gamma = stats.gamma(a=params.a, scale=1/params.b)

    def pdf_scalar(self, u, l):
        return self.pdf_norm.pdf(u) * self.pdf_gamma.pdf(l)


class InferResults(object):
    def __init__(self, p, l):
        self.p = p
        self.c = l
        self.n = len(p)

    def __len__(self):
        return self.n

    def __str__(self, i=None):
        if i is None:
            return '\n'.join([self.__str__(j) for j in range(self.n)])
        else:
            return 'it={:-3d} {:s} c={:10.5f}'.format(i, self.p[i].__str__(),
                                                      self.c[i])

    def opt_index(self):
        return max(enumerate(self.c[1:]), key=lambda x: x[1])[0] + 1

    def opt_params(self):
        return self.p[self.opt_index()]

    def __iter__(self):
        return InferResultsIterator(self)

    def __getitem__(self, i):
        return self.p[i]


class InferResultsIterator(object):
    def __init__(self, r):
        self.r = r
        self.i = 0

    def next(self):
        if self.i == self.r.n:
            raise StopIteration
        else:
            p = self.r.p[self.i]
            self.i += 1
            return p


def costs(p):
    return 0.5 * np.log(1.0 / p.k) + np.log(gamma(p.a)) - p.a * np.log(p.b)


def infer_qposterior(x, pprior_params, init_params=Params(), maxit=100, eps=1e-10):
    xs = sum(x)
    xss = sum(x**2)
    xm = np.mean(x)
    N = len(x)

    p_list = [init_params]
    l_list = [costs(init_params)]
    uc = (pprior_params.k * pprior_params.u + N * xm) / (pprior_params.k + N)
    ac = pprior_params.a + (N + 1) / 2.0
    it = 0
    while it < maxit:
        p_prev = p_list[it]
        eu = p_prev.u
        eus = 1 / p_prev.k + p_prev.u**2
        p_next = Params(
            u=uc,
            k=(pprior_params.k + N) * p_prev.a / p_prev.b,
            a=ac,
            b=pprior_params.b + pprior_params.k * \
            (eus + pprior_params.u**2 - 2 * eu * pprior_params.u) + \
            0.5 * (xss + N * eus - 2 * eu * xs))
        p_list.append(p_next)
        l_list.append(costs(p_next))
        if abs(l_list[it + 1] - l_list[it]) < eps:
            break
        it += 1
    return InferResults(p_list, l_list)


def infer_pposterior(x):
    xm = np.mean(x)
    n = len(x)
    return Params(u=xm, k=n, a=0.5 * n, b=0.5 * np.sum((x - xm)**2))


def simulate(u=0.0, l=1.0, size=100):
    return np.random.normal(loc=u, scale=l**(-0.5), size=size)
