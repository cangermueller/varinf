import unittest
from scipy.special import multigammaln, psi
import numpy as np

import mixgauss


class MixGaussTest(unittest.TestCase):
    def test_logC(self):
        for a in range(1, 20, 2):
            self.assertEqual(mixgauss.logC(a), 0.0)
        self.assertAlmostEqual(mixgauss.logC([1, 2, 3]), 4.0943445622221004)

    def test_logB(self):
        W = np.diag(np.array([0.5, 1.0, 1.5]))
        D = 3
        n = 5
        B = np.log(1.0 / (2.0**(0.5*D*n)*np.linalg.det(W)**(0.5*n)*np.exp(multigammaln(0.5*n, D))))
        self.assertAlmostEqual(mixgauss.logB(W, n), B)

    def test_wishart_Elog(self):
        W = np.diag(np.array([1, 2]))
        n = 10
        Elog = psi(5.0)+psi(4.5) + 2*np.log(2) + np.log(2)
        self.assertAlmostEqual(mixgauss.wishart_Elog(W, n), Elog)

    def test_wishart_H(self):
        W = np.diag(np.array([1, 2]))
        n = 5
        H = -mixgauss.logB(W, 5) - mixgauss.wishart_Elog(W, 5) + 5
        self.assertAlmostEqual(mixgauss.wishart_H(W, n), H)





if __name__ == '__main__':
    unittest.main()


