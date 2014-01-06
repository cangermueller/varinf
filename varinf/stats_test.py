import unittest
import numpy as np
import scipy.stats
from scipy.special import multigammaln
import mixgauss


class TestWishart(unittest.TestCase):
    def test_range(self):
        np.random.seed(0)
        W = np.random.rand(2, 2) + np.eye(2)*2
        w = mixgauss.Wishart(W, 3)
        self.assertEqual(w.d, 2)
        # self.assertEqual(w.W, W)
        self.assertEqual(w.v, 3)
        p_max = w.pdf(w.v * w.W)
        for i in range(10):
            p = w.pdf(np.random.rand(2, 2) + np.eye(2)*2)
            self.assertGreaterEqual(p, 0.0)
            # self.assertLessEqual(p, p_max)

    def test_gamma(self):
        a = 2.0
        b = 1.0
        g = scipy.stats.gamma(a, scale=1.0/b)
        w = mixgauss.Wishart(np.array([[b]]), a)
        for x in np.linspace(0.1, 5):
            print g.pdf(x)
            print w.pdf(np.array([[x]]))
            self.assertAlmostEqual(g.pdf(x), w.pdf(np.array([[x]])))


if __name__ == '__main__':
    unittest.main()
