import unittest

from embedding_uncertainty import polyagamma_sampler
import numpy as np

class PolyaGammaTest(unittest.TestCase):

    def test_polya_gamma(self):
        """
        Test MAP estimation with example dataset
        """
        X = np.array([[1,1,1], [1.5, -0.39, 1.77]]).T
        y = np.array([1,0,1])

        beta_init = np.random.randn(2) / 10
        samples = list(polyagamma_sampler(beta_init, X, y, iterations=250000, mu_prior=np.array([0,0]), sigma_prior=np.array([[1,0], [0,1]])))
        samples = samples[1000:]

        alphas = np.array([alpha for alpha, beta in samples])
        betas = np.array([beta for alpha, beta in samples])

        # Results from Stan with 32 chains and 32,000 iterations
        STAN_ALPHA = 0.04038396
        STAN_BETA = 0.9387743
        print("alpha, PG vs stan", np.mean(alphas), STAN_ALPHA)
        self.assertAlmostEqual(np.mean(alphas), STAN_ALPHA, places=2)
        print("beta, PG vs stan", np.mean(betas), STAN_BETA)
        self.assertAlmostEqual(np.mean(betas), STAN_BETA, places=2)

if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
