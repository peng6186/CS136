import numpy as np
N, D = 100, 1

prng = np.random.RandomState(0)

x_ND = prng.randn(N, D)

x_ND.shape == (N, D)
    # True


t_N = 5 * x_ND[:,0] + 1

t_N.shape == (N,)
   #  True


from FeatureTransformPolynomial import PolynomialFeatureTransform

txfm = PolynomialFeatureTransform(order=1, input_dim=D)


alpha = 1.0

beta = 20.0

from LinearRegressionPosteriorPredictiveEstimator import LinearRegressionPosteriorPredictiveEstimator
ppe = LinearRegressionPosteriorPredictiveEstimator(txfm, alpha, beta)

ppe = ppe.fit(x_ND, t_N)

ppe.mean_M.shape
   # (2,)

ppe.mean_M
   # array([0.99964554, 4.99756957])

    ## Check log evidence

log_ev = ppe.fit_and_calc_log_evidence(x_ND, t_N)

isinstance(log_ev, float)
   # True

np.round(log_ev, 5)
   # 37.28976