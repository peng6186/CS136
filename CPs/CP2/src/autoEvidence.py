import numpy as np
from FeatureTransformPolynomial import PolynomialFeatureTransform
from LinearRegressionPosteriorPredictiveEstimator  import LinearRegressionPosteriorPredictiveEstimator

txfm = PolynomialFeatureTransform(order=1, input_dim=1)
x0 = np.zeros((1,1))

alpha = 1.0
beta = 1.0
ppe = LinearRegressionPosteriorPredictiveEstimator(txfm, alpha, beta)
log_ev = ppe.fit_and_calc_log_evidence(x0, np.asarray([0.0]))
print(np.round(log_ev, 6))
        # -1.265512