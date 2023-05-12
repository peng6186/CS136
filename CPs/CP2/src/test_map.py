import numpy as np
from LinearRegressionMAPEstimator import LinearRegressionMAPEstimator 

N, D = 100, 1
prng = np.random.RandomState(0)
x_ND = prng.randn(N, D)
t_N = 5 * x_ND[:,0] + 1
print(t_N.shape == (N,))
#    True

from FeatureTransformPolynomial import PolynomialFeatureTransform
txfm = PolynomialFeatureTransform(order=1, input_dim=D)

alpha = 1.0
beta = 20.0
map = LinearRegressionMAPEstimator(txfm, alpha, beta)
map = map.fit(x_ND, t_N)
print(map.w_map_M.shape)
#    (2,)
print(map.w_map_M)
   # array([0.99964554, 4.99756957])
print(txfm.get_feature_names())
   # ['bias', 'x 0^1']