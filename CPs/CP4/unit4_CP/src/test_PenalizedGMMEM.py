import numpy as np
from GMM_PenalizedMLEstimator_EM import *

np.set_printoptions(suppress=False, precision=3, linewidth=80)
D = 2

## Verify that variance penalty works as expected
# Empty components (with no assigned data) should have variance equal to the intended "mode" of the penalty
# We'll use a mode of 2.0 (so stddev = sqrt(2.0) = 1.414...)
gmm_em = GMM_PenalizedMLEstimator_EM(K=3, D=2, seed=42, variance_penalty_mode=2.0)
empty_ND = np.zeros((0,D))
log_pi_K, mu_KD, stddev_KD = gmm_em.generate_initial_parameters(empty_ND)
print(calc_neg_log_lik(empty_ND, log_pi_K, mu_KD, stddev_KD))
# -0.0
gmm_em.fit(empty_ND, verbose=False)
print(gmm_em.stddev_KD)
# array([[1.414, 1.414],
#        [1.414, 1.414],
#        [1.414, 1.414]])

N = 25; K = 3
prng = np.random.RandomState(8675309)
x1_ND = 0.1 * prng.randn(N, D) + np.asarray([[0, 0]])
x2_ND = 0.1 * prng.randn(N, D) + np.asarray([[-1, 0]])
x3_ND = np.asarray([[0.2, 0.05]]) * prng.randn(N, D) + np.asarray([[0, +1]])
x_ND = np.vstack([x1_ND, x2_ND, x3_ND])
gmm_em = GMM_PenalizedMLEstimator_EM(K=3, D=2, seed=42, variance_penalty_mode=2.0, max_iter=1)

gmm_em.stddev_KD = 0.1 * np.ones((K,D))
gmm_em.stddev_KD[-1] = [0.2, 0.05]
gmm_em.mu_KD = np.asarray([[0, 0], [-1., 0], [0, 1.]])
gmm_em.log_pi_K = np.log(1./3 * np.ones(K))
print(gmm_em.estep__calc_r_NK(x_ND[:3]))
# array([[1.000e+00, 5.336e-25, 3.829e-75],
#        [1.000e+00, 2.151e-17, 3.063e-97],
#        [1.000e+00, 4.367e-19, 1.984e-90]])
print(gmm_em.estep__calc_r_NK(x_ND[-3:]))
# # array([[4.752e-25, 1.362e-38, 1.000e+00],
# #        [2.278e-17, 7.579e-46, 1.000e+00],
# #        [4.189e-22, 4.117e-34, 1.000e+00]])



gmm_em.fit(x_ND, verbose=False)
print(np.exp(gmm_em.log_pi_K))
# array([0.333, 0.333, 0.333])
print(gmm_em.mu_KD)
# array([[-0.007,  0.01 ],
#        [-1.008,  0.009],
#        [-0.005,  1.005]])
print(gmm_em.stddev_KD)
# array([[0.076, 0.091],
#        [0.098, 0.103],
#        [0.24 , 0.042]])

gmm_em = GMM_PenalizedMLEstimator_EM(
    K=3, D=2, seed=42, variance_penalty_mode=2.0, max_iter=1000,
   do_double_check_correctness=True)
gmm_em.fit(x_ND, verbose=False)
print(np.exp(gmm_em.log_pi_K))
#array([0.333, 0.333, 0.333])
print(gmm_em.mu_KD)
# array([[-1.008,  0.009],
#        [-0.005,  1.005],
#        [-0.007,  0.01 ]])
print(gmm_em.stddev_KD)
# array([[0.098, 0.103],
#        [0.24 , 0.042],
#        [0.076, 0.091]])
