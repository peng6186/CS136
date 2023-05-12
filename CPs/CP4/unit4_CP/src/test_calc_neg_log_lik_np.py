## Setup: Create useful parameters
import numpy as np
from calc_neg_log_lik__np import *
np.set_printoptions(precision=3, suppress=1)
K = 3
D = 2
log_pi_K = np.log([1./3, 1./3, 1./3]);
stddev_KD = np.ones((K, D))
mu_KD = np.zeros((K, D))
mu_KD[0,:] = -1.0
mu_KD[-1,:] = +1.0
mu_KD
# array([[-1., -1.],
#        [ 0.,  0.],
#        [ 1.,  1.]])

## Neg. likelihood of empty dataset should be zero
empty_ND = np.zeros((0,D))
calc_neg_log_lik(empty_ND, log_pi_K, mu_KD, stddev_KD)
# -0.0

## Neg. likelihood of dataset of all zeros should be large
N = 4
allzero_x_ND = np.zeros((N,D))
print("%.3f" % calc_neg_log_lik(allzero_x_ND, log_pi_K, mu_KD, stddev_KD))
#9.540

## Neg. likelihood of bigger dataset should be even larger
N = 8
bigzero_x_ND = np.zeros((N,D))
print("%.3f" % calc_neg_log_lik(bigzero_x_ND, log_pi_K, mu_KD, stddev_KD))
#19.080