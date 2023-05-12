import numpy as np
from viterbi_alg import *


np.set_printoptions(precision=5, suppress=1)
T = 10
K = 2
D = 4
pi_K = np.ones(K) / K
A_KK = (np.ones((K,K)) + 49.0 * np.eye(K)) / (49 + K)
print(A_KK)
# array([[0.98039, 0.01961],
#        [0.01961, 0.98039]])

## Create mean and stddev for each state and dim
# mean will be -1 in first state, and +1 in second state
# stddev always 1
mu_KD = np.ones((K, D))
mu_KD[0] *= -1
stddev_KD = np.ones((K, D))

## Sample 'simple' dataset with T examples from state 0, then T more from state 1
import scipy.stats
prng = np.random.RandomState(0)
x_state0_TD = prng.randn(T, D) * stddev_KD[0] + mu_KD[0]
x_state1_TD = prng.randn(T, D) * stddev_KD[1] + mu_KD[1]
x_TD = np.vstack([x_state0_TD, x_state1_TD])
log_lik_TK = np.vstack([
	np.sum(scipy.stats.norm.logpdf(x_TD, mu_KD[k], stddev_KD[k]), axis=1)
 	for k in range(K)]).T

## Run Viterbi algorithm
zhat_T, log_pdf_x_and_z = run_viterbi_algorithm(np.log(pi_K), np.log(A_KK), log_lik_TK)

## Verify estimated hidden state sequence is correct
print(zhat_T[:T])
#array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)

print(zhat_T[T:])
#array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

## Verify computed log joint proba using HMM is better than a plain mixture model
print("% .5f" % log_pdf_x_and_z)
#-119.02517
gmm_log_pdf_x_and_z = np.sum(np.log(pi_K[zhat_T]) + log_lik_TK[(np.arange(T+T), zhat_T)])
print("% .5f" % gmm_log_pdf_x_and_z)
#-127.90669

