
# Forward algorithm for Hidden Markov Models.

# Includes:
# * solution to run_forward_algorithm
# * a __main__ method with example input
# * doctests showing expected behavior

# Usage
# -----
# To see forward messages on simple example printed to stdout, do
# $ python forward_alg.py

# To verify if the script passes the doctest tests, do
# $ python -m doctest forward_alg.py

# Examples
# --------

import numpy as np
from forward_alg import *

np.set_printoptions(precision=5, suppress=1)
T = 10
K = 2
D = 4
pi_K = np.ones(K) / K
A_KK = (np.ones((K,K)) + 9.0 * np.eye(K)) / (9 + K)
print(A_KK)
# array([[0.90909, 0.09091],
#        [0.09091, 0.90909]])

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

## Run forward algorithm
alpha_TK, hmm_log_pdf_x = run_forward_algorithm(np.log(pi_K), np.log(A_KK), log_lik_TK)

## Verify first 5 forward message vectors are correct
print(alpha_TK[:5])
# array([[0.0591 , 0.9409 ],
#        [0.9427 , 0.0573 ],
#        [0.99758, 0.00242],
#        [0.99905, 0.00095],
#        [0.99985, 0.00015]])

## Verify last 5 forward message vectors are correct
print(alpha_TK[-5:])
# array([[0.04076, 0.95924],
#        [0.00077, 0.99923],
#        [0.00003, 0.99997],
#        [0.00007, 0.99993],
#        [0.00102, 0.99898]])

## Verify computed log proba using HMM is better than a plain mixture model
print("% .5f" % hmm_log_pdf_x)
# -117.84267
gmm_log_pdf_x = np.sum(logsumexp(np.log(pi_K)[np.newaxis,:] + log_lik_TK, axis=1))
print("% .5f" % gmm_log_pdf_x)
# -124.25249
