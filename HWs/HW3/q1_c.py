from os import X_OK
import numpy as np
def sample_from_mv_gaussian(mu_D, Sigma_DD, random_state=np.random):
    ''' Draw sample from multivariate Gaussian
    Args
    ----
    mu_D : 1D array, size D
    Mean vector
    Sigma_DD : 2D array, shape (D, D)
    Covariance matrix. Must be symmetric and positive definite.
    Returns
    -------
    x_D : 1D array, size D
    Sampled value of Gaussian with provided mean and covariance
    '''

    D = mu_D.size
    L_DD = np.linalg.cholesky(Sigma_DD) # compute L from Sigma
    # GOAL: draw each entry of u_D from standard Gaussian
    u_D = random_state.randn(D) #  use random_state.randn(...)
    # GOAL: Want x_D  ̃ Gaussian(mean = m_D, covar=Sigma_DD)
    x_D = L_DD@u_D + mu_D # transform u_D into x_D
    return x_D

x = sample_from_mv_gaussian(np.array([0, 0]), np.array([[1,0], [0, 1]]))
print(x)