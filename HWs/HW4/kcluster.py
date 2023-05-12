import numpy as np

def calc_cost(x_ND, r_NK, mu_KD):
    N, D = x_ND.shape
    cost = 0

    for n in range (N):
        mu_n_D = mu_KD[r_NK[n] == 1, :]
        cost += np.sum(np.square(x_ND[n] - mu_n_D))

    return cost

def update_assignments(x_ND, mu_KD):
    N, D = x_ND.shape
    K, D2 = mu_KD.shape
    assert D == D2

    r_NK = np.zeros((N, K), dtype=np.int32)
    for n in range (N):
        x_n_1D = x_ND[n][np.newaxis, :]
        dist_n_K = np.sum(np.square(x_n_1D - mu_KD), axis= 1)
        target_k = np.argmin(dist_n_K)
        r_NK[n, target_k] = 1

    
    return r_NK


def update_locations(x_ND, r_NK):
    N, D = x_ND.shape
    N2, K = r_NK.shape
    assert N == N2 
    mu_number_KD = np.dot(r_NK.T, x_ND)
    mu_denom_K1 = np.sum(r_NK, axis = 0)[:, np.newaxis]
    mu_KD = mu_number_KD / (1e-6 + mu_denom_K1)

    return mu_KD



x_ND = np.array([
  [-3.0,-2.0],
  [-4.0, 2.0],
  [-3.5, 2.5],
  [-3.5, 2.0],
  [-3.0, 3.0],
  [ 1.5, 3.0],
  [ 2.0, 2.0]])

mu_KD = np.array([
  [-3.0,-2.0],
  [ 1.5, 3.0],
  [ 2.0, 2.0]])


for i in range(2):
    r_NK = update_assignments(x_ND=x_ND, mu_KD=mu_KD)
    cost = calc_cost(x_ND, r_NK, mu_KD)
    print(f'after iteration {i+1}, step 1\n, r_NK = {r_NK}, cost: {cost}')
    mu_KD = update_locations(x_ND=x_ND, r_NK= r_NK)
    cost = calc_cost(x_ND, r_NK, mu_KD)
    print(f'after iteration {i+1}, step 2\n, mu_KD = {mu_KD}, cost: {cost}')

