import numpy as np
import scipy.stats
from RandomWalkSampler import RandomWalkSampler
#Example Usage
  
# Define a target distribution (2-dim. standard normal)
def calc_tilde_log_pdf(z_D):
    logpdf1 = scipy.stats.norm.logpdf(z_D[0], 0, 1)
    logpdf2 = scipy.stats.norm.logpdf(z_D[1], 0, 1)
    return logpdf1 + logpdf2

    # Create sampler
rw_stddev_D = .5 * np.ones(2)
sampler = RandomWalkSampler(
    
    calc_tilde_log_pdf, [rw_stddev_D], random_state=42)

    # Draw samples starting a specified initial value
z_D_list, info = sampler.draw_samples(np.zeros(2), 30000)

    # Use samples in to estimate mean of the distribution
print(np.mean(np.vstack(z_D_list), axis=0))
    #array([-0.00954532,  0.01338581])

print(info['accept_rate'])
    #0.7553