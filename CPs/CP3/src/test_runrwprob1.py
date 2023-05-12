import numpy as np
from run_RW_prob1 import calc_target_log_pdf


z_d = np.array([-0.1, 0.5])
res = calc_target_log_pdf(z_d)
print(res)