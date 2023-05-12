### CS136   
#### CP3 Pengcheng Xu  
#### Collaboration Statement:  

- Total hours spent: 8 hrs
- Consult Resources:  
	- Course's website
	- Numpy website
	- Online resource
- Internet

---
<div style="page-break-after: always;"> </div>

**Problem 1.**
---

**1a**  

After implementing `calc_tart_log_pdf` and `main` block in `run_RW_prob1.py`, we get the following figure:  

![1a](/Users/peng/Desktop/CS136/CPs/CP3/pics/p1_1a.png)

**1b**  

Given the info, I'm not very confident that our MCMC chain has converged to the target distribution.  
The reason is that the accept rate is a little too high (i.e. around 0.8 after 10,000 iterations). The situation is similiar to that in **1a** where the rw_stddev = 0.010. Even though in the current setting, the rw_stddev = 100.0, this value could be still too small compared to the target distribution's standard deviation (e.g. what if our target std is 100,000). In other words, the difference (i.e. randomwalk step) between our old Sample (i.e. z\_old) and new Sample (i.e. z\_new) is too small, so we're sort of stuck in the only small portion of the target distribution (and get a high accept rate), but cann't see the whole picuture just like what we did in **figure 1a** where rw_stddev = 0.010. 

<div style="page-break-after: always;"> </div>

**Problem 2.**
---

**2a**  

The posterior predictive visualization for order-2 polynomial model:

![2a](/Users/peng/Desktop/CS136/CPs/CP3/pics/std0_1.png)

**2b**  

The following is the table of per-example score of models with order 0 and order 2 on the provided test set.  

| order |  test score |
-------:|------------:|
| 0     |   -4.533    |
| 2     |   -4.220    |

**2c**  

The following is the screenshot of my implementation of `calc_score `function

![2c](/Users/peng/Desktop/CS136/CPs/CP3/pics/2c.jpg)
