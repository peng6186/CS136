### CS136   
#### CP2 Pengcheng Xu  
#### Collaboration Statement:  

- Total hours spent: 8 hrs
- Consultd Resources:  
	- Course's website
	- Numpy website
	- TA xiaohui chen
	- Instrutor Mike
- Internet

---
<div style="page-break-after: always;"> </div>

**Problem 1.**
---
**1a.**  
After reading the starter code about "score" function, we could transfrom it to the following mathmatical fumula:  
$$ score = \frac{\sum_{n = 1}^{N} log\,p_{norm}(t_n \,| \mu = \mu_n, \sigma^{2} = \beta^{-1})}{N} $$

Where N is the number of observations, $p_{norm}$ is the pdf of normal distribution, $\mu$ and $\sigma^{2}$ are mean and variance of the corresponding observation (i.e. $t_n$).  

**1b.**  

After experimenting with three different $\beta$ values {4, 100, 2500} (plz see the pics below, from up to bottom, the scores correspond to order 1, order 3, order 9 models, repectively), we get that the order 1 model performs best under small $\beta$ (i.e. 4); order 3 performs best under medium $\beta$ (i.e. 100); order 9 model performs best under big $\beta$ (i.e. 2500). 
 
 - **$\beta = 4$**  
 
   ![beta4](/Users/peng/Desktop/CS136/CPs/CP2/pics/1b_beta=4.jpg)
 - **$\beta = 100$**  
 
   ![beta100](/Users/peng/Desktop/CS136/CPs/CP2/pics/1b_beta=100.jpg)
 - **$\beta = 2500$**  
 
   ![beta2500](/Users/peng/Desktop/CS136/CPs/CP2/pics/1b_beta=2500.jpg)

 
The reason is that, as the feature order increases, the model tends to more overfit the training data (e.g. observation points are far most from order-1 prediction line, while almost on the order-9 curve).  

Since $t_{n}$ follows a normal distribution, as $\beta$ first becomes bigger (i.e. variance $\sigma^{2}$ becomes smaller), the pdf of normal distribution will become higher and narrower, which cause the probability of some observations (e.g. usally those within $[-3\sigma, 3\sigma]$ around the mean) go up (i.e. that's why we see order-3 model gets a better score as we first increase $\beta$). However, as $\beta$ continue to increase, many observations will be out of proper range (i.e. $[-3\sigma, 3\sigma]$ around the mean), which will produce a very low probability.

**1c.**  
After comparing the visuals of MAP and PPE estimators, we would prefer PPE than MAP (i.e. see the visuals below).

The reason is that the PPE estimator is more flexible (i.e. it performs more stable when x values are far from train data).  

- MAP estimator  
![MAP](/Users/peng/Desktop/CS136/CPs/CP2/pics/q1_MAP.jpg)

- PPE estimator  
![PPE](/Users/peng/Desktop/CS136/CPs/CP2/pics/q1_PPE.jpg)
	
From the graph we could see, for extreme x points (i.e. close to edges), the shadow range of MAP is pretty narrow, which means its tolerance to the points that are a bit far from the prediction line(i.e. the solid line) is bad (almost any points that are not in the prediction line will be considered as outliers). On the other hand, PPE still has a relatively wide shadow range for extreme x values, which shows its better tolerance for "outliers" (i.e. points are a bit far away from the solid prediction line). The result is, for those extreme x points, PPE estimator will give a better score (i.e. higher probability) than MAP. 

<div style="page-break-after: always;"> </div>

**Problem 2.**
---
**2a.**  
After implementing `run_grid_search_5fold_heoldout_score.py`, we get the following figure of probabilistic predictions of CV-selected models:  
![2a](/Users/peng/Desktop/CS136/CPs/CP2/pics/2a.png)

**2b.**  
After implementing `run_grid_search_5fold_heoldout_score.py`, we get the figure of model score vs. polynomial order:  
![2b](/Users/peng/Desktop/CS136/CPs/CP2/pics/2b.png)

<div style="page-break-after: always;"> </div>

**Problem 3.**
---
**3a.**  
After implementing `fit_and_calc_log_evidence` method in `LinearRegreessionPosteriorPredictiveEstimator.py` and `run_grid_search_evidence.py`, we get the following figure of "Probabilistic predictions of evidence-selected models":  
![3a](/Users/peng/Desktop/CS136/CPs/CP2/pics/3a.png)

**3b.**   
After running `run_grid_search_evidence.py`, we also get a figure of "model evidence vs. polynomial order":  
![3b](/Users/peng/Desktop/CS136/CPs/CP2/pics/3b.png)