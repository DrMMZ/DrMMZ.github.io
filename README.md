## **Projects**

----

**[Classification on Imbalanced Structured Data using Fully Connected Neural Networks](http://)**

This work uses the [data](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) that were collected from a marketing campaign (i.e., phone calls) related with bank deposit subscription. The goal is to predict whether or not the client will subscribe a term deposit. See the original paper published in [Moro et al., 2014](http://dx.doi.org/10.1016/j.dss.2014.03.001).

We first analyze features given by the data in [Bank Marketing - EDA](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/bank_EDA.nb.html), which is done by R.

Since the data is imbalanced, oversampling and undersampling methods are used in training. However, their results are similar, based on AUC (0.95+/-0.01) and recall (0.91+/-0.06) metrics in a 2-layer neural network with 20 hidden units, ReLU activation and adam optimization. We also use batch normalization to speed up training and L1-regularization to reveal several key features. At the end, error analysis is provided. This part is done by Python. See the [notebook](http://) for the work. 

For the business purpose, the shallow 2-layer neural network model can predict the success of telemarketing calls for selling bank deposits. Such model can increase campaign efficiency by helping in a better selection of a high quality and affordable list of potential buying customers.

----

**[Movie Recommendations](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/Movies.nb.html)**

Implemented in R the collaborative filtering with regularized mean square loss function and Quasi-Newton (BFGS) optimization. Then applied to a [dataset of movie ratings](https://grouplens.org/datasets/movielens/), tested on the author’s ratings and made personal movie recommendations.

----

**[Is grade of failure related to class schedule?](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/School.html)**

Applied SQL technique on a dataset extracted from a [database](https://www.kaggle.com/Madgrades/uw-madison-courses). Since the assumptions of analysis of variance (ANOVA) test for our data are not met, used the Kruskal–Wallis test to discern whether there were real differences between the grade of failure rate of postsecondary students according to their class schedule.
