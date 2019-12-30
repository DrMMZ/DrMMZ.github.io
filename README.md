## Projects

* [Case Study - Bank Marketing](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/bank.nb.html)

This case study uses the [data](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) that includes direct marketing campaigns (i.e., phone calls) of a Portuguese banking institution. The goal is to predict if the client will subscribe a term deposit (indicated in the variable `y`).

The similar dataset was analyzed in [Moro et al., 2014](http://dx.doi.org/10.1016/j.dss.2014.03.001). In this work, we test four binary classificaition models, using the packages `rminer`, `rpart` and `nnet` from `R` and the package `nn_model_np` of `Python` implemented by the author: decision tree (`R`), logistic regression (`R`), $2$-layer neural network (`R`) and $3$-layer neural network (`Python`). By using a certain splitting ratio and more complex model $3$-layer neural network, we are able to improve the results in [Table 3, Moro et al., 2014](http://dx.doi.org/10.1016/j.dss.2014.03.001).

* [Movie Recommendations](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/Movies.nb.html)

Implemented the collaborative filtering with regularization algorithm in R using mean square error (MSE) as the cost function and conjugate gradient as the optimization function. Then applied to a [dataset of movie ratings](https://grouplens.org/datasets/movielens/), tested on the author’s ratings using MSE and made personal movie recommendations.

* [Credit Card Fraud Detection](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/FraudDetection.nb.html)

Detected anomalous behaviour in [credit card transactions](https://www.kaggle.com/mlg-ulb/creditcardfraud). As the dataset is highly imbalanced, the $F_{1}$ score was used. Applied regularized logistic regression trained on under/oversampling dataset and Gaussian models with selected features. Moreover, implemented in R for Gaussian and Multivariate Gaussian models, and an algorithm to select a threshold using the evaluation metric on a cross-validation set.

* [Spam Emails Classification](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/Spam.nb.html)

Based on the [emails](http://spamassassin.apache.org/old/publiccorpus), created a vocabulary list using some techniques in text processing and converted each email into a feature vector. Then used logistic regression, neural networks and SVM to build spam filters. The regularized logistic regression, single hidden layer neural networks with regularization, learning curve and algorithms to select the regularization parameter and number of units were implemented in R.

* [Is grade of failure related to class schedule?](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/School.html)

Applied SQL technique on a dataset extracted from a [database](https://www.kaggle.com/Madgrades/uw-madison-courses). Since the assumptions of analysis of variance (ANOVA) test for our data are not met, used the Kruskal–Wallis test to discern whether there were real differences between the grade of failure rate of postsecondary students according to their class schedule.
