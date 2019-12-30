## Projects

* [Case Study - Bank Marketing](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/bank.nb.html)

This case study uses the [data](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) that includes direct marketing campaigns (i.e., phone calls) of a Portuguese banking institution. The goal is to predict if the client will subscribe a term deposit (indicated in the variable `y`).

The dataset contains $41,188$ examples, $20$ features and a response variable labled by `y`, ordered by date (from May 2008 to November 2010). Since the dataset is imbalanced (about $11\%$ subscribe rate), the $F_1$ score will be used throughout this case study. The similar dataset was analyzed in [Moro et al., 2014](http://dx.doi.org/10.1016/j.dss.2014.03.001). In this case study, we improve their results.

* [Large Scale Time Series Machine Learning](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/housing.nb.html)

Implemented stochastic gradient descent in R and applied it to a [large scale time series dataset](https://www.gov.uk/guidance/about-the-price-paid-data). Moreover, provided a method for model evaluation which can be used in the learning curve for time series data.

* [Movie Recommendations](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/Movies.nb.html)

Implemented the collaborative filtering with regularization algorithm in R using mean square error (MSE) as the cost function and conjugate gradient as the optimization function. Then applied to a [dataset of movie ratings](https://grouplens.org/datasets/movielens/), tested on the author’s ratings using MSE and made personal movie recommendations.

* [Digit Recognizer](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/dr.nb.html)

Used multi-class classification algorithms such as one-vs-all regularized logistic regression and single hidden layer neural networks with regularization to predict [handwritten digits](https://www.kaggle.com/c/digit-recognizer). One-vs-all regularized logistic regression was implemented in R.

* [Credit Card Fraud Detection](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/FraudDetection.nb.html)

Detected anomalous behaviour in [credit card transactions](https://www.kaggle.com/mlg-ulb/creditcardfraud). As the dataset is highly imbalanced, the $F_{1}$ score was used. Applied regularized logistic regression trained on under/oversampling dataset and Gaussian models with selected features. Moreover, implemented in R for Gaussian and Multivariate Gaussian models, and an algorithm to select a threshold using the evaluation metric on a cross-validation set.

* [Spam Emails Classification](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/Spam.nb.html)

Based on the [emails](http://spamassassin.apache.org/old/publiccorpus), created a vocabulary list using some techniques in text processing and converted each email into a feature vector. Then used logistic regression, neural networks and SVM to build spam filters. The regularized logistic regression, single hidden layer neural networks with regularization, learning curve and algorithms to select the regularization parameter and number of units were implemented in R.

* [Is grade of failure related to class schedule?](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/School.html)

Applied SQL technique on a dataset extracted from a [database](https://www.kaggle.com/Madgrades/uw-madison-courses). Since the assumptions of analysis of variance (ANOVA) test for our data are not met, used the Kruskal–Wallis test to discern whether there were real differences between the grade of failure rate of postsecondary students according to their class schedule.

* [Some analyses on PhD graduates in Canada](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/Grad.html)

Visualized some statistics on [graduates in Canada](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3710003001), specially in population of PhD graduates and who pursued further education after PhD graduation. The results were interpreted as geographic patterns in the map of Canada.
