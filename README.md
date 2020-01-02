## Projects

* Case Study - Bank Marketing

This case study uses the [data](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) that were collected from a Portuguese marketing campaign (i.e., phone calls) related with bank deposit subscription, from 2008 to 2013. The goal is to predict if the client will subscribe a term deposit.

In this work, we analyze a set of 20 features given by the data in [Bank Marketing - EDA](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/bank_EDA.nb.html). 

We also compare 4 models: decision tree, logistic regression, 2-layer neural network and 3-layer neural network in [Bank Marketing - Modeling](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/bank_modeling.nb.html). By using two metrics F1 score and area of the receiver operating characteristic curve (AUC), the four models were trained on the randomly selected set (80% of the data, 32950 examples) and tested on another randomly selected set (10% of the data, 4119 examples). The best model is given by the 3-layer neural network (F1=0.64, reported on the rest of 10% of the data 4119 examples). This improves the result in [Table 3, Moro et al., 2014](http://dx.doi.org/10.1016/j.dss.2014.03.001). Moreover, the decision tree model was applied to measure feature importance and reveal several key features, e.g., phone call duration, social and economic indicators such as employment rate and contact month of year.

For the business purpose, the 3-layer neural network model can predict the success of telemarketing calls for selling bank deposits. Such model can increase campaign efficiency by helping in a better selection of a high quality and affordable list of potential buying customers.

* [Movie Recommendations](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/Movies.nb.html)

Implemented the collaborative filtering with regularization algorithm in R using mean square error (MSE) as the cost function and conjugate gradient as the optimization function. Then applied to a [dataset of movie ratings](https://grouplens.org/datasets/movielens/), tested on the author’s ratings using MSE and made personal movie recommendations.

* [Credit Card Fraud Detection](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/FraudDetection.nb.html)

Detected anomalous behaviour in [credit card transactions](https://www.kaggle.com/mlg-ulb/creditcardfraud). As the dataset is highly imbalanced, the $F_{1}$ score was used. Applied regularized logistic regression trained on under/oversampling dataset and Gaussian models with selected features. Moreover, implemented in R for Gaussian and Multivariate Gaussian models, and an algorithm to select a threshold using the evaluation metric on a cross-validation set.

* [Spam Emails Classification](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/Spam.nb.html)

Based on the [emails](http://spamassassin.apache.org/old/publiccorpus), created a vocabulary list using some techniques in text processing and converted each email into a feature vector. Then used logistic regression, neural networks and SVM to build spam filters. The regularized logistic regression, single hidden layer neural networks with regularization, learning curve and algorithms to select the regularization parameter and number of units were implemented in R.

* [Is grade of failure related to class schedule?](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/School.html)

Applied SQL technique on a dataset extracted from a [database](https://www.kaggle.com/Madgrades/uw-madison-courses). Since the assumptions of analysis of variance (ANOVA) test for our data are not met, used the Kruskal–Wallis test to discern whether there were real differences between the grade of failure rate of postsecondary students according to their class schedule.
