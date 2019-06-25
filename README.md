## Projects

* [Spam Emails Classification](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/Spam.nb.html)

We implement logistic regression with regularization and single hidden layer neural networks with regularization in R. All source code are attached. Then we use them alone with SVM to build spam filters. 

Given an email, we train a classifier to classify whether the email is spam or non-spam. In particular, we create a vocabulary list $\mathcal{L}$ using some standard techniques in text processing and convert each email into a feature vector $\vec{x} \in \mathbb{R}^{n}$ for the size $n=1662$ of $\mathcal{L}$. 

The dataset is based on a subset of the [SpamAssassin Public Corpus](http://spamassassin.apache.org/old/publiccorpus), which has 6046 messages, with about a 31% spam ratio. We only consider the body of the email, excluding the email headers. The dataset is divided into 60% for training, 20% for cross validation and 20% for test. By using the cross validation set, we are able to determine the optimal regularization parameters in all classifiers.

* [Is grade of failure related to class schedule?](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/School.html)

We would like to discern whether there are real differences between the grade of failure rate of postsecondary students according to their class schedule (4 groups): starting in Early Morning (before 9:50 am), Morning (from 9:55 am to 12:00 pm), Afternoon (from 12:05 pm to 2:25 pm), and Late Afternoon (after 2:30 pm). 

We will use a dataset extracted from a [database](https://www.kaggle.com/Madgrades/uw-madison-courses). Our data includes reports for all courses, schedules, and grade reports for the academic year 2016-2017 (Fall, Spring and Summer terms) at the University of Wisconsin - Madison. 

By using analysis of variance (ANOVA) test, one can check whether the means across the above 4 groups are equal. However, as the assumptions of ANOVA test for our data are not met, we use the Kruskal–Wallis test. The p-value of the test is less than 0.05, indicating the evidence is strong enough to reject the null hypothesis at a significance level of 0.05. That is, the data provides strong evidence that the average grade of failure rate varies by 4 different class schedules. By using Wilcoxon rank test with the Bonferroni correction, we are able to find a strong evidence of a difference in the means of groups Morning and Late Afternoon. 

* [Some analyses on PhD graduates in Canada](http://htmlpreview.github.io/?https://github.com/DrMMZ/drmmz.github.io/blob/master/Grad.html)

We are interested in some statistics on PhD graduates in Canada, specially in population of PhD graduates and who pursued further education after PhD graduation. The data is extracted from [National graduates survey, postsecondary graduates by province of study and level of study, Table 37-10-0030-01, Statistics Canada](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3710003001).

Our analyses show that the population of PhD graduates in Canada had a decrease from 2000 to 2005, and an increase from 2005 to 2010. In particular, British Columbia had the most decrease from 2000 to 2005; Quebec had the most growth from 2005 to 2010; and from 2000 to 2010, Saskatchewan had the most growth.

Interestingly, in the population of PhD graduates in Canada, there are some graduates who pursued further education after graduation. We found that the PhD graduates in Alberta who pursued further education after graduation is the only province in Canada had a decrease from 2000 to 2010; in the contrast, British Columbia and Ontario shared the most growth. The results are interpreted as geographic patterns in the map of Canada.
