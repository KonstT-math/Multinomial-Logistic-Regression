# Multinomial-Logistic-Regression
Multinomial Logistic Regression on iris dataset (3 categories in target variable) from scratch

Preprocessing on iris dataset: standard rescaling with mean 0 and standard deviation 1 for each feature variable (see preprocess1.py); create dummy variables for the three categories of the nominal target variable (see preprocess4.py)

Create 3 logistic regression classifiers, one for each dummy variable.

Take argmax on the probability given by each classifier, in order to make predictions.
