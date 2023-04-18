
# multinomial logistic regression for 3 classes in target

import numpy as np
import pandas as pd
from lreg import LogisticRegression

test_length = 74

nofeats = 4

# -----------------------------------------

# data:

# for the iris dataset, we split the target variable into 3 dummy variables, and the features are transformed in standard scale with mean 0 and std 1 (see preprocess1.py and preprocess4.py)
data = pd.read_csv('iris_dummy.csv')

data = np.array(data)
m,n = data.shape

np.random.shuffle(data)

data_test = data[0:test_length]
X_test = data_test[:,0:nofeats]
Y_test0 = data_test[:,nofeats]
Y_test1 = data_test[:,nofeats+1]
Y_test2 = data_test[:,nofeats+2]
Y_test_all = data_test[:,nofeats+3]

Y_test0 = Y_test0.T
Y_test1 = Y_test1.T
Y_test2 = Y_test2.T
Y_test_all = Y_test_all.T

data_train = data[test_length:m]
X_train = data_train[:, 0:nofeats] 
Y_train0 = data_train[:,nofeats] 
Y_train1 = data_train[:,nofeats+1]
Y_train2 = data_train[:,nofeats+2]

Y_train0 = Y_train0.T
Y_train1 = Y_train1.T
Y_train2 = Y_train2.T

# -----------------------------------------

def accuracy(y_pred, y_test):
	return np.sum(y_pred==y_test)/len(y_test)

# create 3 classifiers, one for each dummy target
classifier0 = LogisticRegression()
classifier1 = LogisticRegression()
classifier2 = LogisticRegression()

# ----- train classifiers for each binary dummy target -----

# --- Binary classification problem 1 ---
print("Binary classifier 1:")
classifier0.fit(X_train , Y_train0)
y_pred0 = classifier0.predict(X_test)

print("real :",Y_test0)
print("predicts: ",y_pred0)

print("--------------")

# --- Binary classification problem 2 ---
print("Binary classifier 2:")
classifier1.fit(X_train , Y_train1)
y_pred1 = classifier1.predict(X_test)

print("real :",Y_test1)
print("predicts: ",y_pred1)

print("--------------")

# --- Binary classification problem 3 ---
print("Binary classifier 3:")
classifier2.fit(X_train , Y_train2)
y_pred2 = classifier2.predict(X_test)

print("real :",Y_test2)
print("predicts: ",y_pred2)

print("--------------")

# -------------------- ONE VS ALL ---------------------

total_pred = np.array([y_pred0, y_pred1, y_pred2])

# take the max from each probability obtained from each classifier, and create an array with their indices:
multi_pred = np.argmax(total_pred,axis=0)
print(multi_pred)

# compare the indices with the value of the target variable to get accuracy:
acc = accuracy(multi_pred, Y_test_all)

print(acc)

