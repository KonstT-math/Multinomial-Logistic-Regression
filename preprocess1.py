import numpy as np
import pandas as pd
from scipy.stats import kstest
#from sklearn import preprocessing

# get a column from dataframe
def select_data(data, ny):
	
	yName = data.columns[ny]
	Y = data[yName]

	return Y

# see which feature is normally distributed from dataframe
def normal_test(df):
	for i in range(len(df.columns)):
		y = select_data(df,i)
		p = kstest(y,'norm')
		print("feature {}, p-value = {}".format(i,p[1]))

# rescale feature i in dataframe
def standard_rescale(df, i):
	y = select_data(df,i)
	m = np.mean(y)
	s = np.std(y)
	y = (y-m)/s
	return y 

# log-transform feature of dataframe
def log_transform(df,i):
	y = select_data(df,i)
	y = np.log(y)
	return y

# square root transform feature of dataframe
def sqrt_transform(df,i):
	y = select_data(df,i)
	y = np.sqrt(y)
	return y

# cube root transform feature of dataframe
def cbrt_transform(df,i):
	y = select_data(df,i)
	y = np.cbrt(y)
	return y

# transform dataframe into one of: standard, log, sqrt, cbrt
def transform_dataframe(df, transformation):
	df_new = []
	if transformation == "standard":
		for i in range(len(df.columns)-1):
			y = standard_rescale(df,i)
			df_new.append(y)
		df_new.append(df.iloc[:,no_feats])
	elif transformation == "log":
		for i in range(len(df.columns)-1):
			y = log_transform(df,i)
			df_new.append(y)
		df_new.append(df.iloc[:,no_feats])
	elif transformation == "sqrt":
		for i in range(len(df.columns)-1):
			y = sqrt_transform(df,i)
			df_new.append(y)
		df_new.append(df.iloc[:,no_feats])
	elif transformation == "cbrt":
		for i in range(len(df.columns)-1):
			y = cbrt_transform(df,i)
			df_new.append(y)
		df_new.append(df.iloc[:,no_feats])
	else:
		return "wrong arguments"

	df_new = pd.DataFrame(df_new)
	df_new = df_new.T

	return df_new
		


df = pd.read_csv('iris.csv')
no_feats = 4

df.columns =['0', '1', '2', '3', '4']


#normal_test(df)

df_standard = transform_dataframe(df, "standard")
#df_log = transform_dataframe(df, "log")
#df_sqrt = transform_dataframe(df, "sqrt")
#df_cbrt = transform_dataframe(df, "cbrt")
#df_wrong = transform_dataframe(df, "lo")

#print("standard-----------------------------------------")
#normal_test(df_standard)
#print("log-----------------------------------------")
#normal_test(df_log)
#print("square root-----------------------------------------")
#normal_test(df_sqrt)
#print("cube root-----------------------------------------")
#normal_test(df_cbrt)

result = df_standard

# create new csv file with new dataframe
result.to_csv(r'iris_std.csv', index = False, header=True)

