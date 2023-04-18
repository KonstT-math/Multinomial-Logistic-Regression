# split categorical variable (target) into dummy variables

import pandas as pd


df = pd.read_csv('iris_std.csv')

df.columns =['0', '1', '2', '3', '4']

df_dum = pd.get_dummies(df, columns=['4'])

df.iloc[:,4].replace(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'],
                        [0, 2, 1], inplace=True)

df = pd.concat([df_dum, df.iloc[:,4]], axis=1)





# create new csv file with new dataframe
df.to_csv(r'iris_dummy.csv', index = False, header=True)
