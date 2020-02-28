# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:39:06 2019

@author: hari4
"""
# y = b0 + b1*x1 
# find all sum(y - y^)^2 and gets minimum value

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")
x_ind_mtx = dataset.iloc[:, :-1].values
y_dep_vctr = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_ind_mtx, y_dep_vctr, test_size=1/3, random_state=0)

"""from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))"""

#import sklearn to fit simple linear regression to training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

my_data = pd.read_csv("my_data.csv")
my_x = my_data.iloc[:, :].values
y_predc = regressor.predict(my_x)

#visualising the training set results

plt.scatter(x_train, y_train, color="red")

plt.plot(x_train, regressor.predict(x_train), color="blue")

plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")

#visualising the test set results

plt.scatter(x_test, y_test, color="red")

plt.plot(x_train, regressor.predict(x_train), color="blue")

plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Person's Salary")
plt.show()