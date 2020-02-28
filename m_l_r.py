# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:00:22 2019

@author: hari4
"""

# y = b0 + b1*x1 + b2*x2 +... + bn*xn
#multiple linear regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("50_Startups.csv")
x_mtx = dataset.iloc[:, :-1].values
y_vctr = dataset.iloc[:, -1].values

#importing sklearn to convert categorical data to numbers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x_mtx = np.array(ct.fit_transform(x_mtx), dtype=np.float)

#avoiding the dummy variable trap
x_mtx = x_mtx[:, 1:]

#importing sklearn for splitting data set into training set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_mtx, y_vctr, test_size=0.2, random_state=0)

#importong sklearn for feature scaling
"""from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))"""

#import sklearn to fit multiple linear regression to training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_predc = regressor.predict(x_test)

#import statsmodels for backward elimination method

import statsmodels.api as sm

x_mtx = np.append(arr = np.ones((50, 1)).astype(int), values=x_mtx, axis=1)

x_opt = x_mtx[:, [0, 1, 2, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y_vctr, exog=x_opt).fit()
regressor_ols.summary()

x_opt = x_mtx[:, [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y_vctr, exog=x_opt).fit()
regressor_ols.summary()

x_opt = x_mtx[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y_vctr, exog=x_opt).fit()
regressor_ols.summary()

x_opt = x_mtx[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog=y_vctr, exog=x_opt).fit()
regressor_ols.summary()

x_opt = x_mtx[:, [0, 3]]
regressor_ols = sm.OLS(endog=y_vctr, exog=x_opt).fit()
regressor_ols.summary()

am = round(float(regressor_ols.pvalues[0].astype(float)), 3)

print(len(x_mtx[0]))

