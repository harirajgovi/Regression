# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:18:41 2019

@author: hari4
"""

#polynomial regression
# y = b0 + b1x1 + b2x1^2 + ... + bnx1^n 

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
x_mtx = dataset.iloc[:, 1:-1].values
y_vctr = dataset.iloc[:, -1].values

#Splitting dataset into training set and test set
"""from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_mtx, y_vctr, test_size=0.2, random_state=0)"""

#Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg1 = LinearRegression()
lin_reg1.fit(x_mtx, y_vctr)
y_prdc1 = lin_reg1.predict(x_mtx)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x_mtx)
poly_reg.fit(x_poly, y_vctr)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y_vctr)
y_prdc2 = lin_reg2.predict(x_poly)

#Linear Regression data visualization
plt.scatter(x_mtx, y_vctr, color="red")
plt.plot(x_mtx, lin_reg1.predict(x_mtx), color="blue")
plt.title("Truth or Bluff(LR Test)")
plt.xlabel("Job position level")
plt.ylabel("salary")
plt.show()

#Polynomial Regression regression data visualization
x_grid = np.arange(min(x_mtx), max(x_mtx), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x_mtx, y_vctr, color="red")
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color="blue")
plt.title("Truth or Bluff(PR Test)")
plt.xlabel("Job position level")
plt.ylabel("salary")
plt.show()

#results
y_res1 = lin_reg1.predict(np.array(6.5, dtype=np.float).reshape(-1, 1))
y_res2 = lin_reg2.predict(poly_reg.fit_transform(np.array(6.5, dtype=np.float).reshape(-1, 1)))


