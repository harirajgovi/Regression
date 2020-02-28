# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:29:28 2019

@author: hari4
"""

#importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
x_mtx = dataset.iloc[:, 1:-1].values
y_vctr = dataset.iloc[:, -1].values

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(x_mtx, y_vctr)

y_prdc = dt_regressor.predict(np.array([[6.5]], dtype=np.float))

#gridding the x values by step size 0.1
x_grid = np.arange(min(x_mtx), max(x_mtx), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)

#Data Visualization
plt.scatter(x_mtx, y_vctr, color="red")
plt.plot(x_grid, dt_regressor.predict(x_grid), color="blue")
plt.title("Truth or Bluff (DTR Test)")
plt.xlabel("Job Position Level")
plt.ylabel("Salary")
plt.show()
