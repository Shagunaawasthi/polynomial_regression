# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 23:05:13 2020

@author: shaguna awasthi
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
#fitting polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
lin_reg2.fit(X_poly, y)

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
#creating a new linear regression model to fit our poly into it
lin_reg2 = LinearRegression()
#visualizing linear reg results
plt.scatter(X,y, color="red")
plt.plot(X,lin_reg.predict(X), color="green")
plt.title("truth or bluff linear reg")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()
#visualize polynomial regression results
#instead of incrementing with 1 on the axis we will increment it with 0.1 
X_grid= np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y, color="red")
#we won't put lin_reg2 cause apparently it is a linear reg object
#now if we put X_poly and for some reason plot function mai X change hua toh X_poly toh already calculated hai.
#so we would rather put whole equation that calculated X_poly in the first place

plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)), color="green")
plt.title("truth or bluff poly reg")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()

#predicting new result with Linear reg
lin_reg.predict(6.5)

#predicting a new result with polynomial reg