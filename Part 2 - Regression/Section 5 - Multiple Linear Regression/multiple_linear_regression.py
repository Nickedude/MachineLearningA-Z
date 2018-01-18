# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:54:02 2018

@author: Nickedude
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the state
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
X[:, 3] = labelEncoderX.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap (not really needed in these libraries)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# We use 20% for testing the model and 80% for training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling will be done by the library
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)
plt.scatter(y_test, y_test-y_pred, color = 'red')
plt.xlabel('Real profits')
plt.ylabel('Difference between prediction and test data')
plt.show()

# Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
#significance level = 0.05
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Ordinary least squares
regressor_OLS.summary()
