# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:18:18 2018

@author: Nickedude
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
X[:, 3] = labelEncoderX.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toArray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# We use 20% for testing the model and 80% for training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling (getting variables in the same range and scale)
'''from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_train = scX.fit_transform(X_train)    # We FIT it to X_train, not needed for X_train though
X_test = scX.transform(X_test)'''

