# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values    # We want X as a matrix
y = dataset.iloc[:, 2].values


# Due to lack of data we won't split into test and train set.
# No feature scaling, algorithm will fix it
'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# We use 20% for testing the model and 80% for training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling (getting variables in the same range and scale)
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_train = scX.fit_transform(X_train)    # We FIT it to X_train, not needed for X_train though
X_test = scX.transform(X_test)'''
'''

