# Artificial Neural Network

# Installing Theano
# Open source library for computations. 
# Doesn't only run on the CPU, but on the GPU as well.
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Also a library for computations. Can run on a CPU or GPU. 
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# Used for building deep learning models with few lines of code.
# "Wraps" Theano and Tensorflow.
#pip install --upgrade keras

# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Avoid dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling (getting variables in the same range and scale, needed when the algorithm is based on eucledian distance)from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_train = scX.fit_transform(X_train)    # We fit scX to X_train, and transform X_train
X_test = scX.transform(X_test)

# Part 2 - Creating the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential    # Used to initialize the ANN
from keras.layers import Dense         # Used for creating the layers


# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Arguments: nr of nodes in layer, init for weights, activation function, input dimension
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11)) 

# Second hidden layer, knows to expect 6 nodes from previous layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Add output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# Optimizer - The algorithm for finding the weights for each node, we will use SGD.
# Loss - The objective function/cost function. Sum of squared differences for example. SGD will optimize this function. Since we use a sigmoid function we will use the logarithmic loss function.
# Several outputs => use 'categorical_crossentropy'
# Metrics - Defines how the ANN is evaluated. A criterion. When the weights are updated, after each observation/batch, the algorithm uses this criterion to improve the models performance. 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Choose the number of epochs - how many times the ANN trains on the dataset

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)