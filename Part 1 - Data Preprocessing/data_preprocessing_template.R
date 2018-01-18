# Data preprocessing

# Importing the dataset
dataset = read.csv('Data.csv')
#dataset = dataset[, 2:3] # Selecting subset

# Splitting the dataset into the Training set and Test set
# install.packages('caTools') # Only needs to be done once
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
trainingSet = subset(dataset, split == TRUE)
testSet = subset(dataset, split == FALSE)

# Feature scaling
# Exclude non-numeric columns
# trainingSet[, 2:3] = scale(trainingSet[, 2:3])
# testSet[, 2:3] = scale(testSet[, 2:3])
