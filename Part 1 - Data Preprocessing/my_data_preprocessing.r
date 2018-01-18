# Data preprocessing

# Importing the dataset
dataset = read.csv('Data.csv')

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

# Encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                          levels = c('Yes', 'No'),
                          labels = c(1,0))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools') # Only needs to be done once
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
trainingSet = subset(dataset, split == TRUE)
testSet = subset(dataset, split == FALSE)

# Feature scaling
# Exclude non-numeric columns
trainingSet[, 2:3] = scale(trainingSet[, 2:3])
testSet[, 2:3] = scale(testSet[, 2:3])
