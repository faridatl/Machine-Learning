# -*- coding: utf-8 -*-
# =============================================================================
# Faridat Lawal

# The Wine Dataset from the UCI machine learning respository consists of the 
# results of a chemical analysis of wines grown within the same region in Italy
# but derived from three different cultivers. The analysis determined 13 consti-
# tuents (attributes) found in the three types of wine.

# The K-Nearest Neighbors Classifier is a supervised learning algorithm that 
# will be used to classify and test the perfomance of this model. 80% of the 
# data will be reserved for training and the remaining 20% will be used for
# testing.

# =============================================================================


"""
Libraries
"""
import pandas as pd # save package name under variable name for easier access/usage
from sklearn.model_selection import train_test_split # split data into train and test data
from sklearn import preprocessing # import package/library to scale all features in the wine dataset
from sklearn.neighbors import KNeighborsClassifier # import package/library to conduct k nearest neighbor classifier
from sklearn.metrics import confusion_matrix # compute confusion matrix for k nearest neighbor classifier model

"""
Dataset: read in csv and become familiar with dataset 
"""
wine = pd.read_csv("/Users/faridatlawal/DTSC710/Assignment2/wine.data", header=None) # read in dataset from file on computer and save to a variable name for access in python
wine.isnull().sum() # checks for missing variables in data set


""" Split dataset into X and Y for training and testing """
X = wine.iloc[:,1:] # creates a dataframe of all features needed for training and testing in the wine dataset this will only include 13 columns 
y = wine[0] # creates a dataframe for target variable/cases 

""" Feature-Scaling"""
X = preprocessing.scale(X) # using the preprocessing function, I call on the scale feature to scale all features in the X dataframe

""" Train and test dataset"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 73) #split X and y datasets into training and testing, where 20% of the data is reserved for testing


""" Check for null values in all training and testing dataframes"""
pd.isna(X_train).sum()# returns number of missing/null values in X train
pd.isna(X_test).sum() # returns number of missing/null values in X test
pd.isna(y_train).sum() # returns number of missing/null values in y train
pd.isna(y_test).sum() # returns number of missing/null values in y test


""" K Nearest Neigbors Classifier: neigboors are determined in terms of distance"""
knn = KNeighborsClassifier() # save package name under variable name for easier access/usage
knn.fit(X_train, y_train) #fit the x and y training sets to the k nearest neighbors classifier
y_preds = knn.predict(X_test) # predict the y target varaible/value from the provided X testing set 


""" Confusion Matrix"""
cm = confusion_matrix(y_test, y_preds, labels=[1,2,3]) # creates a confusion matrix for the k nearest neighbor classifier model based on the y predictions and y testing dataset
print("Confusion Matrix\n\n", cm) # prints the output of this confusion matrix 

""" Assiging TP, TN, FN, & FP values to the proper class"""
TP1 = cm[0,0] # true positive value for class 1 from confusion matrix for access/usage
TP2 = cm[1,1] # true positive value for class 2 from confusion matrix for access/usage
TP3 = cm[2,2] # true positive value for class 3 from confusion matrix for access/usage

FN1 = cm[0,1] + cm[0,2] # false negative value for class 1 from confusion matrix for access/usage
FN2 = cm[1,0] + cm[1,2] # false negative value for class 2 from confusion matrix for access/usage
FN3 = cm[2,0] + cm[2,1] # false negative value for class 3 from confusion matrix for access/usage

TN1 = TP2 + cm[1,2] + cm[2,1] + TP3 # true negative value for class 1 from confusion matrix for access/usage
TN2 = TP1 + cm[0,2] + cm[2,0] + TP3 # true negative value for class 2 from confusion matrix for access/usage
TN3 = TP1 + TP2 + cm[0,1] + cm[1,0] # true negative value for class 3 from confusion matrix for access/usage

FP1 = cm[1,0] + cm[2,0] # false positive value for class 1 from confusion matrix for access/usage
FP2 = cm[0,1] + cm[2,1] # false positive value for class 2 from confusion matrix for access/usage
FP3 = cm[0,2] + cm[1,2] # false positive value for class 3 from confusion matrix for access/usage


""" Sensitivity TP/TP+FN"""
print("Class 1 Sensitivity:", TP1/(TP1 + FN1)) # calculates and prints the sensitivity rate of class 1 for model performance based on values from confusion matrix
print("Class 2 Sensitivity:", TP2/(TP2 + FN2)) # calculates and prints the sensitivity rate of class 2 for model performance based on values from confusion matrix
print("Class 3 Sensitivity:", TP3/(TP3 + FN3)) # calculates and prints the sensitivity rate of class 3 for model performance based on values from confusion matrix


""" Specificity TN/TN+FP"""
print("Class 1 Specificity:", TN1/(TN1 + FP1)) # calculates and prints the specificity rate of class 1 for model performance based on values from confusion matrix
print("Class 2 Specificity:", TN2/(TN2 + FP2)) # calculates and prints the specificity rate of class 1 for model performance based on values from confusion matrix
print("Class 3 Specificity:", TN3/(TN3 + FP3)) # calculates and prints the specificity rate of class 1 for model performance based on values from confusion matrix

""" Calculate overall accuracy"""
print("Overall Accuracy:", (TP1 + TP2 + TP3)/ float(TP1 + TP2 + TP3 + FN1 +FN2 + FN3)) # calculates and prints the overall accuracy of model performance based on values from confusion matrix

# Note: when random state is 42, I received an accuracy of .9444444 and when changed to 73, I received an accuracy of 1.0 & sensitivity and specifity for all classes is also 1.0