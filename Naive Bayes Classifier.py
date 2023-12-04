# -*- coding: utf-8 -*-
# =============================================================================
# Faridat Lawal

# The Breast Cancer Wisconsin Dataset from the UCI machine learning respository 
# is a study of classification of 568 patients that are labelled as either (M)
# Malignant of (B)Benign based on 30 attributes/features computed from a digi-
# tized image of a fine needle aspirate of a breast mass.

# The Naive Bayes Classifier is a supervised learning algorithm that will be used
# to classify & test the performance of this model. 70% of the data will be rese-
# rved for training and the remaining 30% will be used for testing.

# =============================================================================

"""
Libraries
"""
import pandas as pd # save package name under variable name for easier access/usage
from sklearn.model_selection import train_test_split # split data into train and test data
from sklearn.naive_bayes import GaussianNB # import package/library to conduct naive bayes classifier
from sklearn.metrics import confusion_matrix # compute confusion matrix for naive bayes classifier model


"""
Dataset: read in csv and become familiar with dataset 
"""
wdbc = pd.read_csv("/Users/faridatl/Downloads/wdbc2.csv", header=None) # read in dataset from file on computer and save to a variable name for access in python
wdbc.head() # returns first 5 entries of dataset
wdbc.tail() # returns last 5 entries of dataset
wdbc.shape # returns dimensions of dataset i.e number of rows and columns present
wdbc.isnull().sum()  # checks for missing variables in data set


""" Split dataset into X and Y for training and testing """

X = wdbc.iloc[:, 2:32] # creates a dataframe of all features needed for training and testing in the wdbc dataset this will only include columns 2-31
y = wdbc[1] # creates a dataframe for target variable/cases 


""" Train and test dataset"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=35) #split X and y datasets into training and testing, where 30% of the data is reserved for testing 


""" Check for null values in all training and testing dataframes"""

X_train.isnull().sum() # returns number of missing/null values in X train
X_test.isnull().sum() # returns number of missing/null values in X test
y_train.isnull().sum() # returns number of missing/null values in y train
y_test.isnull().sum() # returns number of missing/null values in y test


""" Gaussian Naive Bayes Classifier: good for making predictions on normally distributed data """

gnb = GaussianNB() # save package name under variable name for easier access/usage

gnb.fit(X_train, y_train) #fit the x and y training sets to the gaussian naive bayes classifier

y_pred = gnb.predict(X_test) # predict the y target varaible/value from the provided X testing set 

""" Confusion Matrix"""

cm = confusion_matrix(y_test, y_pred) # creates a confusion matrix for the gaussian naive bayes classifier model based on the y predictions and y testing dataset
print("Confusion Matrix\n\n", cm) # prints the output of this confusion matrix 

""" Sensitivity TP/TP+FN"""
TP = cm[0,0] # true positive value from confusion matrix for access/usage
FP = cm[0,1] # false positive value from confusion matrix for access/usage
TN = cm[1,1] # true negative value from confusion matrix for access/usage
FN = cm[1,0] # false negative value from confusion matrix for access/usage

print(TP / (TP +FN)) # calculates and prints the sensitivity rate for model performance based on values from confusion matrix


""" Specificity TN/TN+FP"""
print(TN / (TN + FP)) # calculates and prints the specificity rate for model performance based on values from confusion matrix


""" Calculate overall accuracy"""
print((TP + TN) / float(TP + TN + FP + FN)) # calculates and prints the overall accuracy of model performance based on values from confusion matrix
