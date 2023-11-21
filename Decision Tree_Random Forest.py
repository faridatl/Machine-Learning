# -*- coding: utf-8 -*-
# =============================================================================
# Faridat Lawal

# The Ionosphere Dataset from the UCI machine learning respository consists of
# a phased array of 16 high-frequency antennas with a total transmitted power 
# on the order of 6.4 kilowatts. Received signals were processed using an 
# autocorrelation function whose arguments are the time of a pulse and the 
# pulse number. There were 17 pulse numbers for the system. Instances in this 
# database are described by 2 attributes per pulse number, corresponding to the
# complex values returned by the function resulting from the complex electroma-
# gnetic signal. "Good" (g) radar returns are those showing evidence of some 
# type of structure in the ionosphere. "Bad" (b) returns are those that do not; 
# their signals pass through the ionosphere.

# Both the Decition Tree and Random Forest Classifier willl be employed to
# evaluate the classification performance. Both classifiers will be compared to 
# determine which performed the best by recording the results of their Confusi-
# on matrix, sensitivity, specifity, total accuracy, F1-score, ROC and AUC scores.
# 5-Fold cross validation will also be employed to determine if the both models
# better.
# =============================================================================


"""
Libraries
"""
import pandas as pd # save package name under variable name for easier access/usage
from sklearn.model_selection import train_test_split # split data into train and test data
from sklearn.tree import DecisionTreeClassifier # import package/library to conduct decision tree classifier
from sklearn.ensemble import RandomForestClassifier # import package/library to conduct random forest classifier 
from sklearn.metrics import confusion_matrix # compute confusion matrix for k nearest neighbor classifier model
from sklearn.metrics import accuracy_score # will compute overall accuracy for model performance
from sklearn.metrics import RocCurveDisplay # imports package/library to plot ROC and compute AUC 
import matplotlib.pyplot as plt # imports and saves package name under variable name for easier acces/usage

"""
Dataset: read in csv and become familiar with dataset 
"""
ions = pd.read_csv("/Users/faridatlawal/DTSC710/Assignment3/ionosphere.csv", header=None) # read in dataset from file on computer and save to a variable name for access in python
ions.isnull().sum() # checks for missing variables in data set

""" Split dataset into X and Y for training and testing """
X = ions.iloc[:,:34] # creates a dataframe of all features needed for training and testing in the wine dataset this will only include 13 columns 
y = ions[34] # creates a dataframe for target variable/cases 


""" Train and test dataset"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state= 42) #split X and y datasets into training and testing, where 20% of the data is reserved for testing

""" Decision Tree Classifier: Using Entropy """
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0) # create decision tree variable name for easier access/usage with the criterion as entropy
dt_entropy.fit(X_train, y_train) # fit the x and y training sets to the entropy decision tree classifier
y_preds_entropy = dt_entropy.predict(X_test) # predict the y target varaible/value from the provided X testing set


""" Confusion Matrix"""
cm_entropy = confusion_matrix(y_test, y_preds_entropy) # creates a confusion matrix for the entropy decision tree classifier model based on the y predictions and y testing dataset
print("Confusion Matrix\n\n", cm_entropy) # prints the output of this confusion matrix 

""" Sensitivity TP/TP+FN"""
TP_dt = cm_entropy[0,0] # true positive value from confusion matrix for easy access/usage
FP_dt = cm_entropy[0,1] # false positive value from confusion matrix for easy access/usage
TN_dt = cm_entropy[1,1] # true negative value from confusion matrix for easy access/usage
FN_dt = cm_entropy[1,0] # false negative value from confusion matrix for easy access/usage

print(TP_dt / (TP_dt +FN_dt)) # calculates and prints the sensitivity rate for entropy decision tree classifier model performance based on values from confusion matrix

""" Specificity TN/TN+FP"""
print(TN_dt / (TN_dt + FP_dt)) # calculates and prints the specificity rate for entropy decision tree classifier model performance based on values from confusion matrix

""" Overall Accuracy"""
accuracy_score(y_test, y_preds_entropy) # calculates and prints the overall accuracy of the entropy decistion tree classifier model performance

""" F1 Score"""
precision = (TP_dt/(TP_dt+FP_dt)) # calculates and stores the precision value for decision tree classifier model
recall = (TN_dt / (TN_dt + FP_dt)) # calculates and stores the precision value for decision tree classifier model
print(2*(precision*recall)/(precision+recall)) # calculates and prints the f1 score for the decision tree classifier

""" Random Forest Classifier"""
rfc = RandomForestClassifier() # save package name under variable name for easier access/usage
rfc.fit(X_train, y_train) # fit the x and y training sets to the random forest classifier
y_preds_rfc = rfc.predict(X_test) # predicts the y target varaible/value from the provided X testing set and stores in variable name

""" Confusion Matrix"""
cm_rfc = confusion_matrix(y_test, y_preds_rfc) # creates a confusion matrix for the random forest classifier model based on the y predictions and y testing dataset
print("Confusion Matrix\n\n", cm_rfc) # prints the output of this confusion matrix 

""" Sensitivity TP/TP+FN"""
TP = cm_rfc[0,0] # true positive value from confusion matrix for easy access/usage
FP = cm_rfc[0,1] # false positive value from confusion matrix for easy access/usage
TN = cm_rfc[1,1] # true negative value from confusion matrix for easy access/usage
FN = cm_rfc[1,0] # false negative value from confusion matrix for easy access/usage

print(TP / (TP +FN)) # calculates and prints the sensitivity rate for random forest classifier model performance based on values from confusion matrix

""" Specificity TN/TN+FP"""
print(TN / (TN + FP)) # calculates and prints the specificity rate for random forest classifier model performance based on values from confusion matrix

""" Overall Accuracy"""
accuracy_score(y_test, y_preds_rfc) # calculates and prints the overall accuracy of the random forest classifier model performance

""" F1 Score"""
precision_rfc = (TP/(TP+FP)) # calculates and stores the precision value for decision tree classifier model
recall_rfc = (TN / (TN + FP)) # calculates and stores the precision value for decision tree classifier model
print(2*(precision_rfc*recall_rfc)/(precision_rfc+recall_rfc)) # calculates and prints the f1 score for the decision tree classifier

""" 5-fold Cross Validation"""
from sklearn.pipeline import make_pipeline # imports pipeline package to aid in cross validation
from sklearn.impute import SimpleImputer # imports imputer package to aid in creating 5 folds for cross validation 
from sklearn.model_selection import cross_val_score # imports package that will compute the requested cross validation scores 

""" Decision Tree Classsifier: Entropy"""
my_pipe_dt_entropy = make_pipeline(SimpleImputer(),dt_entropy) # creates a pipeline 
scores_dtf = cross_val_score(my_pipe_dt_entropy, X, y, scoring='accuracy') # computes and stores the accuracy cross validation scores for the entropy decision tree classifier 
print(scores_dtf) # prints the 5 accuracy cross validation scores for entropy decision tree classifier 

""" Random Forest Classifier"""
my_pipe_rfc = make_pipeline(SimpleImputer(),rfc) # creates a pipeline for random forest classifier validation
scores_rfcf1 = cross_val_score(my_pipe_rfc, X, y, scoring='accuracy') # computes and stores the accuracy cross validation scores for the random forest classifier 
print(scores_rfcf1) # prints the 5 accuracy cross validation scores for random forest classifier 

""" ROC & AUC"""
figdt, ax1 = plt.subplots() # create empty plot/graph
RocCurveDisplay.from_estimator(rfc, X_test, y_test, color="green", ax=ax1) # calculate AUC and create ROC for random forest classifier then add to graph 
RocCurveDisplay.from_estimator(dt_entropy, X_test, y_test, color="red", ax=ax1) # calculate AUC and create ROC for decision tree classifier then add to graph 
ax1.plot([0,1],[0,1], linestyle='--') #plot baseline
ax1.set_title('Decision Tree (Entropy) & Random Forest Classifier ROC') # add title to graph
ax1.set_ylabel('True Positive Rate') # add ylabel to graph 
ax1.set_xlabel('False Positive Rate') #add x label to graph
ax1.legend(loc='lower right') #add legend and set location to bottom right of graph
figdt # show the graph created 




