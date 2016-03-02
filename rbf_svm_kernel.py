#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
clf=svm.SVC(kernel='rbf', C=10000.0)
clf.fit(features_train, labels_train)
pred=clf.predict(features_test)
#print pred[10]
#print pred[26]
#print pred[50]
#finding number of points in Chris class
count=0
for i in pred:
    if pred[i]==1:
        count=count+1
print count
print(accuracy_score(labels_test, pred))








