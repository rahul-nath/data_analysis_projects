"""
- grid search is meant to optimize the model -- reduce bias and normalize variance
- it does this by selecting the optimal parameter combination for the model
- it generates the optimal parameter combination by creating a grid of different parameter combinations
- it then tries to fit each combination to the classifier, selecting whichever one is most optimal


more on GridSearch: http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html
"""

from sklearn import grid_search, datasets, svm

iris = datasets.load_iris()
parameters = {"kernel":("linear", "rbf"), "C":[1,10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)

"""
In this and the following exercises, you'll be adding train test splits to the data
to see how it changes the performance of each classifier

The code provided will load the Titanic dataset like you did in project 0, then train
a decision tree (the method you used in your project) and a Bayesian classifier (as
discussed in the introduction videos). You don't need to worry about how these work for
now. 

What you do need to do is import a train/test split, train the classifiers on the
training data, and store the resulting accuracy scores in the dictionary provided.
"""

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')
# Limit to numeric data
X = X._get_numeric_data()
# Separate the labels
y = X['Survived']
# Remove labels from the inputs, and age due to missing data
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

# The decision tree classifier
clf1 = DecisionTreeClassifier()
clf1.fit(features_train, labels_train)
decision_tree_score = accuracy_score(clf1.predict(features_test),labels_test)
print "Decision Tree has accuracy: ", decision_tree_score
# The naive Bayes classifier

clf2 = GaussianNB()
clf2.fit(features_train,labels_train)
naive_bayes_score = accuracy_score(clf2.predict(features_test),labels_test)
print "GaussianNB has accuracy: ", naive_bayes_score

answer = { 
 "Naive Bayes Score": naive_bayes_score,
 "Decision Tree Score": decision_tree_score
}

"""
In this exercise, we'll use the Titanic dataset as before, train two classifiers and
look at their confusion matrices. Your job is to create a train/test split in the data
and report the results in the dictionary at the bottom.
"""

# Load the dataset
from sklearn import datasets

X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
conf_mat_dt = confusion_matrix(clf.predict(features_test),labels_test)
print "Confusion matrix for this Decision Tree:\n", conf_mat_dt

clf1 = GaussianNB()
clf1.fit(features_train,labels_train)
conf_mat_nb = confusion_matrix(clf1.predict(features_test),labels_test)
print "GaussianNB confusion matrix:\n", conf_mat_nb

#TODO: store the confusion matrices on the test sets below

confusions = {
 "Naive Bayes": conf_mat_nb,
 "Decision Tree": conf_mat_dt
}
