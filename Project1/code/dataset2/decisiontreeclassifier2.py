#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 23:04:35 2017

@author: melisandezonta
"""
import numpy as np                                                     
import pandas as pd                                                    
import matplotlib.pyplot as plt  
import seaborn as sb                                                                              
from sklearn import model_selection, tree                                               
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, ShuffleSplit, GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from functions import plot_learning_curve, plot_validation_curve
from dataset2 import *

# In[1]:
#Splitting datas
validation_size = 0.33
seed = 20
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(arr_in,arr_out, test_size = validation_size, random_state =seed)            
                                                       

# Create the classifier
decision_tree_classifier = DecisionTreeClassifier()

# Train the classifier on the training set
decision_tree_classifier.fit(X_train, Y_train)

# Validate the classifier on the training set using classification accuracy
score_training = decision_tree_classifier.score(X_train, Y_train)
print('The training score is : ',score_training)

# Validate the classifier on the testing set using classification accuracy
score_test = decision_tree_classifier.score(X_validation, Y_validation)
print('The testing score is : ',score_test)


# In[2]:
# Test the accuracy score over 100 iterations to see if it's iterable
model_accuracies = []
plt.figure()
for repetition in range(100):
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = train_test_split(arr_in, arr_out, train_size=0.66)
    
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(training_inputs, training_classes)
    classifier_accuracy = decision_tree_classifier.score(testing_inputs, testing_classes)
    model_accuracies.append(classifier_accuracy)
    
sb.distplot(model_accuracies)


# In[3]:
# Apply cross validation
decision_tree_classifier = DecisionTreeClassifier()
plt.figure()
cv_scores = cross_val_score(decision_tree_classifier, arr_in, arr_out, cv=3)
sb.distplot(cv_scores)
plt.title('Average score: {} and Std score : {}'.format(np.mean(cv_scores),np.std(cv_scores)))


# In[4]:
#Tune the parameters to best fit to the training data
size = data.shape
max_d = 20
parameter_grid = {'max_depth': np.arange(1,max_d+1),
                  'min_samples_split' : np.arange(2,size[1]+1)}

cross_validation = StratifiedKFold(arr_out, n_folds=10)

grid_search = GridSearchCV(decision_tree_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(arr_in, arr_out)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# In[5]:

#Visualisation of the grid over the tuning parameters
grid_visualization = []
grid_visualization.append(grid_search.cv_results_['mean_test_score'])
grid_visualization = np.array(grid_visualization)
grid_visualization.shape = (max_d, size[1]-1)

plt.figure(figsize=((size[1]-1)/2, max_d/2))
sb.heatmap(grid_visualization, cmap='Blues')
plt.xticks(np.arange(size[1]-1) + 0.5, grid_search.param_grid['min_samples_split'])
plt.yticks(np.arange(max_d) + 0.5, grid_search.param_grid['max_depth'][::-1])
plt.xlabel('min_samples_split')
plt.ylabel('max_depth')

# In[6]:
#Draw learning curve  
X, y = arr_in, arr_out
title1 = "Learning Curves (Decision Tree Classifier)"
cross_validation = StratifiedKFold(arr_out, n_folds=3)
estimator = DecisionTreeClassifier()
plot_learning_curve(estimator, title1, X, y, ylim=(0.4, 1.01), cv = None, n_jobs=4)

# In[7]
#Draw validation curve
title2 = "Validation Curve with Decision Tree Classifier "
xlabel = "max_depth"
ylabel = "Score"
plot_validation_curve(estimator, title2, xlabel, ylabel,X, y,param_name = 'max_depth', ylim=None, 
                          cv = cross_validation,n_jobs=1, param_range = np.arange(1,max_d))

plt.show()




