#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:44:32 2017

@author: melisandezonta
"""

import numpy as np                                                     
import pandas as pd                                                    
import matplotlib.pyplot as plt  
import seaborn as sb                                                                              
from sklearn import model_selection                                             
from sklearn.ensemble import AdaBoostClassifier
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

classifier = DecisionTreeClassifier(max_depth = 8)
# Create the classifier
ADB = AdaBoostClassifier(base_estimator= classifier, n_estimators=50, 
                         learning_rate=1.0, algorithm='SAMME.R', random_state=None)

# Train the classifier on the training set
ADB.fit(X_train,Y_train)

# Validate the classifier on the training set using classification accuracy
score_training = ADB.score(X_train, Y_train,sample_weight=None)
print('The training score is : ',score_training)

# Validate the classifier on the testing set using classification accuracy
score_test = ADB.score(X_validation, Y_validation,sample_weight=None)
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
    
    ADB = AdaBoostClassifier()
    ADB.fit(training_inputs, training_classes)
    classifier_accuracy = ADB.score(testing_inputs, testing_classes)
    model_accuracies.append(classifier_accuracy)
    
sb.distplot(model_accuracies)

# In[3]:
# Apply cross validation
ADB = AdaBoostClassifier(base_estimator = classifier)
plt.figure()
cv_scores = cross_val_score(ADB, arr_in, arr_out, cv=3)
sb.distplot(cv_scores)
plt.title('Average score: {} and Std score : {}'.format(np.mean(cv_scores),np.std(cv_scores)))

# In[4]:
#Tune the parameters to best fit to the training data
N_E = 200
N_LR = 5
ADB = AdaBoostClassifier(base_estimator = classifier)
parameter_grid = {'n_estimators' : np.arange(1,N_E+20,20),
                  'learning_rate' : np.linspace(0.1,2,N_LR)}

cross_validation = StratifiedKFold(arr_out, n_folds=3)

grid_search = GridSearchCV(ADB,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(arr_in, arr_out)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# In[5]:
#Visualisation of the grid over the tuning parameters
learning_rate = np.linspace(0.1,2,N_LR)
n_estimators = np.arange(1,N_E+20,20)
plt.figure() 
grid_visualization = []
grid_visualization.append(grid_search.cv_results_['mean_test_score'])
grid_visualization = np.array(grid_visualization)
grid_visualization.shape = (len(learning_rate), len(n_estimators))

plt.figure(figsize=(len(n_estimators), len(learning_rate)))
sb.heatmap(grid_visualization, cmap='Blues')
plt.xticks(np.arange(len(n_estimators)) + 0.5, grid_search.param_grid['n_estimators'])
plt.yticks(np.arange(len(learning_rate)) + 0.5, grid_search.param_grid['learning_rate'][::-1])
plt.xlabel('n_estimators')
plt.ylabel('learning_rate')

# In[6]:
#Draw learning curve 
X, y = arr_in, arr_out
title1 = "Learning Curves (Adaboost)"
cv = StratifiedKFold(arr_out, n_folds=3)
ADB = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                               n_estimator = grid_search.best_params['n_estimators'],
                               learning_rate=grid_search.best_params_['learning_rate'],
                               random_state=0)
plot_learning_curve(ADB, title1, X, y, ylim=(0.4, 1.01), cv=cv, n_jobs=-1)

# In[7]:
#Draw validation curve 
title2 = "Validation Curve with Adaboost "
xlabel = "n_estimators"
ylabel = "Score"
ADB = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                               learning_rate=grid_search.best_params_['learning_rate'],
                               random_state=0)
plot_validation_curve(ADB, title2, xlabel, ylabel,X, y,param_name = 'n_estimators', ylim=None, 
                          cv=cv,n_jobs=-1, param_range = np.arange(1,200,20))

plt.show()
