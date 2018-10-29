#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 20:08:10 2017

@author: melisandezonta
"""


import numpy as np                                                     
import pandas as pd                                                    
import matplotlib.pyplot as plt  
import seaborn as sb                                                                              
from sklearn import model_selection                                             
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, ShuffleSplit,GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.datasets import load_digits
from functions import *
from dataset2 import *


# In[1]:

#Splitting datas
validation_size = 0.33
seed = 20
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(arr_in,arr_out, test_size = validation_size, random_state =seed)


# Create the classifier
svm = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', 
          coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
          class_weight=None, verbose=False, max_iter=5000, decision_function_shape=None, 
          random_state=None)

# Train the classifier on the training set
svm.fit(X_train,Y_train)

# Validate the classifier on the training set using classification accuracy
score_training = svm.score(X_train, Y_train,sample_weight=None)
print('The training score is : ',score_training)

# Validate the classifier on the testing set using classification accuracy
score_test = svm.score(X_validation, Y_validation,sample_weight=None)
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
    
    svm = SVC(max_iter = 20000)
    svm.fit(training_inputs, training_classes)
    classifier_accuracy = svm.score(testing_inputs, testing_classes)
    model_accuracies.append(classifier_accuracy)
    
sb.distplot(model_accuracies)
# In[3]:
# Apply cross validation
svm = SVC(max_iter = 20000)
plt.figure()
cv_scores = cross_val_score(svm, arr_in, arr_out, cv=3)
sb.distplot(cv_scores)
plt.title('Average score: {} and Std score : {}'.format(np.mean(cv_scores),np.std(cv_scores)))

# In[4]:
#Tune the parameters to best fit to the training data
N_degree = 6
svm = SVC(max_iter = 20000)

parameter_grid = {'kernel': ['linear', 'sigmoid', 'poly', 'rbf'],
                  'degree' : np.arange(1,N_degree+1)}

cross_validation = StratifiedKFold(arr_out, n_folds=3)

grid_search = GridSearchCV(svm,
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
grid_visualization.shape = (N_degree, 4)

plt.figure(figsize=(N_degree, 4))
sb.heatmap(grid_visualization, cmap='Blues')
plt.xticks(np.arange(4) + 0.5, grid_search.param_grid['kernel'])
plt.yticks(np.arange(N_degree+1) + 0.5, grid_search.param_grid['degree'][::-1])
plt.xlabel('kernel')
plt.ylabel('degree')

# In[6]:
#Draw learning curve   
X, y = arr_in, arr_out
title1 = "Learning Curves (support Vector Machine)"
cv = StratifiedKFold(arr_out, n_folds=3)
estimator = SVC(kernel = 'poly',degree = 1,max_iter = 20000)
plot_iterative_learning_curve(estimator, title1, X, y, ylim=None, cv= cv, n_jobs=-1,
                              iterations=np.arange(1, 100000, 10000), exploit_incremental_learning=True)

# In[7]:
#Draw validation curve  
title2 = "Validation Curve with Support Vector Machine "
xlabel = 'degree'
ylabel = "Score"
plot_validation_curve(estimator, title2, xlabel, ylabel,X, y,param_name = 'degree', ylim=None, 
                          cv=cv,n_jobs=-1, param_range = np.arange(1,5,1))
plt.show()



