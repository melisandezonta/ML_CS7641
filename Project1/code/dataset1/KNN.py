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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, ShuffleSplit, GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.datasets import load_digits
from functions import plot_learning_curve, plot_validation_curve
from dataset1 import *


# In[1]:
#Splitting datas 
validation_size = 0.33
seed = 20
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(arr_in,arr_out, test_size = validation_size, random_state =seed)


# Create the classifier
KNN = KNeighborsClassifier(n_neighbors=5, weights='distance', 
                           algorithm='auto', leaf_size=30, p=2, 
                           metric='minkowski', metric_params=None, n_jobs=1)

# Train the classifier on the training set
KNN.fit(X_train,Y_train)

# Validate the classifier on the training set using classification accuracy
score_training = KNN.score(X_train, Y_train,sample_weight=None)
print('The training score is : ',score_training)

# Validate the classifier on the testing set using classification accuracy
score_test = KNN.score(X_validation, Y_validation,sample_weight=None)
print('The testing score is : ',score_test)


# In[2]:
# Test the accuracy score over 100 iterations to see if it's iterable
model_accuracies = []
plt.figure()
for repetition in range(1000):
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = train_test_split(arr_in, arr_out, train_size=0.66)
    
    KNN = KNeighborsClassifier()
    KNN.fit(training_inputs, training_classes)
    classifier_accuracy = KNN.score(testing_inputs, testing_classes)
    model_accuracies.append(classifier_accuracy)
    
sb.distplot(model_accuracies)
# In[3]:
# Apply cross validation
KNN = KNeighborsClassifier()
plt.figure()
cv_scores = cross_val_score(KNN, arr_in, arr_out, cv=10)
sb.distplot(cv_scores)
plt.title('Average score: {} and Std score : {}'.format(np.mean(cv_scores),np.std(cv_scores)))

# In[4]:
#Tune the parameters to best fit to the training data
max_neighbors = 15
KNN = KNeighborsClassifier(weights = 'distance')
parameter_grid = {'n_neighbors' : np.arange(1,max_neighbors+1,1),
                  'weights' : ['uniform','distance']}

cross_validation = StratifiedKFold(arr_out, n_folds=10)

grid_search = GridSearchCV(KNN,
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
grid_visualization.shape = (max_neighbors, 2)

plt.figure(figsize=(2, max_neighbors))
sb.heatmap(grid_visualization, cmap='Blues')
plt.xticks(np.arange(2) + 0.5, grid_search.param_grid['weights'])
plt.yticks(np.arange(max_neighbors) + 0.5, grid_search.param_grid['n_neighbors'][::-1])
plt.xlabel('weights')
plt.ylabel('n_neighbors')

# In[6]:
# Draw the learning curve
X, y = arr_in, arr_out
title1 = "Learning Curve (K Nearest Neighbors)"
cv = ShuffleSplit(n_splits=100, test_size=0.33, random_state=0)
estimator = KNeighborsClassifier()
plot_learning_curve(estimator, title1, X, y, ylim=None, cv=cv, n_jobs=4)

# In[7]:
#Draw the validation curve
title2 = "Validation Curve with K Nearest Neighbors "
xlabel = "n_neighbors"
ylabel = "Score"
plot_validation_curve(estimator, title2, xlabel, ylabel,X, y,param_name = 'n_neighbors', ylim=None, 
                          cv=cv,n_jobs=1, param_range = np.arange(1,10,1))

plt.show()


