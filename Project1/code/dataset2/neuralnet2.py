#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:37:03 2017

@author: melisandezonta
"""
import numpy as np                                                     
import pandas as pd                                                    
import matplotlib.pyplot as plt  
import seaborn as sb                                                                              
from sklearn import model_selection                                             
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, ShuffleSplit,GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.datasets import load_digits
from functions import *
from dataset1 import *



# In[1]:
#Splitting datas
validation_size = 0.33
seed = 20
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(arr_in,arr_out, test_size = validation_size, random_state =seed)


# Create the classifier
MLP = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, 
                    batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
                    power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=0.0001, 
                    verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# Train the classifier on the training set
MLP.fit(X_train,Y_train)

# Validate the classifier on the training set using classification accuracy
score_training = MLP.score(X_train, Y_train,sample_weight=None)
print('The training score is : ',score_training)

# Validate the classifier on the testing set using classification accuracy
score_test = MLP.score(X_validation, Y_validation,sample_weight=None)
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
    
    MLP = MLPClassifier()
    MLP.fit(training_inputs, training_classes)
    classifier_accuracy = MLP.score(testing_inputs, testing_classes)
    model_accuracies.append(classifier_accuracy)
    
sb.distplot(model_accuracies)

# In[3]:
# Apply cross validation
MLP = MLPClassifier()
plt.figure()
# cross_val_score returns a list of the scores, which we can visualize
# to get a reasonable estimate of our classifier's performance
cv_scores = cross_val_score(MLP, arr_in, arr_out, cv=3)
sb.distplot(cv_scores)
plt.title('Average score: {} and Std score : {}'.format(np.mean(cv_scores),np.std(cv_scores)))

# In[4]:
#Tune the parameters to best fit to the training data

#array_layer_1 = [(x,) for x in range(1,50,10)]
#array_layer_2 = [(x,x+10) for x in range(1,50,10)]
#array_layer_3 = [(x,x+10,x+20) for x in range(1,50,10)]
#array_layer = array_layer_1 + array_layer_2 + array_layer_3
array_layer = [(100,),(100,100),(100,100,100)]
size_layer = len(array_layer)
MLP = MLPClassifier()
parameter_grid = {'hidden_layer_sizes': array_layer,
                  'learning_rate' : ['constant','invscaling']}

cross_validation = StratifiedKFold(arr_out, n_folds=3)

grid_search = GridSearchCV(MLP,
                           param_grid=parameter_grid,
                           cv=cross_validation,n_jobs=-1)

grid_search.fit(arr_in, arr_out)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# In[5]:
#Visualisation of the grid over the tuning parameters
grid_visualization = []
grid_visualization.append(grid_search.cv_results_['mean_test_score'])
grid_visualization = np.array(grid_visualization)
grid_visualization.shape = (size_layer,2)

plt.figure(figsize=(2,size_layer))
sb.heatmap(grid_visualization, cmap='Blues')
plt.xticks(np.arange(2) + 0.5, grid_search.param_grid['learning_rate'])
plt.yticks(np.arange(size_layer) + 0.5, grid_search.param_grid['hidden_layer_sizes'][::-1])
plt.xlabel('learning_rate')
plt.ylabel('hidden_layer_sizes')


# In[6]:
#Draw learning curve 
X, y = arr_in, arr_out
title = "Learning Curves (Neural Network)"
cv = StratifiedKFold(arr_out, n_folds=3)
estimator = MLPClassifier(hidden_layer_sizes = (100,100),max_iter = 5000, learning_rate = 'constant')
plot_iterative_learning_curve(estimator, title, X, y, ylim=None, cv= cv, n_jobs=-1,
                              iterations=np.arange(1, 100, 5), exploit_incremental_learning=True)



