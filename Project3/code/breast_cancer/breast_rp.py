#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#username : mzr3
"""
Created on Fri Mar 31 23:38:09 2017

@author: melisandezonta
"""
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn import  datasets, metrics, decomposition, random_projection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, ShuffleSplit,GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functions import *
from sklearn.random_projection import johnson_lindenstrauss_min_dim

breast_cancer = pd.read_csv("./breast-cancer-wisconsin.csv")
breast_cancer = pd.DataFrame(breast_cancer.values, columns = list(breast_cancer))
data = breast_cancer.values
X, y = data[:,0:-1], data[:,-1]

johnson_lindenstrauss_min_dim(1797,eps=0.1)

accuracies = []
components = range(2,X.shape[1])

split = train_test_split(X, y, test_size = 0.3,
    random_state = 42)
#digits = datasets.load_digits()
#split = train_test_split(digits.data, digits.target, test_size = 0.3,
#    random_state = 42)
(trainData, testData, trainTarget, testTarget) = split

model = LinearSVC()
model.fit(trainData, trainTarget)
baseline = metrics.accuracy_score(model.predict(testData), testTarget)

# loop over the projection sizes
for comp in components:
    # create the random projection
    sp = SparseRandomProjection(n_components = comp)
    X_new = sp.fit_transform(trainData)
 
    # train a classifier on the sparse random projection
    model = LinearSVC()
    model.fit(X_new, trainTarget)
 
    # evaluate the model and update the list of accuracies
    test = sp.transform(testData)
    accuracies.append(metrics.accuracy_score(model.predict(test), testTarget))
    
# create the figure
plt.figure()
plt.suptitle("Accuracy of Sparse Projection on Digits")
plt.xlabel("# of Components")
plt.ylabel("Accuracy")
plt.ylim([0, 1.0])
 
# plot the baseline and random projection accuracies
plt.plot(components, [baseline] * len(accuracies), color = "r")
plt.plot(components, accuracies)

plt.show()

indices = []

for i in range(X.shape[1]-2) :

    if accuracies[i] >= baseline:
        
        indices.append(i)

sp = SparseRandomProjection(n_components = 3)
output = sp.fit_transform(X)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    fig=plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    for lab, col in zip((0,1), 
                        ('blue', 'red')):
        ax.scatter(output[y==lab, 0],output[y==lab, 1],output[y==lab, 2],label = lab,c =col)
        ax.view_init(20,-120)
        ax.set_xlabel('Component 0')
        ax.set_ylabel('Component 1')
        ax.set_zlabel('Component 2')
        plt.legend(loc='lower right')
        plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/rp_3D.png')
        plt.tight_layout()
    plt.show()
    plt.clf()
    
cross_validation = ShuffleSplit(n_splits=50, test_size=0.33, random_state=0)

estimator = LinearSVC()
dim_reductor = SparseRandomProjection()

title = "Data reconstruction precision for Randomized Projection"


plot_clustering_evaluation(dim_reductor, estimator, title, X, y, ylim=None, cv=None, rca_iter=50, n_jobs=-1)
plt.ylim(0.500,0.920)
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/rp_accuracy.png')

