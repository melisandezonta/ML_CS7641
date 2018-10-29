#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# username : mzr3
"""
Created on Fri Mar 31 15:34:45 2017

@author: melisandezonta
"""

from sklearn import  datasets, metrics, decomposition, random_projection
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, ShuffleSplit,GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from functions import *

breast_cancer = pd.read_csv('./breast-cancer-wisconsin.csv') 
breast_cancer = pd.DataFrame(breast_cancer.values, columns = list(breast_cancer))
breast_cancer = pd.DataFrame(breast_cancer.values, columns = list(breast_cancer))
data = breast_cancer.values
X, y = data[:,0:-1], data[:,-1]

kurts = []
ica = decomposition.FastICA(n_components = None, whiten=True)
output =ica.fit_transform(X)
for i in range(0,output.shape[1]):
    kurt = kurtosis(output[:,i])
    kurts.append(kurt)
kurts_pairs = [(i, k) for i,k in enumerate(kurts)]
kurts_pairs.sort(key=lambda x: x[1])

"""
Plot Kurtosis for ICA
"""
fig=plt.figure()
plt.bar(*zip(*kurts_pairs), alpha=0.5, align='center',
            label='individual explained variance')
plt.xlabel('Dimension')
plt.ylabel('Kurtosis')
plt.title('Kurtosis vs Dimension')
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/ica_kurtosis.png')
plt.show()

min_indices = list(zip(*kurts_pairs))[0][:3]


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    fig=plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    for lab, col in zip((0,1), 
                        ('blue', 'red')):
        ax.scatter(output[y==lab, min_indices[0]],output[y==lab, min_indices[1]], list(output[y==lab, min_indices[2]])[::-1],label = lab,c =col)
        ax.view_init(20,-120)
        ax.set_xlabel('Component 0')
        ax.set_ylabel('Component 1')
        ax.set_zlabel('Component 2')
        plt.legend(loc='lower right')
        plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/ica_3D.png')
        plt.tight_layout()
    plt.show()
    plt.clf()
    
    
cross_validation = ShuffleSplit(n_splits=50, test_size=0.33, random_state=0)

estimator = LinearSVC()
dim_reductor = decomposition.FastICA()

title = "Data reconstruction precision for Independent Component Analysis"


plot_clustering_evaluation(dim_reductor, estimator, title, X, y, ylim=None, cv=None, rca_iter=10, n_jobs=-1)
plt.ylim(0.900,1)
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/ica_accuracy.png')



