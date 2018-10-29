#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#username : mzr3
"""
Created on Fri Mar 31 12:02:42 2017

@author: melisandezonta
"""

from sklearn import  datasets, metrics, decomposition, random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  as LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, ShuffleSplit,GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from functions import *

breast_cancer = pd.read_csv('./breast-cancer-wisconsin.csv') 
breast_cancer = pd.DataFrame(breast_cancer.values, columns = list(breast_cancer))
data = breast_cancer.values
X, y = data[:,0:-1], data[:,-1]

dim = X.shape[1]
np.set_printoptions(precision=dim)

mean_vectors = []
for cl in range(0,2):
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))


S_W = np.zeros((dim,dim))
for cl,mv in zip(range(1,dim), mean_vectors):
    class_sc_mat = np.zeros((dim,dim))                  # scatter matrix for every class
    for row in X[y == cl]:
        row, mv = row.reshape(dim,1), mv.reshape(dim,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat 
    
overall_mean = np.mean(X, axis=0)

S_B = np.zeros((dim,dim))
for i,mean_vec in enumerate(mean_vectors):  
    n = X[y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(dim,1) # make column vector
    overall_mean = overall_mean.reshape(dim,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('between-class Scatter Matrix:\n', S_B)

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(dim,1)   
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])

print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

W = np.hstack((eig_pairs[0][1].reshape(dim,1), eig_pairs[1][1].reshape(dim,1), eig_pairs[2][1].reshape(dim,1)))
print('Matrix W:\n', W.real)

X_lda = X.dot(W)
assert X_lda.shape == (X.shape[0],3) 


label_dict = (0,1)
def plot_step_lda():
    with plt.style.context('seaborn-whitegrid'):

        fig = plt.figure(figsize=(10, 10))
        plt.scatter(np.random.randn(len(X_lda[y==0,0])),X_lda[y==0,0],c='r',label = '0')
        plt.scatter(np.random.randn(len(X_lda[y==1,0])),X_lda[y==1,0],c='b', label = '1')
        plt.grid()
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/lda_3D.png')

plot_step_lda()

cross_validation = ShuffleSplit(n_splits=50, test_size=0.33, random_state=0)

estimator = LinearSVC()
dim_reductor = LinearDiscriminantAnalysis()

title = "Data reconstruction precision for Lindear Discriminant Analysis"


plot_clustering_evaluation(dim_reductor, estimator, title, X, y, ylim=None, cv=None, rca_iter=10, n_jobs=-1)
plt.ylim(0.900,1)
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/lda_accuracy.png')



