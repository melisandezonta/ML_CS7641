#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#username : mzr3
"""
Created on Thu Mar 30 00:28:29 2017

@author: melisandezonta
"""
from sklearn import datasets, metrics, decomposition
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, ShuffleSplit,GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_std = StandardScaler().fit_transform(X)

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(range(X.shape[1]), eig_vals, 'b.-')
ax2.plot(range(X.shape[1]), cum_var_exp, 'r')
ax1.set_xlabel('Principal components')
ax1.set_ylabel('Eigen Values', color='b')
ax2.set_ylabel('Cumulative Variability', color='r')
plt.title('Completeness Score vs. K Clusters')
plt.show()

plt.clf()

with plt.style.context('seaborn-whitegrid'):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(range(X.shape[1]), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    ax1.step(range(X.shape[1]), cum_var_exp, where='mid',
             label='cumulative explained variance')
    ax2.plot(range(X.shape[1]), eig_vals, 'r')
    ax2.set_ylabel('Eigen values', color = 'r')
    ax1.set_ylabel('Explained variance ratio', color = 'b')
    ax1.set_xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/digits/pca_eigen_values.png')

matrix_w = np.hstack((eig_pairs[0][1].reshape(X.shape[1],1), 
                      eig_pairs[1][1].reshape(X.shape[1],1),
                      eig_pairs[2][1].reshape(X.shape[1],1)))

print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)


with plt.style.context('seaborn-whitegrid'):
    #plt.figure(figsize=(6, 4))
    fig=plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    for lab, col in zip((np.arange(0,10)), 
                        ('blue', 'red','green','black','cyan','pink','purple','yellow','grey','orange')):

        ax.scatter(Y[y==lab, 0], Y[y==lab, 1], Y[y==lab, 2],label = lab,c =col)
        ax.set_xlabel('Principal Component 0')
        ax.set_ylabel('Principal Component 1')
        ax.set_zlabel('Principal Component 2')
        ax.view_init(20,300)
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/digits/pca_3D.png')
    plt.show()
    
cross_validation = ShuffleSplit(n_splits=50, test_size=0.33, random_state=0)

estimator = LinearSVC()
dim_reductor = decomposition.PCA()

title = "Data reconstruction precision for Principal Component Analysis"


plot_clustering_evaluation(dim_reductor, estimator, title, X, y, ylim=None, cv=None, rca_iter=10, n_jobs=-1)
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/digits/pca_accuracy.png')


        
        

