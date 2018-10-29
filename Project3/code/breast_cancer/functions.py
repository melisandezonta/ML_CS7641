#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#username : mzr3
"""
Created on Sun Apr  2 12:02:49 2017

@author: melisandezonta
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve, learning_curve, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def plot_clustering_evaluation(dim_reductor, estimator, title, X, y, ylim=None, cv=None, rca_iter=10, n_jobs=-1):

    baseline = np.mean(cross_val_score(estimator, X, y, cv=cv, n_jobs=n_jobs))
    accuracies = []

    # loop over the projection sizes
    components = range(2,X.shape[1])
    for comp in components:
        cv_scores = []
        if type(dim_reductor) is GaussianRandomProjection:
            for i in range(0,10):
                # reduce the dimensionality of the data 
                dim_reductor.set_params(n_components = comp)
                X_new = dim_reductor.fit_transform(X)
 
                # train a classifier on the dimensionality reduced data
                cv_scores.append(cross_val_score(estimator, X_new, y, cv=cv, n_jobs=n_jobs))
        if type(dim_reductor) is LinearDiscriminantAnalysis:
            for i in range(0,10):
                # reduce the dimensionality of the data 
                dim_reductor.set_params(n_components = comp)
                X_new = dim_reductor.fit_transform(X,y)
         
                # train a classifier on the dimensionality reduced data
                cv_scores.append(cross_val_score(estimator, X_new, y, cv=cv, n_jobs=n_jobs))

        else:
            # reduce the dimensionality of the data 
            dim_reductor.set_params(n_components = comp)
            X_new = dim_reductor.fit_transform(X)
 
            # train a classifier on the dimensionality reduced data
            cv_scores.append(cross_val_score(estimator, X_new, y, cv=cv, n_jobs=n_jobs))
 
        # evaluate the model and update the list of accuracies
        accuracies.append(np.mean(cv_scores))
    
    # create the figure
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.xlabel("Number of Components")
    plt.ylabel("Accuracy")
    if ylim is not None:
        plt.ylim(*ylim)
    # plot the baseline and dimensionality reduced data accuracies
    plt.plot(components, [baseline] * len(accuracies), 'o-', label="Baseline score", color = "r")
    plt.plot(components, accuracies, 'o-', label="Dimensionality reduced data score", color="b")
    plt.legend(loc="best")
    return plt




def plot_iterative_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, iterations=np.arange(1, 206, 5), exploit_incremental_learning=False):
    
    plt.figure(figsize=(10, 10))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Iterations")
    plt.ylabel("Score")

    parameter_grid = {'max_iter': iterations}
    grid_search = GridSearchCV(estimator, param_grid=parameter_grid, n_jobs=-1, cv=cv)
    grid_search.fit(X, y)

    train_scores_mean = grid_search.cv_results_['mean_train_score']
    train_scores_std = grid_search.cv_results_['std_train_score']
    test_scores_mean = grid_search.cv_results_['mean_test_score']
    test_scores_std = grid_search.cv_results_['std_test_score']
    plt.grid()

    plt.fill_between(iterations, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(iterations, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(iterations, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(iterations, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
