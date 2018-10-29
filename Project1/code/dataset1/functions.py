#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 10:41:42 2017

@author: melisandezonta
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve, learning_curve, GridSearchCV



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 15)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_validation_curve(estimator, title, xlabel, ylabel,X, y,param_name, ylim=None, 
                          cv=None,n_jobs=1, param_range = np.linspace(1, 1, 10)):
   
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name= param_name, param_range=param_range,
        cv=cv, scoring= "accuracy", n_jobs = n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if ylim is not None:
        plt.ylim(*ylim)
    lw = 2
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
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
