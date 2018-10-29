#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#username : mzr3
"""
Created on Thu Mar 30 15:10:54 2017

@author: melisandezonta
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import mixture, metrics, decomposition
from scipy.spatial.distance import cdist



class ExpectationMaximizationTestCluster():
    def __init__(self, X, y, clusters, plot=False, stats=False):
        self.X = X
        self.y = y
        self.clusters = clusters
        self.gen_plot = plot
        self.stats = stats

    def run(self):
        vmeasure_scores=[]
        silhouette=[]
        log_likelihood=[]
        bic=[]

        for k in self.clusters:
                model = mixture.GaussianMixture(n_components=k, max_iter= 5000, n_init=50, init_params='kmeans')
                model.fit(self.X,self.y)
                labels = model.predict(self.X)
                vmeasure_scores.append(metrics.v_measure_score(self.y, labels))
                log_likelihood.append(model.score(self.X))
                bic.append(model.bic(self.X))
                silhouette.append(metrics.silhouette_score(self.X, labels , metric='euclidean',sample_size=self.X.shape[0]))
            
        if self.gen_plot:
            self.plot(vmeasure_scores, log_likelihood, silhouette, bic)
        else:
            return silhouette,vmeasure_scores

    def plot(self,vmeasure_scores, log_likelihood, silhouette, bic):
        

        
        """
        Plot Vmeasure from observations from the cluster centroid
        to use the Elbow Method to identify number of clusters to choose
        """
        plt.plot(self.clusters, vmeasure_scores, 'b.-', self.clusters, vmeasure_scores, 'k')
        plt.xlabel('Number of clusters')
        plt.ylabel('V Measure')
        plt.title('V Measure vs. K Clusters')
        plt.show()
        
        """
        Plot Silhouette Score from observations from the cluster centroid
        to use the Elbow Method to identify number of clusters to choose
        """

        plt.plot(self.clusters, silhouette, 'b.-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs. K Clusters')
        plt.show()
        
        """
        Plot Log Likelihood Score from observations from the cluster centroid
        to use the Elbow Method to identify number of clusters to choose
        """
        plt.plot(self.clusters, log_likelihood, 'b.-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Log Likelihood')
        plt.title('Log Likelihood vs. K Clusters')
        plt.show()
        
        """
        Plot BIC Score from observations from the cluster centroid
        to use the Elbow Method to identify number of clusters to choose
        """
        plt.plot(self.clusters, bic, 'b.-')
        plt.xlabel('Number of clusters')
        plt.ylabel('BIC Score')
        plt.title('BIC Score vs. K Clusters')
        plt.show()
        
        """
        Plot Log Likelihood and BIC Score from observations from the cluster centroid
        to use the Elbow Method to identify number of clusters to choose
        """
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(self.clusters, bic, 'b.-')
        ax1.plot(self.clusters, log_likelihood, 'r.-')
        ax1.set_xlabel('Number of clusters')
        ax2.set_ylabel('BIC Score', color = 'b')
        ax1.plot(self.clusters, log_likelihood, 'r.-')
        ax1.set_ylabel('Log Likelihood', color = 'r')
        plt.title('Log Likelihood and BIC Score vs. K Clusters')
        plt.grid(True)
        plt.xlim(2,20)
        plt.text(9, 260000, 'Digits', style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':2})
        plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/digits/log_likelihood_BIC.png')
        plt.show()
 