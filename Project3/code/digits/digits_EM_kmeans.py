#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#username : mzr3
"""
Created on Thu Mar 30 22:32:46 2017

@author: melisandezonta
"""

from sklearn import  datasets
import EM as em
import KMeansTestCluster as kmtc

import matplotlib.pyplot as plt



digits = datasets.load_digits()
X = digits.data
y = digits.target
clusters=range(2,20)



tester = em.ExpectationMaximizationTestCluster(X, y, clusters=range(2,20), plot=False, stats=True)
silhouette_EM,vmeasure_scores = tester.run()

tester = kmtc.KMeansTestCluster(X, y, clusters=range(2,20), plot=False, stats=True)
silhouette_kmeans,V_measure = tester.run()



"""
Plot Silhouette Score from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""
plt.plot(clusters, silhouette_kmeans,'r^-', label = "K Means")
plt.plot(clusters, silhouette_EM,'b^-', label = "EM")
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. K Clusters')
plt.legend(loc="best")
plt.xlim(2,20)
plt.text(9.5, 0.190, 'Digits', style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':2})
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/digits/silhouette_EM_Kmeans.png')
plt.show()  

"""
Plot Vmeasure from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""
plt.plot(clusters, V_measure, 'r^-',label = 'K Means')
plt.plot(clusters, vmeasure_scores, 'b^-',label = 'EM')
plt.xlabel('Number of clusters')
plt.ylabel('V Measure')
plt.title('V Measure vs. K Clusters')
plt.legend( loc = "best")
plt.xlim(2,20)
plt.text(9, 0.8, 'Digits', style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':2})
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/digits/V_measure_EM_Kmeans.png')
plt.show()