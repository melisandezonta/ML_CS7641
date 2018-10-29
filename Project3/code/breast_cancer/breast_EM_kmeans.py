#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# username : mzr3
"""
Created on Thu Mar 30 22:32:46 2017

@author: melisandezonta
"""

from sklearn import  datasets, metrics
import EM as em
import KMeansTestCluster as kmtc
import pandas as pd
import matplotlib.pyplot as plt


breast_cancer = pd.read_csv('./breast-cancer-wisconsin.csv') 
li=list(breast_cancer)
breast_cancer = pd.DataFrame(breast_cancer.values, columns = li)

Class=li[-1]


arr = breast_cancer.values                                                      
y = arr[:,-1]     
X= arr[:,0:-1]
clusters=range(2,15)



tester = em.ExpectationMaximizationTestCluster(X, y, clusters=range(2,15), plot=False, stats=True)
silhouette_EM,vmeasure_scores = tester.run()

tester = kmtc.KMeansTestCluster(X, y, clusters=range(2,15), plot=False, stats=True)
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
plt.legend(bbox_to_anchor=(0.76, 0.99), loc=2, borderaxespad=0.)
plt.text(6, 0.59, 'Breast Cancer', style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':2})
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/silhouette_EM_Kmeans.png')
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
plt.legend(bbox_to_anchor=(0.76, 0.99), loc=2, borderaxespad=0.)
plt.text(6, 0.75, 'Breast Cancer', style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':2})
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/V_measure_EM_Kmeans.png')
plt.show()