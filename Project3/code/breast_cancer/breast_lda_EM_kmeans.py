#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#username : mzr3
"""
Created on Sat Apr  1 01:25:23 2017

@author: melisandezonta
"""

from sklearn import  datasets, mixture, decomposition, discriminant_analysis
import KMeansTestCluster as kmtc
import EM as em
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans



breast_cancer = pd.read_csv('./breast-cancer-wisconsin.csv') 
li=list(breast_cancer)
breast_cancer = pd.DataFrame(breast_cancer.values, columns = li)

Class=li[-1]


arr = breast_cancer.values                                                      
y = arr[:,-1]     
X= arr[:,0:-1]
clusters=range(2,15)

ld = discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)
output = ld.fit_transform(X,y)

tester = em.ExpectationMaximizationTestCluster(output, y, clusters=range(2,15), plot=False, stats=True)
silhouette_EM,vmeasure_scores = tester.run()

tester = kmtc.KMeansTestCluster(output, y, clusters=range(2,15), plot=False, stats=True)
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
plt.text(6, 0.80, 'Breast Cancer', style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':2})
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/lda_silhouette_EM_Kmeans.png')
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
plt.text(6, 0.79, 'Breast Cancer', style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':2})
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/lda_V_measure_EM_Kmeans.png')
plt.show()

# Kmeans

np.random.seed(42)
       
data = X

n_samples, n_classes = data.shape
n_classes = len(np.unique(y))
labels = y

lda2=discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=2, store_covariance=False, tol=0.0001)
reduced_data = lda2.fit_transform(data, labels)
kmeans = KMeans(init='k-means++', n_clusters= n_classes, n_init=50)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (LDA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


# EM

np.random.seed(42)
       
data = X

n_samples, n_classes = data.shape
n_classes = len(np.unique(y))
labels = y
lda2=discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=2, store_covariance=False, tol=0.0001)
reduced_data = lda2.fit_transform(data, labels)
model = mixture.GaussianMixture(n_components=2, max_iter= 5000, n_init=50, init_params='kmeans')
model.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
#Plot the centroids as a white X

plt.title('EM clustering on the digits dataset (LDA-reduced data)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()