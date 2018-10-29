#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#username : mzr3
"""
Created on Sat Apr  1 01:25:23 2017

@author: melisandezonta
"""

from sklearn import  mixture, datasets, metrics,decomposition
import EM as em
import KMeansTestCluster as kmtc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans




digits = datasets.load_digits()
X = digits.data
y = digits.target
clusters=range(2,20)

pc = decomposition.PCA(n_components=None, copy=True, whiten=False, svd_solver='full', tol=0.0, iterated_power='auto', random_state=None)
output = pc.fit_transform(X)

tester = em.ExpectationMaximizationTestCluster(output, y, clusters=range(2,20), plot=False, stats=True)
silhouette_EM,vmeasure_scores = tester.run()

tester = kmtc.KMeansTestCluster(output, y, clusters=range(2,20), plot=False, stats=True)
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
plt.legend(loc = "best")
plt.text(6, 0.19, 'Breast Cancer', style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':2})
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/digits/pca_silhouette_EM_Kmeans.png')
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
plt.legend(loc = "best")
plt.text(6, 0.77, 'Breast Cancer', style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':2})
plt.grid(True)
plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/digits/pca_V_measure_EM_Kmeans.png')
plt.show()


# Kmeans

np.random.seed(42)
       
data = X

n_samples, n_classes = data.shape
n_classes = len(np.unique(y))
labels = y

sample_size = 300
reduced_data = decomposition.PCA(n_components=2,svd_solver = 'full').fit_transform(data)
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
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


np.random.seed(42)
       
data = X

n_samples, n_classes = data.shape
n_classes = len(np.unique(y))
labels = y

sample_size = 300
reduced_data = decomposition.PCA(n_components=3).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters= n_classes, n_init=50)
output = kmeans.fit_transform(reduced_data)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    fig=plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    for lab, col in zip(np.arange(0,10), 
                        ('blue', 'red','yellow','green','pink','purple','black','orange','grey','cyan')):
        ax.scatter(output[y==lab, 0],output[y==lab, 1],output[y==lab, 2],label = lab,c =col)
        ax.view_init(20,120)
        ax.set_xlabel('Principal Component 0')
        ax.set_ylabel('Principal Component 1')
        ax.set_zlabel('Principal Component 2')
        plt.legend(loc='lower right')
        plt.tight_layout()
    plt.show()
    plt.clf()


# EM

np.random.seed(42)
       
data = X

n_samples, n_classes = data.shape
n_classes = len(np.unique(y))
labels = y
reduced_data = decomposition.PCA(n_components=2,svd_solver = 'full').fit_transform(data)
model = mixture.GaussianMixture(n_components=10, max_iter= 5000, n_init=50, init_params='kmeans')
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

plt.title('EM clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


#np.random.seed(42)
#       
#data = X
#
#n_samples, n_classes = data.shape
#n_classes = len(np.unique(y))
#labels = y
#
#sample_size = 300
#reduced_data = decomposition.PCA(n_components=3).fit_transform(data)
#model = mixture.GaussianMixture(n_components=10, max_iter= 5000, n_init=50, init_params='kmeans')
#model.fit(reduced_data)
#
#with plt.style.context('seaborn-whitegrid'):
#    plt.figure(figsize=(6, 4))
#    fig=plt.figure()
#    ax=fig.add_subplot(111, projection='3d')
#    for lab, col in zip(np.arange(0,10), 
#                        ('blue', 'red','yellow','green','pink','purple','black','orange','grey','cyan')):
#        ax.scatter(output[y==lab, 0],output[y==lab, 1],output[y==lab, 2],label = lab,c =col)
#        ax.view_init(20,120)
#        ax.set_xlabel('Principal Component 0')
#        ax.set_ylabel('Principal Component 1')
#        ax.set_zlabel('Principal Component 2')
#        plt.legend(loc='lower right')
#        plt.tight_layout()
#    plt.show()
#    plt.clf()
