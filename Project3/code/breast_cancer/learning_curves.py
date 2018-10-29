#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#username : mzr3
"""
Created on Sun Apr  2 21:20:50 2017

@author: melisandezonta
"""

import numpy as np                                                     
import pandas as pd                                                    
import matplotlib.pyplot as plt  
import seaborn as sb                                                                              
from sklearn import model_selection                                             
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, ShuffleSplit,GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.datasets import load_digits
from functions import *
from sklearn import  datasets, mixture, metrics, decomposition, discriminant_analysis
from sklearn.random_projection import SparseRandomProjection
from sklearn.cluster import KMeans

# In[1]:
#Splitting datas

breast_cancer = pd.read_csv("/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/assignment 3/breast_cancer/test/breast-cancer-wisconsin.csv") 
li=list(breast_cancer)
breast_cancer = pd.DataFrame(breast_cancer.values, columns = li)

Class=li[-1]


arr = breast_cancer.values                                                      
y = arr[:,-1]     
X = arr[:,0:-1]


# In[2]:
#Learning curves after PCA

pc = decomposition.PCA(n_components='mle', copy=True, whiten=False, svd_solver='full', tol=0.0, iterated_power='auto', random_state=None)
output = pc.fit_transform(X)

new_inputs = output
          
title = "Learning Curves (Neural Network) after PCA"
cv = ShuffleSplit(n_splits=50, test_size=0.33, random_state=0)
estimator = MLPClassifier(hidden_layer_sizes = (100,100),max_iter = 5000, learning_rate = 'constant')
plot_iterative_learning_curve(estimator, title, new_inputs, y, ylim=None, cv= cv, n_jobs=-1,
                              iterations=np.arange(1, 100, 5), exploit_incremental_learning=True)
#plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/learning_curve_pca.png')
plt.savefig('/Users/jacquelineroudes/Desktop/SIMO/learning_curve_pca.png')
# In[3]:
#Learning curves after ICA

ica = decomposition.FastICA(n_components = None, whiten=True)
output =ica.fit_transform(X)

new_inputs = output
title = "Learning Curves (Neural Network) after ICA"
cv = ShuffleSplit(n_splits=50, test_size=0.33, random_state=0)
estimator = MLPClassifier(hidden_layer_sizes = (100,100),max_iter = 5000, learning_rate = 'constant')
plot_iterative_learning_curve(estimator, title, new_inputs, y, ylim=None, cv= cv, n_jobs=-1,
                              iterations=np.arange(1, 100, 5), exploit_incremental_learning=True)
#plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/learning_curve_ica.png')
plt.savefig('/Users/jacquelineroudes/Desktop/SIMO/learning_curve_ica.png')
# In[4]:
#Learning curves after LDA

ld = discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)
output = ld.fit_transform(X,y)

new_inputs = output

title = "Learning Curves (Neural Network) after LDA"
cv = ShuffleSplit(n_splits=50, test_size=0.33, random_state=0)
estimator = MLPClassifier(hidden_layer_sizes = (100,100),max_iter = 5000, learning_rate = 'constant')
plot_iterative_learning_curve(estimator, title, output, y, ylim=None, cv= cv, n_jobs=-1,
                              iterations=np.arange(1, 100, 5), exploit_incremental_learning=True)
#plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/learning_curve_lda.png')
plt.savefig('/Users/jacquelineroudes/Desktop/SIMO/learning_curve_lda.png')
# In[5]:
#Learning curves after RP

sp = SparseRandomProjection(n_components = 40)
output = sp.fit_transform(X)

new_inputs = output
          
title = "Learning Curves (Neural Network) after RP"
cv = ShuffleSplit(n_splits=50, test_size=0.33, random_state=0)
estimator = MLPClassifier(hidden_layer_sizes = (100,100),max_iter = 5000, learning_rate = 'constant')
plot_iterative_learning_curve(estimator, title, output, y, ylim=None, cv= cv, n_jobs=-1,
                              iterations=np.arange(1, 100, 5), exploit_incremental_learning=True)
#plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/learning_curve_rp.png')

plt.savefig('/Users/jacquelineroudes/Desktop/SIMO/learning_curve_rp.png')
# In[6]:
#Learning curves after K Means

n_samples, n_classes = X.shape
n_classes = len(np.unique(y))

kmeans = KMeans(init='k-means++', n_clusters= n_classes, n_init=50)
output = kmeans.fit_predict(X)

new_inputs = np.ndarray((output.shape[0], 1))
new_inputs[:,0] = output
          
title = "Learning Curves (Neural Network) after K Means Clustering"
cv = ShuffleSplit(n_splits=50, test_size=0.33, random_state=0)
estimator = MLPClassifier(hidden_layer_sizes = (100,100),max_iter = 5000, learning_rate = 'constant')
plot_iterative_learning_curve(estimator, title, new_inputs, y, ylim=None, cv= cv, n_jobs=-1,
                              iterations=np.arange(1, 100, 5), exploit_incremental_learning=True)
#plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/learning_curve_kmeans.png')
plt.savefig('/Users/jacquelineroudes/Desktop/SIMO/learning_curve_kmeans.png')
# In[7]:
#Learning curves after EM

n_samples, n_classes = X.shape

EM = mixture.GaussianMixture(n_components= n_classes, max_iter= 5000, n_init=50, init_params='kmeans')
EM.fit(X)
output = EM.predict(X,y)

new_inputs = np.ndarray((output.shape[0], 1))
new_inputs[:,0] = output
          
title = "Learning Curves (Neural Network) after EM Clustering"
cv = ShuffleSplit(n_splits=50, test_size=0.33, random_state=0)
estimator = MLPClassifier(hidden_layer_sizes = (100,100),max_iter = 5000, learning_rate = 'constant')
plot_iterative_learning_curve(estimator, title, new_inputs, y, ylim=None, cv= cv, n_jobs=-1,
                              iterations=np.arange(1, 100, 5), exploit_incremental_learning=True)
#plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/learning_curve_EM.png')

plt.savefig('/Users/jacquelineroudes/Desktop/SIMO/learning_curve_em.png')