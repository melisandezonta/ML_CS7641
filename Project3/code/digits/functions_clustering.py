#username : mzr3

#from sklearn import   metrics, decomposition, cluster
import numpy as np
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans






def ratio_BSS_TSS(df, n):
        kMeansVar = [KMeans(n_clusters=k).fit(df) for k in range(1, n)]
        centroids = [X.cluster_centers_ for X in kMeansVar]
        k_euclid = [cdist(df, cent) for cent in centroids]
        dist = [np.min(ke, axis=1) for ke in k_euclid]
        wcss = [sum(d**2) for d in dist]
        tss = sum(pdist(df)**2)/df.shape[0]
        bss = tss - wcss
        ratio=bss/tss
        return(ratio)
    

def WSS_error(X,K):
# scipy.cluster.vq.kmeans
    kMeansVar = [KMeans(n_clusters=k).fit(X) for k in range(1, K)]
    centroids = [X.cluster_centers_ for X in kMeansVar]  # cluster centroids

        
    # alternative: scipy.spatial.distance.cdist
    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/X.shape[0] for d in dist]
    
    return avgWithinSS,cIdx
    
def SWC_distance(df,K):
    
    kMeansVar = [KMeans(n_clusters=k).fit(df) for k in range(1, K)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_manhattan = [cdist(df, cent,'cityblock') for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_manhattan]
    wcss = [sum(d)/K for d in dist]
  
    return(wcss)