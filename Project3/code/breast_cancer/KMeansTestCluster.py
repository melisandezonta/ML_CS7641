# username : mzr3

from functions_clustering import ratio_BSS_TSS as ratio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
#import pandas as pd
from sklearn import metrics #, decomposition
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from functions_clustering import WSS_error,SWC_distance


class KMeansTestCluster():
    def __init__(self, X, y, clusters, plot=False, stats=False):
        self.X = X
        self.y = y
        self.clusters = clusters
        self.gen_plot = plot
        self.stats = stats

    def run(self):
        
        SWCD = []
        WSS=[]
        V_measure=[]
        silhouette=[]
        BSS_TSS=[]
        cIDx = []
        
        BSS_TSS = ratio(self.X,len(self.clusters)+1)
        SWCD = SWC_distance(self.X,len(self.clusters)+1)
        WSS,cIDx = WSS_error(self.X,len(self.clusters)+1)
        
        for k in self.clusters:
            
            model = KMeans(n_clusters=k, max_iter=5000, init='k-means++',n_init = 100)
            labels = model.fit_predict(self.X)
            V_measure.append(metrics.v_measure_score(self.y, labels))

            silhouette.append(metrics.silhouette_score(self.X, labels , metric='euclidean',sample_size=self.X.shape[0]))

            
        if self.gen_plot:
            self.plot(WSS,SWCD, cIDx,V_measure, silhouette,BSS_TSS)

        else:
            return silhouette,V_measure

    def plot(self, WSS,SWCD, cIDx,V_measure, silhouette, BSS_TSS):


            """
            Plot V_measure from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            plt.plot(self.clusters, V_measure, 'b.-')
            plt.xlabel('Number of clusters')
            plt.ylabel('V_measure')
            plt.title('V_measure Score vs. K Clusters')
            plt.show()

            plt.clf()


     

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
            Plot ratio BSS to TSS from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """

            plt.plot(self.clusters, BSS_TSS, 'b*-')
            plt.xlabel('Number of clusters')
            plt.ylabel('BSS/TSS')
            plt.title('Ratio of Between Sum of Square error Score to Total Sum of Square error vs. K Clusters')
            plt.grid(True)
            plt.text(6, 0.8, 'Breast Cancer', style='italic',bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':2})
            plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/BSS_TSS.png')
            plt.show()

            plt.clf()
            
            """
            Plot WSS and SWCD from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(self.clusters, WSS, 'b*-')
            ax2.plot(self.clusters, SWCD, 'r*-')
            plt.grid(True)
            ax1.set_xlabel('Number of clusters')
            ax1.set_ylabel('Within Sum of Square Error', color='b')
            ax2.set_ylabel('Sum of Within Clusters Distances', color='r')
            plt.text(6, 960, 'Breast Cancer', style='italic',bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':2})
            plt.savefig('/Users/jacquelineroudes/Documents/GTL_courses/Machine_Learning/Homework3/images/breast_cancer/WSS_SWCD.png')
            

            
            