#username : mzr3
from sklearn import datasets
import pandas as pd
import KMeansTestCluster as kmtc

if __name__ == "__main__":
    breast_cancer = pd.read_csv('./breast-cancer-wisconsin.csv') 
    #digits = datasets.load_digits()
    li=list(breast_cancer)
    breast_cancer = pd.DataFrame(breast_cancer.values, columns = li)
    
    Class=li[-1]
    
    
    arr = breast_cancer.values                                                      
    y = arr[:,-1]     
    X= arr[:,0:-1]

        
    tester = kmtc.KMeansTestCluster(X, y, clusters=range(2,15), plot=True, stats=True)
    tester.run()