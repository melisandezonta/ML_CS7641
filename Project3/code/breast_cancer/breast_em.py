
#username = mzr3
import EM as em
import pandas as pd

if __name__ == "__main__":
    



    breast_cancer = pd.read_csv('./breast-cancer-wisconsin.csv') 
    li=list(breast_cancer)
    breast_cancer = pd.DataFrame(breast_cancer.values, columns = li)
    
#   Class=li[-1]

    arr = breast_cancer.values                                                      
    y = arr[:,-1]     
    X= arr[:,0:-1]


    tester = em.ExpectationMaximizationTestCluster(X, y, clusters=range(2,15), plot=True, stats=True)
    tester.run()
    
    

