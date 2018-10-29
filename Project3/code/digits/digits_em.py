#username : mzr3
from sklearn import  datasets
import EM as em


if __name__ == "__main__":
    

    

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    tester = em.ExpectationMaximizationTestCluster(X, y, clusters=range(2,20), plot=True, stats=True)
    tester.run()
    
    

