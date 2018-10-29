#username : mzr3
from sklearn import datasets
import pandas as pd
import KMeansTestCluster as kmtc

if __name__ == "__main__":

    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    tester = kmtc.KMeansTestCluster(X, y, clusters=range(2,20), plot=True, stats=True)
    tester.run()