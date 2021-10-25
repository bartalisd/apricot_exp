from apricot import FacilityLocationSelection, FeatureBasedSelection, MaxCoverageSelection
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

def featureb(X_train, X_test, n, function, opt):
    X_train_arr = csr_matrix(X_train.values)
    X_test_arr = csr_matrix(X_test.values)
    selector = FeatureBasedSelection(n, concave_func=function, optimizer=opt, verbose=False)
    selector.fit(X_train_arr)
    Xtr, Xte = selector.transform(X_train_arr, X_test_arr)
    return(Xtr, Xte)

def facilityloc(X_train, X_test, n, metric, optimizer):
    X_train_arr = X_train.to_numpy() 
    X_test_arr = X_test.to_numpy() 
    selector = FacilityLocationSelection(n, metric, optimizer, verbose=True) 
    selector.fit(X_train)
    Xtr, Xte = selector.transform(X_train_arr, X_test_arr)
    return(Xtr, Xte)

def maxcov(X_train, X_test, n, optimizer):
    X_train_arr = csr_matrix(X_train.values)
    X_test_arr = csr_matrix(X_test.values)
    selector = MaxCoverageSelection(n, optimizer)
    selector.fit(X_train_arr)
    Xtr, Xte = selector.transform(X_train_arr, X_test_arr)
    return(Xtr, Xte)