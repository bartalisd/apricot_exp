from apricot import FacilityLocationSelection, FeatureBasedSelection, MaxCoverageSelection
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

def facilityloc(X_train, Y_train, n, metric, optimizer):
    X_train_arr = X_train.to_numpy() 
    Y_train_arr = np.array(Y_train) 
    selector = FacilityLocationSelection(n, metric, optimizer, verbose=True) 
    selector.fit(X_train)
    Xi, yi = selector.transform(X_train_arr, Y_train_arr)
    return(Xi, yi)

def featureb(X_train, Y_train, n, function, optimizer):
    X_train_arr = csr_matrix(X_train.values)
    Y_train_arr = np.array(Y_train)
    selector = FeatureBasedSelection(n, concave_func=function, optimizer, verbose=False)
    selector.fit(X_train_arr)
    Xi, yi = selector.transform(X_train_arr, Y_train_arr)
    return(Xi, yi)

def maxcov(X_train, Y_train, n, optimizer):
    X_train_arr = csr_matrix(X_train.values)
    Y_train_arr = np.array(Y_train)
    selector = MaxCoverageSelection(n, optimizer)
    selector.fit(X_train_arr)
    Xi, yi = selector.transform(X_train_arr, Y_train_arr)
    return(Xi, yi)

def mixed(X_train, Y_train, n, metric, function):
    X_train_arr = csr_matrix(X_train.values)
    Y_train_arr = np.array(Y_train)
    selector1 = FacilityLocationSelection(int(n/3), metric, verbose=True)
    selector1.fit(X_train)
    selector2 = FeatureBasedSelection(int(n/3), concave_func=function, optimizer='two-stage', verbose=False)
    selector2.fit(X_train_arr)
    selector3 = MaxCoverageSelection(int(n/3), optimizer='naive')
    selector3.fit(X_train_arr)
    X1, y1 = selector1.transform(X_train_arr, Y_train_arr)
    X2, y2 = selector1.transform(X_train_arr, Y_train_arr)
    X3, y3 = selector1.transform(X_train_arr, Y_train_arr)
    X1 = pd.DataFrame(X1.toarray())
    X2 = pd.DataFrame(X2.toarray())
    X3 = pd.DataFrame(X3.toarray())
    X_frames = [X1, X2, X3]
    X_train_a=pd.concat(X_frames)
    y1 = pd.DataFrame(y1)
    y2 = pd.DataFrame(y2)
    y3 = pd.DataFrame(y3)
    Y_frames = [y1, y2, y3]
    Y_train_a=pd.concat(Y_frames)
    return(X_train_a, Y_train_a)

