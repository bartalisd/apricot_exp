from apricot import FacilityLocationSelection, FeatureBasedSelection, MaxCoverageSelection
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

def facilityloc(X_train, Y_train, n, metric):
    X_train_arr = X_train.to_numpy() 
    Y_train_arr = np.array(Y_train) 
    selector = FacilityLocationSelection(n, metric, verbose=True) 
    selector.fit(X_train)
    Xi, yi = selector.transform(X_train_arr, Y_train_arr)
    return(Xi, yi)

def featureb(X_train, Y_train, n, function):
    X_train_arr = csr_matrix(X_train.values)
    Y_train_arr = np.array(Y_train)
    selector = FeatureBasedSelection(n, concave_func=function, optimizer='two-stage', verbose=False)
    selector.fit(X_train_arr)
    Xi, yi = selector.transform(X_train_arr, Y_train_arr)
    return(Xi, yi)

def maxcov(X_train, Y_train, n):
    X_train_arr = csr_matrix(X_train.values)
    Y_train_arr = np.array(Y_train)
    selector = MaxCoverageSelection(n, optimizer='naive')
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

def randomtrain_eval(model, X_tr, Y_tr, X_te, Y_te):
    X_train_arr = X_tr.to_numpy()
    Y_train_arr = np.array(Y_tr)
    idxs = np.arange(X_tr.shape[0])
    np.random.shuffle(idxs)
    idx = idxs[:n]
    Xi, yi = X_tr[idx], Y_tr[idx]  
    model.fit(Xi, yi)
    Y_pred = model.predict(X_te)
    Y_pred_proba = model.predict_proba(X_te)[:,1]
    acc = accuracy_score(Y_te, Y_pred)
    pre = precision_score(Y_te, Y_pred)
    rec = recall_score(Y_te, Y_pred)
    roc = roc_auc_score(Y_te, Y_pred_proba)
    #neptune.log_metric('acc',acc)
    #neptune.log_metric('pre',pre)
    #neptune.log_metric('rec',rec)
    #neptune.log_metric('roc',roc)
    return acc, pre, rec, roc
