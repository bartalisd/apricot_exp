from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
import pandas as pd
    
def traintest(X,Y,size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size, random_state=42)
    return X_train, X_test, Y_train, Y_test

def train_eval(model, X_tr, Y_tr, X_te, Y_te, experiment=None):
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_te)
    Y_pred_proba = model.predict_proba(X_te)[:,1]
    acc = accuracy_score(Y_te, Y_pred)
    pre = precision_score(Y_te, Y_pred)
    rec = recall_score(Y_te, Y_pred)
    roc = roc_auc_score(Y_te, Y_pred_proba)
    if experiment != None:
        experiment.log_metric('acc',acc)
        experiment.log_metric('pre',pre)
        experiment.log_metric('rec',rec)
        experiment.log_metric('roc',roc)
    return acc, pre, rec, roc

def randomtrain_eval(model, X_tr, Y_tr, X_te, Y_te, n, experiment=None):
    X_train_arr = X_tr.to_numpy()
    Y_train_arr = np.array(Y_tr)
    print(X_train_arr.shape, Y_train_arr.shape)
    idxs = np.arange(X_tr.shape[0])
    np.random.shuffle(idxs)
    idx = idxs[:n]
    Xi, yi = X_train_arr[idx,:], Y_train_arr[idx]  
    model.fit(Xi, yi)
    Y_pred = model.predict(X_te)
    Y_pred_proba = model.predict_proba(X_te)[:,1]
    acc = accuracy_score(Y_te, Y_pred)
    pre = precision_score(Y_te, Y_pred)
    rec = recall_score(Y_te, Y_pred)
    roc = roc_auc_score(Y_te, Y_pred_proba)
    if experiment != None:
        experiment.log_metric('acc',acc)
        experiment.log_metric('pre',pre)
        experiment.log_metric('rec',rec)
        experiment.log_metric('roc',roc)
    return acc, pre, rec, roc
    
    