from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    
def traintest(X,Y,size):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size, random_state=42)
    return X_train, X_test, Y_train, Y_test

def train_eval(model, X_tr, Y_tr, X_te, Y_te):
    model.fit(X_tr, Y_tr)
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
    
    