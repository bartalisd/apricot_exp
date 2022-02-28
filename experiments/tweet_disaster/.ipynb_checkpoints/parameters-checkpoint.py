from comet_ml import Experiment
from apricot_exp.func_tp import featureb, facilityloc, maxcov

def param(function):
    if function == "featurebased":
        parameters = [
            ("function", "choice", ["log","sqrt"]),
            ("optimizer", "choice", ["lazy", "approximate-lazy", "stochastic", "sample"]),
        ]
    elif function == "facilitylocation":
        parameters = [
            ("function", "choice", ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "correlation"]),
            #("optimizer", "choice", ["random","modular", "naive", "lazy", "approximate-lazy", "two-stage", "stochastic", "sample", "greedi", "bidirectional"]),
        ]
    elif function == "maxcoverage":
        parameters = [
            ("optimizer", "choice", ["random","modular", "naive", "lazy", "approximate-lazy", "two-stage", "stochastic", "sample", "greedi", "bidirectional"]),
        ]
    return parameters

def extract_grid(parameters):
    """Extract grid space from manually defined search parameters"""
    grid = {}
    for c_name, c_type, c_vals in parameters:
        if c_type == "choice":
            grid[c_name] = c_vals
        elif c_type == "fixed":
            grid[c_name] = [c_vals]
        else:
            raise ValueError("GridSearch can only use categorical search space!")
    return grid

from optuna.samplers import TPESampler, RandomSampler, GridSampler
def algo(search_alg, parameters):
    if search_alg == "GRID":
        algo = GridSampler(extract_grid(parameters))
    elif search_alg == "RND":
        algo = RandomSampler()
    elif search_alg == "TPE":
        algo = TPESampler()
    return algo

def objective_fb(trial, n):
    config = suggest_config(parameters, trial)
    Xtr_t, Xte_t = featureb(features_t, features_test_t, n, config["function"], config["optimizer"])
    Xtr = Xtr_t.transpose()
    Xte = Xte_t.transpose()
    acc, pre, rec, roc = train_eval(model, Xtr, Y_train, Xte, Y_test, experiment)
    return roc

def objective_fa(trial, n):
    config = suggest_config(parameters, trial)
    Xtr_t, Xte_t = facilityloc(features_t, features_test_t, n, config["function"], config["optimizer"])
    Xtr = Xtr_t.transpose()
    Xte = Xte_t.transpose()
    acc, pre, rec, roc = train_eval(model, Xtr, Y_train, Xte, Y_test, experiment)
    return roc

def objective_mc(trial, n):
    config = suggest_config(parameters, trial)
    Xtr_t, Xte_t = maxcov(features_t, features_test_t, n, config["optimizer"])
    Xtr = Xtr_t.transpose()
    Xte = Xte_t.transpose()
    acc, pre, rec, roc = train_eval(model, Xtr, Y_train, Xte, Y_test, experiment)
    return roc