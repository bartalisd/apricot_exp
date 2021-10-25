from comet_ml import Experiment

def suggest_config(parameters, trial):
    """Convert manually defined search parameters into Optuna supported search spaces"""
    config = {}
    for c_name, c_type, c_vals in parameters:
        if c_type == "choice":
            config[c_name] = trial.suggest_categorical(c_name, c_vals)
        elif c_type == "int":
            config[c_name] = trial.suggest_int(c_name, c_vals[0], c_vals[1], step=c_vals[2] if len(c_vals) > 2 else 1, log=c_vals[3] if len(c_vals) > 3 else False)            
        elif c_type == "float":
            config[c_name] = trial.suggest_float(c_name, c_vals[0], c_vals[1], step=c_vals[2] if len(c_vals) > 2 else 1, log=c_vals[3] if len(c_vals) > 3 else False)            
        elif c_type == "fixed":
            config[c_name] = c_vals
        else:
            raise ValueError("Parameter type '%s' was not implemented!" % c_type)
    return config

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

