import pandas as pd
import numpy as np
from apricot_exp.functions import facilityloc

def test_facilityloc():
    X = np.array([[1, 2], [3, 4]])
    Y = np.array([1, 1])
    X = pd.DataFrame(X)
    X_small, Y_small = facilityloc(X, Y, 1, 'euclidean')
    assert len(X_small) == 1
    assert len(Y_small) == 1
