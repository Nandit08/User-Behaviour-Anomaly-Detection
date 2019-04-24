import pandas as pd
import numpy as np

def labeling(Y_train):

    index = np.isin(Y_train,['normal.']).ravel()
    Y_train[~index] = 'Attack.'


    return Y_train
