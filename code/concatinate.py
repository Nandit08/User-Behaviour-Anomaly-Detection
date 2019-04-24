import pandas as pd
import numpy as np
from sklearn.externals import joblib

def concatinate_data(Xtrain,Xtest):
    Traindata_x =pd.DataFrame(Xtrain)
    TestData_x =pd.DataFrame(Xtest)
    Frames = [Traindata_x,TestData_x]
    train_test = pd.concat(Frames)
    return train_test
def concatinate_unknown(Xtrain,Xtest):
    Traindata_x =pd.DataFrame(Xtrain)
    TestData_x =pd.DataFrame(Xtest)

    Frames = [TestData_x,Traindata_x]
    train_test = pd.concat(Frames)


    return train_test
