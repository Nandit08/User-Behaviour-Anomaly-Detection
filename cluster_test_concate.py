import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

def concate_test(X_test,predicted_labels_test):


    Xfinal = pd.concat([pd.DataFrame(X_test),pd.DataFrame(predicted_labels_test)], axis =1)
    return Xfinal
