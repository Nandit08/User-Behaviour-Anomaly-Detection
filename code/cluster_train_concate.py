import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

def concate_train(X_train,predicted_labels_train):

    Xfinal = pd.concat([pd.DataFrame(X_train),pd.DataFrame(predicted_labels_train)], axis =1)
    return Xfinal
