import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans

def apply_kmeans_attacks(concated_data):
    X_train_cluster = pd.DataFrame(concated_data).iloc[0:4898431]
    X_test_cluster = pd.DataFrame(concated_data).iloc[4898431:5209460]
    X_train = np.array(X_train_cluster, dtype = np.int32)
    X_test = np.array(X_test_cluster, dtype = np.int32)
    train_size = X_train.shape[0]
    X_data = np.r_[X_train, X_test]
    data_size = X_data.shape[0]

    clustering = MiniBatchKMeans(init='k-means++',
                      n_clusters=2,
                      max_no_improvement=10,
                      verbose=0)

    clustering.fit(X_data)
    predited_labels = clustering.predict(X_data)
    #print (predited_labels.shape)
    return predited_labels
