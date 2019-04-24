import pandas as pd
import numpy as np
from sklearn.externals import joblib
from concatinate import *
from clustering import apply_kmeans
from cluster_train_concate import concate_train
from cluster_test_concate import concate_test
from DecisionTreeClassifier import *
from DecisionTreeClassifier_unknown import *
from clustering import apply_kmeans
from sklearn.metrics import roc_curve, auc
from clustering_unknown import *
from sklearn.externals import joblib
import time
def main():

    X_train = joblib.load('Dump/Xdata.pkl')
    Y_train = joblib.load('Dump/Ydata.pkl')
    X_train_data = np.array(X_train)
    X_train_data = np.delete(X_train_data, np.where(Y_train=='Result'), axis=0)
    Y_train = np.delete(Y_train, np.where(Y_train=='Result'), axis=0)

    X_test = joblib.load('Dump/Xtest.pkl')
    Y_test = joblib.load('Dump/Ytest.pkl')
    X_test_data = np.array(X_test, dtype = np.int32)
    X_test_data = np.delete(X_test_data, np.where(Y_test=='Result'), axis=0)
    Y_test = np.delete(Y_test, np.where(Y_test=='Result'), axis=0)

    idx = np.isin(Y_test, np.unique(Y_train)).ravel()
    X_test_known = X_test_data[idx]
    Y_test_known = Y_test[idx]
    y_label = np.unique(Y_test_known)



    index = np.isin(Y_test,['normal.']).ravel()
    X_test_unknown = X_test_data[~index]
    Y_test_unknown = Y_test[~index]

    concated_data = concatinate_data(X_train_data,X_test_known)
    X_data = np.r_[X_train_data, X_test_data]
    data_size = X_data.shape[0]

    predicted_labels = apply_kmeans(concated_data)
    predicted_array = np.split(predicted_labels, [X_train_data.shape[0],data_size])
    predicted_labels_train = predicted_array[0]
    predicted_labels_test = predicted_array[1]

    concated_train = concate_train(X_train_data,predicted_labels_train)
    concated_test = concate_test(X_test_known,predicted_labels_test)

    apply_decision_trees(concated_train,Y_train,concated_test,Y_test_known,y_label)
    time.sleep(5)

    #unknown_testing(X_train_data,Y_train,X_test_unknown,Y_test_unknown)



def unknown_testing(X_train_data,Y_train,X_test_unknown,Y_test_unknown):
    concated_data = concatinate_unknown(X_train_data,X_test_unknown)
    X_data = np.r_[X_test_unknown,X_train_data]
    y_label_uk = np.unique(Y_test_unknown)
    data_size = X_data.shape[0]

    predicted_labels = apply_kmeans_unknown(concated_data)
    predicted_array = np.split(predicted_labels,[X_test_unknown.shape[0],data_size])
    predicted_labels_train = predicted_array[1]
    predicted_labels_test = predicted_array[0]

    concated_train = concate_train(X_train_data,predicted_labels_train)
    concated_test = concate_test(X_test_unknown,predicted_labels_test)

    apply_decision_trees_unknown(concated_train,Y_train,concated_test,Y_test_unknown,y_label_uk)

main()
