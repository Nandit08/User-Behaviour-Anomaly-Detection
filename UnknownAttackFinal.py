import pandas as pd
import numpy as np
from sklearn.externals import joblib
from concatinate import *
from cluster_train_concate import concate_train
from cluster_test_concate import concate_test
from DecisionTreeClassifier import *
import time
from Unknown_labelling_model import *
from Clustering_Unknown_attacks import apply_kmeans_attacks

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



    Y_label_train = labeling(Y_train)
    Y_label_test = labeling(Y_test)
    y_label = np.unique(Y_label_test)




    concated_data = concatinate_data(X_train_data,X_test_data)
    X_data = np.r_[X_train_data, X_test_data]
    data_size = X_data.shape[0]
    print(data_size)
    print(concated_data.shape)

    predicted_labels = apply_kmeans_attacks(concated_data)
    predicted_array = np.split(predicted_labels, [X_train_data.shape[0],data_size])
    predicted_labels_train = predicted_array[0]
    predicted_labels_test = predicted_array[1]

    concated_train = concate_train(X_train_data,predicted_labels_train)
    concated_test = concate_test(X_test_data,predicted_labels_test)

    apply_decision_trees(concated_train,Y_label_train,concated_test,Y_label_test,y_label)
    #np.savetxt('train.csv',Y_label_train,delimiter =',' )

main()
