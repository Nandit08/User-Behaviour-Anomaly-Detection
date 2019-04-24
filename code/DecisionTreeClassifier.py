import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from ROC_AUC import *
from sklearn.externals import joblib
def apply_decision_trees(concated_train,Y_train,concated_test,Y_test_known,y_label):
    #ROC_Known(concated_train,Y_train,concated_test,Y_test_known)
    numerical_features=['src_bytes','dst_bytes','duration','hot','num_failed_logins','num_compromised','num_root',
                            'num_file_creations','num_access_files','count_f','srv_count','dst_host_count','dst_host_srv_count',
                            'srv_count']

    #sgd = SVC(random_state=0)
    sgd = DecisionTreeClassifier(random_state=0,splitter='random')
    #sgd = SGDClassifier(class_weight='balanced')
    #sgd = GradientBoostingClassifier()
    sgd.fit(concated_train,Y_train)
    #print(concated_train.shape)
    #joblib.dump(sgd,'Dump/trained_model.pkl')

    Y_pred = sgd.predict(concated_test)
    score = accuracy_score(Y_test_known, Y_pred)

    print(score)
    print(classification_report(Y_test_known, Y_pred,target_names = numerical_features))
    #ROC_Known(concated_train,Y_train,concated_test,Y_test_known)
    #conf = confusion_matrix(Y_test, Y_pred, labels = y_label)

    # Plot non-normalized confusion matrix

    #plt.figure()
    #plot_confusion_matrix(conf, classes=y_label, title='Confusion matrix')


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
