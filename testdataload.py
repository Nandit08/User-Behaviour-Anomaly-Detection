import pandas as pd
import numpy as np
from sklearn.externals import joblib


def testdata():
    testdata = pd.read_csv('data/corrected', low_memory = False)
    numerical_features=['src_bytes','dst_bytes','duration','hot','num_failed_logins','num_compromised','num_root',
                        'num_file_creations','num_access_files','count_f','srv_count','dst_host_count','dst_host_srv_count',
                        'srv_count']
    Xtest = pd.DataFrame(testdata,columns= numerical_features)
    Ytest = pd.DataFrame(testdata,columns= ['Result'])
    print(Xtest.shape)
    print(Ytest.shape)
    print('data loaded')
    joblib.dump(np.array(Xtest), 'sorted_dump/Xtest.pkl')
    joblib.dump(np.array(Ytest), 'sorted_dump/Ytest.pkl')
    print('data dumped')
