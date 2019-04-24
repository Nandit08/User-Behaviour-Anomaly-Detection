import pandas as pd
import numpy as np
from sklearn.externals import joblib



def traindata():
    traindata = pd.read_csv('data/kddcup.csv', low_memory = False)
    numerical_features=['src_bytes','dst_bytes','duration','hot','num_failed_logins','num_compromised','num_root',
                        'num_file_creations','num_access_files','count_f','srv_count','dst_host_count','dst_host_srv_count',
                        'srv_count']
    Xdata = pd.DataFrame(traindata,columns=features)
    Ydata= pd.DataFrame(traindata,columns= ['Result'])
    print('data loaded')
    joblib.dump(np.array(Xdata), 'sorted_dump/Xdata.pkl')
    joblib.dump(np.array(Ydata), 'sorted_dump/Ydata.pkl')
    print('data dumped')
