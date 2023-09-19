#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ivis import Ivis
from tensorflow.keras.datasets import boston_housing
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ivis_explanations import LinearExplainer
from sklearn.model_selection import GridSearchCV


# In[ ]:


#基于davis特征数据VT处理后的特征进行树的特征选择
#X = np.loadtxt("all_ssaf_asaf_gaaf_aaaf.csv",delimiter = ",",skiprows=1)
#X = np.loadtxt("my_VarianceThreshold/Davis_p_VT_Xnew.txt")
X = np.loadtxt("train_Davis_p_VT_Xnew.txt")

y = np.loadtxt("train_affi.txt")
#y = np.loadtxt("Trans_davis_binding_affinity.txt")


# In[ ]:


#ivis的超参数
ivisparams = {
    "embedding_dims": [128, 256, 512,1024,2048],
    "k": range(10,151,30),
    'n_epochs_without_progress':range(10,21,5),
    
    #"model": ['szubert','hinton','maaten'],
    "model": ['maaten'],
    #"supervision_metric":['mae','mse','rms']
    "supervision_metric":['mse']
}


# In[ ]:


ivis_m = Ivis()
ivis_grid_search = GridSearchCV(ivis_m, ivisparams, cv=5,scoring='neg_mean_squared_error')
ivis_grid_search.fit(X, y)


# In[ ]:


a = ivis_grid_search.best_params_
b = str(a)
with open("ivis_p_afterVT.txt","w") as file:
    file.write(b)

