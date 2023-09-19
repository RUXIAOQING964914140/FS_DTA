#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#基于树的特征选择
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error


# In[ ]:


#基于davis特征数据VT处理后的特征进行树的特征选择
#X = np.loadtxt("all_ssaf_asaf_gaaf_aaaf.csv",delimiter = ",",skiprows=1)
#X = np.loadtxt("my_VarianceThreshold/Davis_p_VT_Xnew.txt")
X = np.loadtxt("train_Davis_p_VT_Xnew.txt")

y = np.loadtxt("train_affi.txt")
#y = np.loadtxt("Trans_davis_binding_affinity.txt")


# In[ ]:


lightparams = {
    #"boosting_type":['gbdt','dart','goss','rf'],
    "boosting_type":['gbdt'],
    #"n_estimators": range(100,1100,100),
    "n_estimators":[200,400,600,800],
    #"max_depth": range(1,7,1),
    "max_depth": [3,4,5,6],
    "learning_rate":[0.02,0.04,0.06,0.08]

   
    #"learning_rate": np.linspace(0.01,0.1,10)
    
    
}


# In[ ]:


light_m = LGBMRegressor()
light_grid_search = GridSearchCV(light_m, lightparams, cv=5,scoring='neg_mean_squared_error')
light_grid_search.fit(X, y)


# In[ ]:


a = light_grid_search.best_params_
b = str(a)
with open("light_davis_p_d_afterVT.txt","w") as file:
    file.write(b)

