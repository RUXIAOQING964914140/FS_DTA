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
from catboost import CatBoostRegressor


# In[ ]:


#基于davis特征数据VT处理后的特征进行树的特征选择
#X = np.loadtxt("all_ssaf_asaf_gaaf_aaaf.csv",delimiter = ",",skiprows=1)
X = np.loadtxt("train_Davis_p_VT_Xnew.txt")

y = np.loadtxt("train_affi.txt")


# In[ ]:


catparams = {
    #"iterations": range(100,1100,100),
    "iterations":[200,400,600,800],
    #"depth": range(1,7,1),
    "depth":[3,4,5,6],
    #"loss_function":['MAE','Poisson','RMSE','R2','Quantile'],
    "loss_function":['RMSE'],
    #"custom_metric":['MAE','Poisson','RMSE','R2','Quantile'],
    "custom_metric":['RMSE'],
    #"eval_metric":['MAE','Poisson','RMSE','R2','Quantile'],
    "eval_metric":['RMSE'],
    #'min_data_in_leaf':range(1,7,1),
    'min_data_in_leaf':[2,3,4,5],
    "learning_rate":[0.02,0.04,0.06,0.08]
    #'learning_rate':np.linspace(0.01,0.1,10)
   

    
    
}


# In[ ]:


cat_m = CatBoostRegressor()
cat_grid_search = GridSearchCV(cat_m, catparams, cv=5,scoring='neg_mean_squared_error')
cat_grid_search.fit(X, y)


# In[ ]:


a = cat_grid_search.best_params_
b = str(a)
with open("cat_davis_p_afterVT.txt","w") as file:
    file.write(b)

