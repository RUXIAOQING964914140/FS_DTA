#!/usr/bin/env python
# coding: utf-8

# In[1]:


#lasso
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, ensemble
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV


# In[3]:


X = np.loadtxt("my_VarianceThreshold/Davis_d_VT_Xnew.txt")

y = np.loadtxt("davis_binding_affinity.txt")


# In[4]:


X.shape


# In[ ]:


#利用LassoCV找出alpha最优值
reg = linear_model.LassoCV(cv=5).fit(X,y)
model = SelectFromModel(reg, prefit=True)
X_new = model.transform(X)

#print(reg.alpha_)


# In[ ]:


print(reg.alpha_)
print(X_new.shape)
print(reg.coef_)
print(reg.coef_.shape)


# In[ ]:


c = np.savetxt("my_Lasso/davis_d_lasso_Xnew.txt",X_new,fmt='%.03f')


# In[ ]:


d = np.savetxt("my_Lasso/davis_d_lasso_Xnew_coef.txt",reg.coef_,fmt='%.07f')

