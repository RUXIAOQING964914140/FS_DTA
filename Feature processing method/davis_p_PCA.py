#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.decomposition import PCA


# In[3]:


X = np.loadtxt("my_VarianceThreshold/Davis_p_VT_Xnew.txt")


# In[ ]:


clf = PCA(n_components='mle',svd_solver='full')
clf.fit(X)
newX=clf.fit_transform(X)


# In[ ]:


print(newX)


# In[ ]:


clf.mean_


# In[ ]:


c = np.savetxt("my_PCA/davis_p_PCA_Xnew.txt",newX,fmt='%.03f')
d = np.savetxt("my_PCA/davis_p_PCA_Xnew_mean_.txt",clf.mean_,fmt='%.07f')


# In[ ]:




