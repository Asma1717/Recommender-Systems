#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Recommendation System
# ## Classification-based Collaborative Filterning Systems
# ### Logistic Regression as a Classifier

# In[4]:


import pandas as pd
import numpy as np

from pandas import Series,DataFrame
from sklearn.linear_model import LogisticRegression


# In[5]:


bank_full = pd.read_csv('C:/Users/HP/Desktop/bank_full_w_dummy_vars.csv')
bank_full.head()


# In[6]:


bank_full.info()


# In[7]:


x = bank_full.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
x


# In[8]:


y = bank_full.ix[:,17].values
y


# In[9]:


LogReg = LogisticRegression()
LogReg.fit(x,y)


# In[46]:


new_user =np.array([0, 0, 0, 0, 0, 1, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])



# In[48]:


y_pred = LogReg.predict(new_user.reshape(1,-1))
y_pred


# In[ ]:




