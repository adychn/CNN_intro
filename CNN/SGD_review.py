#!/usr/bin/env python
# coding: utf-8

# # SGD

# In[4]:


y = (x - 1) ** 2


# In[9]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[13]:


xs = []
ys = []
for x in np.arange(-1,4,0.05):
    xs.append(x)
    ys.append((x - 1) ** 2)


# In[14]:


plt.scatter(xs,ys)


# In[23]:


lr = 0.01
x = 4
log = []
for i in range(300):
    grad = 2 * x - 2
    x = x - grad * lr
    log.append(x)


# In[25]:


import pandas as pd


# In[26]:


pd.DataFrame(log).plot()


# # vector SGD

# In[27]:


y = (x - 1) ** 2


# In[28]:


lr = 0.01
x = np.random.randn(10)
log = []
for i in range(300):
    grad = 2 * x - 2
    x = x - grad * lr
    log.append(x)


# In[32]:


pd.DataFrame(log).plot()


# In[ ]:




