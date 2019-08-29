#!/usr/bin/env python
# coding: utf-8

# 假设我们需要预测一个线性回归的参数w1,w2,b：
#     
# ```
# pred = wx + b (w,x均为向量)
# ```   
# 
# 线性回归的损失函数   loss = (pred - targ) ^ 2
# 

# In[1]:


import random
import numpy as np


# In[2]:


xs = []
targs = []


# In[3]:


for i in range(100):
    for j in range(100):
        i = float(i)
        j = float(j)
        xs.append(np.asarray((i,j)))
        targs.append(i * 3.5 + (-1.5) * j + 4)


# In[4]:


w = np.asarray([random.random(),random.random()])
b = random.random()


# In[5]:


train_tuples = list(zip(xs,targs))


# In[6]:


log = []


# In[7]:


learning_rate = 0.00001
for batch_number in range(500):
    x,targ = random.choice(train_tuples)
    delta_w = 2 * (np.matmul(x,w) +  b - targ) * x
    delta_b = 2 * (np.matmul(x,w) + b - targ)
    
    w -= delta_w * learning_rate
    b -= delta_b * learning_rate
    
    loss = (np.matmul(x,w) + b - targ) ** 2
    log.append([w[0],w[1],b,loss])


# In[8]:


from matplotlib import pyplot as plt
import numpy as np


# In[9]:


log = np.asarray(log)


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(log[:,3])


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplot(1,2,1)
plt.plot(log[:,0])
plt.subplot(1,2,2)
plt.plot(log[:,1])


# In[ ]:




