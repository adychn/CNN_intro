#!/usr/bin/env python
# coding: utf-8
# 假设我们需要预测一个线性回归的参数w,b：
#
# ```
# pred = wx + b
# ```
#
# 线性回归的损失函数   loss = (pred - targ) ^ 2
#
# In[1]:
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# In[2]:
xs = []
targets = []

# w = 3.5,b = 4
# In[3]:
for i in range(100):
    i = float(i)
    xs.append(i)
    targets.append(3.5 * i + 4)

# In[4]:
w = random.random()
b = random.random()
w, b

# In[5]:
train_tuples = list(zip(xs,targets))
train_tuples

# In[6]:
learning_rate = 1e-5
log = []
for batch_nubmer in range(500):
    x, target = random.choice(train_tuples)
    delta_w = 2 * (w * x + b - target) * x # derivative of loss
    delta_b = 2 * (w * x + b - target)

    # 一輪跌代的更新
    w = w - learning_rate * delta_w
    b = b - learning_rate * delta_b
    loss = (w * x + b - target) ** 2

    log.append([w,b,loss])

log
# In[7]:
from matplotlib import pyplot as plt
import numpy as np

# In[8]:
log = np.asarray(log)

# In[9]:
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(log[:,2])


# In[10]:


w,b


# In[ ]:
