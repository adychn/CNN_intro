#!/usr/bin/env python
# coding: utf-8

# 假设我们需要预测一个线性回归的参数w1,w2,b：
#
# ```
# pred = w1x1 + w2x2 + b
# ```
#
# 线性回归的损失函数   loss = (pred - targ) ^ 2
#
# In[13]:
import random

# In[19]:
xs = []
targs = []

# w1 = 3.5,w2 = -1.5, b = 4

# In[20]:
for i in range(100):
    for j in range(100):
        i = float(i)
        j = float(j)
        xs.append((i,j))
        targs.append(i * 3.5 + (-1.5) * j + 4)

# In[21]:
w1 = random.random()
w2 = random.random()
b = random.random()

w1, w2, b
# In[22]:
train_tuples = list(zip(xs,targs))
train_tuples

# In[24]:
log = []
learning_rate = 0.00001
for batch_number in range(500):
    (x1, x2), targ = random.choice(train_tuples)
    delta_w1 = 2 * (w1 * x1 + w2 * x2 + b - targ) * x1
    delta_w2 = 2 * (w1 * x1 + w2 * x2 + b - targ) * x2
    delta_b = 2 * (w1 * x1 + w2 * x2 + b - targ)

    w1 -= delta_w1 * learning_rate
    w2 -= delta_w2 * learning_rate
    b -= delta_b * learning_rate * 10000 # delta b is too small so increase learning rate.

    log.append([w1, w2, b, (w1 * x1 + w2 * x2 + b - targ) ** 2])

# In[26]:
from matplotlib import pyplot as plt
import numpy as np

# In[27]:
log = np.asarray(log)


# In[28]:
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(log[:,2])

# In[29]:
w1,w2,b

# In[30]:
get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplot(1,2,1)
plt.plot(log[:,0])
plt.subplot(1,2,2)
plt.plot(log[:,1])


# In[31]:


random.choice(train_tuples)
