#!/usr/bin/env python
# coding: utf-8

# In[99]:


import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[494]:


def conv_forward(inputx,kernel_size,pad,filters,params = None):
    # inputx should be like (batch_number,channel,width,height)
    N, C, H, W = inputx.shape
    if params is None:
        w = np.random.randn(filters,C,kernel_size,kernel_size)
        b = np.random.randn(filters)
    else:
        (w,b) = params
    
    x_pad = np.pad(inputx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    x_pad_H,x_pad_W = x_pad.shape[2:]
    
    
    x_out = np.zeros((N, filters, x_pad_H - kernel_size + 1, x_pad_W - kernel_size + 1))
    for n in range(N):
        for f in range(filters):
            for i in range(x_pad_H - kernel_size + 1):
                for j in range(x_pad_W - kernel_size + 1):
                    begin_x_cord = i
                    end_x_cord = i + kernel_size
                    begin_y_cord = j 
                    end_y_cord = j + kernel_size
                    window = x_pad[n,:,begin_x_cord:end_x_cord,begin_y_cord:end_y_cord]
                    x_out[n,f,i,j] = np.sum(window * w[f]) + b[f]
    #print((N, filters, x_pad_H - kernel_size, x_pad_W - kernel_size))
    return x_out,(w,b)            
    
def conv_backword(x_input,dy,params,kernel_size,pad,filters):
    (w,b) = params
    x_pad = np.pad(x_input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_pad = np.zeros_like(np.pad(x_input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant'))
    N, C, H, W = x_input.shape
    x_pad_H,x_pad_W = x_pad.shape[2:]
    
    dx = np.zeros_like(x_input)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    for n in range(N):       # ith image
        for f in range(filters):   # fth filter
            for i in range(x_pad_H - kernel_size + 1):
                for j in range(x_pad_W - kernel_size + 1):
                    begin_x_cord = i
                    end_x_cord = i + kernel_size
                    begin_y_cord = j 
                    end_y_cord = j + kernel_size
                    window = x_pad[n,:,begin_x_cord:end_x_cord,begin_y_cord:end_y_cord]
                    db[f] += dy[n, f, i, j]
                    dw[f] += dy[n, f, i, j] * window
                    dx_pad[n,:,begin_x_cord:end_x_cord,begin_y_cord:end_y_cord] += w[f] * dy[n, f, i, j]
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]
    return dx,(dw,db)


# In[495]:


inputx = np.random.randn(2,1,3,3)


# In[496]:


conved_x,conv_param = conv_forward(inputx,3,1,3)
dx,(dw,db) = conv_backword(inputx,2 * conved_x,conv_param,3,1,3)


# In[497]:


db


# In[498]:


conv_param[1]


# In[499]:


lr = 0.001
log = []
for i in range(3000):
    conved_x,_ = conv_forward(inputx,3,1,3,conv_param)
    dx,(dw,db) = conv_backword(inputx,2 * conved_x,conv_param,3,1,3)
    
    (w,b) = conv_param
    w -= dw * lr
    b -= db * lr
    
    loss = np.sum(conved_x * conved_x)
    log.append(loss)


# In[501]:


pd.DataFrame(log[:]).plot()


# # NOW MAGIC TIME

# In[502]:


def conv_forward(inputx,kernel_size,pad,filters,params = None):
    # inputx should be like (batch_number,channel,width,height)
    N, C, H, W = inputx.shape
    if params is None:
        w = np.random.randn(filters,C,kernel_size,kernel_size)
        b = np.random.randn(filters)
    else:
        (w,b) = params
    
    x_pad = np.pad(inputx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    x_pad_H,x_pad_W = x_pad.shape[2:]
    
    
    x_out = np.zeros((N, filters, x_pad_H - kernel_size + 1, x_pad_W - kernel_size + 1))
    for n in range(N):
        for f in range(filters):
            for i in range(x_pad_H - kernel_size + 1):
                for j in range(x_pad_W - kernel_size + 1):
                    begin_x_cord = i
                    end_x_cord = i + kernel_size
                    begin_y_cord = j 
                    end_y_cord = j + kernel_size
                    window = x_pad[n,:,begin_x_cord:end_x_cord,begin_y_cord:end_y_cord]
                    x_out[n,f,i,j] = np.sum(window * w[f]) + b[f]
    #print((N, filters, x_pad_H - kernel_size, x_pad_W - kernel_size))
    return x_out,(w,b)            
    
def conv_backword(x_input,dy,params,kernel_size,pad,filters):
    (w,b) = params
    x_pad = np.pad(x_input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_pad = np.zeros_like(np.pad(x_input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant'))
    N, C, H, W = x_input.shape
    x_pad_H,x_pad_W = x_pad.shape[2:]
    
    dx = np.zeros_like(x_input)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    for n in range(N):       # ith image
        for f in range(filters):   # fth filter
            for i in range(x_pad_H - kernel_size + 1):
                for j in range(x_pad_W - kernel_size + 1):
                    begin_x_cord = i
                    end_x_cord = i + kernel_size
                    begin_y_cord = j 
                    end_y_cord = j + kernel_size
                    window = x_pad[n,:,begin_x_cord:end_x_cord,begin_y_cord:end_y_cord]
                    db[f] += dy[n, f, i, j]
                    dw[f] += dy[n, f, i, j] * window
                    dx_pad[n,:,begin_x_cord:end_x_cord,begin_y_cord:end_y_cord] += w[f] * dy[n, f, i, j]
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]
    return dx,(dw,db)


# In[444]:


import pickle
with open('../pre_CNN/xy.pkl','rb') as whdl:
    (IM_train,y_train,IM_test) = pickle.load(whdl)


# In[503]:


IM_train[0].shape


# In[504]:


plt.imshow(IM_train[188])


# In[505]:


target = np.expand_dims(np.transpose(IM_train[188],[2,0,1]),0)


# In[506]:


target.shape


# In[507]:


inputx = np.random.randn(1,1,100,100)


# In[508]:


target = target / 255


# In[509]:


inputx = np.expand_dims(np.sum(target,axis=1),1)
inputx = inputx


# In[510]:


conved_x,conv_param = conv_forward(inputx,1,0,3)
dx,(dw,db) = conv_backword(inputx,2 * conved_x,conv_param,1,0,3)


# In[511]:


showimg = (conved_x - np.min(conved_x)) / (np.max(conved_x) - np.min(conved_x))


# In[512]:


showimg.shape


# In[513]:


plt.imshow(np.transpose(showimg[0],[1,2,0]))


# In[514]:


lr = 0.1
log = []
norm = len(inputx.reshape(-1))
for i in range(100):
    conved_x,_ = conv_forward(inputx,1,0,3,conv_param)
    dx,(dw,db) = conv_backword(inputx,(2 * conved_x - 2 * target) / norm,conv_param,1,0,3)
    
    (w,b) = conv_param
    w -= dw * lr
    b -= db * lr
    
    loss = np.sum((target - conved_x) *(target -  conved_x))  / norm
    log.append(loss)
    print(i,loss)


# In[515]:


resultimg = (conved_x - np.min(conved_x)) / (np.max(conved_x) - np.min(conved_x))
resultimg = resultimg
plt.imshow(np.transpose(resultimg[0],[1,2,0]))


# In[516]:


resultimg = (target - np.min(target)) / (np.max(target) - np.min(target))
plt.imshow(np.transpose(resultimg[0],[1,2,0]))


# In[ ]:




