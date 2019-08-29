#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import random
import time
import numpy as np
from scipy import misc

from matplotlib import pyplot as plt
from scipy import misc
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import os
import sys
sys.path.append('../common/')
import utils


# In[2]:


import pickle


# In[3]:


import pickle
with open('./xy.pkl','rb') as whdl:
    (IM_train,y_train,IM_test) = pickle.load(whdl)


# In[4]:


len(IM_train),len(y_train)


# In[66]:


def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted
def calculate_histogram(array,weights,bins=9):
    bins_range = (-75, 75)
    hist,_ = np.histogram(array,bins=bins,range=bins_range,weights=weights)
    return hist


# In[171]:


plt.figure(figsize=(20,10))

sample_image = IM_train[0]
plt.subplot(2,5,1)
plt.imshow(sample_image)


# STEP 1 convert to gray
sample_image = cv2.cvtColor(sample_image,cv2.COLOR_RGB2GRAY)
plt.subplot(2,5,2)
plt.imshow(sample_image,cmap='gray')


# STEP 2 normalize
sample_image = sample_image ** 0.5
plt.subplot(2,5,3)
plt.imshow(sample_image,cmap='gray')

# STEP 3 calculate GX,GY
GX = translate(sample_image,1,0) - translate(sample_image,-1,0)
plt.subplot(2,5,4)
plt.imshow(GX,cmap='gray')

GY = translate(sample_image,0,1) - translate(sample_image,0,-1)
plt.subplot(2,5,5)
plt.imshow(GY,cmap='gray')

# STEP 4 calculate DELTA G and angle
delta_G = np.sqrt(GX ** 2 + GY ** 2)
plt.subplot(2,5,6)
plt.imshow(delta_G,cmap='gray')

angle = np.arctan(GY / GX) / np.pi * 180
plt.subplot(2,5,7)
plt.imshow(angle)

delta_G[angle == 0] = 0

# step 5 calcalute features
features = np.zeros((4,4,9))
for i in range(0,100,25):
    for j in range(0,100,25):
        features[i // 25,j // 25] = calculate_histogram(angle[i:i + 25,j:j + 25],delta_G[i:i + 25,j:j + 25])
        
plt.subplot(2,5,8)
plt.imshow(features.reshape(4,36))

# step 6 local normalize
one_feature = []
for i in range(3):
    for j in range(3):
        mat_norm = features[i:i + 2,j:j + 2]
        mag = np.linalg.norm(mat_norm)
        arr_list = (mat_norm / mag).flatten().tolist()
        one_feature += arr_list
plt.subplot(2,5,9)
plt.imshow(np.asarray(one_feature).reshape(9,-1))


# In[94]:


len(one_feature)


# In[95]:


# import pandas as pd
pd.DataFrame(angle.reshape(-1)).hist(bins=90)
pd.DataFrame(delta_G.reshape(-1)).hist(bins=16)


# In[103]:



def extrace_feature(sample_image):
    sample_image = cv2.cvtColor(sample_image,cv2.COLOR_RGB2GRAY)



    # STEP 2 normalize
    sample_image = sample_image ** 0.5


    # STEP 3 calculate GX,GY
    GX = translate(sample_image,1,0) - translate(sample_image,-1,0)


    GY = translate(sample_image,0,1) - translate(sample_image,0,-1)


    # STEP 4 calculate DELTA G and angle
    delta_G = np.sqrt(GX ** 2 + GY ** 2)


    angle = np.arctan(GY / GX) / np.pi * 180


    delta_G[angle == 0] = 0

    # step 5 calcalute features
    features = np.zeros((4,4,9))
    for i in range(0,100,25):
        for j in range(0,100,25):
            features[i // 25,j // 25] = calculate_histogram(angle[i:i + 25,j:j + 25],delta_G[i:i + 25,j:j + 25])

    # step 6 local normalize
    one_feature = []
    for i in range(3):
        for j in range(3):
            mat_norm = features[i:i + 2,j:j + 2]
            mag = np.linalg.norm(mat_norm)
            arr_list = (mat_norm / mag).flatten().tolist()
            one_feature += arr_list
    return one_feature


# In[105]:


#histr = cv2.calcHist([IM_train[0]],[1],None,[256],[0,256])

plt.subplot(2,1,1)
plt.imshow(IM_train[188])
plt.subplot(2,1,2)
plt.plot(extrace_feature(IM_train[188]))


# In[107]:


np.where(np.asarray(y_train)==1)[0][:10]


# In[111]:


from collections import Counter


# In[115]:


Counter(y_train)


# In[118]:



pb = utils.ProgressBar(worksum=len(IM_train))
pb.startjob()
features = []
targets = []
for one_image,one_target in zip(IM_train,y_train):
    one_feature = []
    if len(one_image.shape) != 3:
        continue
    if one_target == 0 and random.random() > 0.2:
        pb.complete(1)
        continue
    features.append(extrace_feature(one_image))
    targets.append(one_target)
    pb.complete(1)


# In[119]:


len(features),len(targets)


# In[120]:


train_x,val_x = features[:-2000], features[-2000:]
train_y,val_y = targets[:-2000], targets[-2000:]


# In[122]:


import sklearn
from sklearn.svm import SVC


# In[130]:


svm = SVC(probability=True)


# In[131]:


train_x = np.nan_to_num(train_x,0)


# In[132]:


svm_model = svm.fit(train_x,train_y)


# In[133]:


val_x = np.nan_to_num(val_x,0)


# In[134]:


val_pred = svm_model.predict_proba(val_x)[:,1]


# In[135]:


val_pred


# In[136]:


from sklearn import metrics


# In[137]:


fpr, tpr, thresholds = metrics.roc_curve(val_y, val_pred, pos_label = 1)


# In[138]:


metrics.auc(fpr, tpr)


# In[176]:



sample_image = np.copy(IM_test[23])
plt.subplot(1,2,1)
plt.imshow(sample_image)
for i in range(0,250 - 100,10):
    for j in range(0,250 - 100,10):
        one_feature = extrace_feature(sample_image[i:i + 100,j:j + 100])
        one_feature = np.nan_to_num(one_feature,0)
        result = svm_model.predict_proba([one_feature])
        #print(result)
        if result[0][1] > 0.5:
            cv2.rectangle(sample_image,(i,j),(i + 100,j + 100),(0,0,0),3)
plt.subplot(1,2,2)
plt.imshow(sample_image)


# In[117]:


result


# In[ ]:




