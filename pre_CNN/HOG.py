#!/usr/bin/env python
# coding: utf-8
# In[70]:
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
sys.path.append('common/')
import python_utils

# In[72]:
import pickle
with open('data/xy.pkl','rb') as whdl:
    (IM_train,y_train,IM_test) = pickle.load(whdl)


# In[74]:
len(IM_train),len(y_train)

# In[75]:
# # 定义两个通用函数
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted
def calculate_histogram(array,weights,bins=9):
    bins_range = (-75, 75)
    hist,_ = np.histogram(array,bins=bins,range=bins_range,weights=weights)
    return hist

# In[77]:
sample_image = IM_train[0]
plt.imshow(np.asarray(sample_image,dtype=np.uint8))

# In[78]:
sample_image.shape

#郭助教 08:32 PM
#一方面。因为rgb转灰度的最好视觉效果并不是三者直接平均。
#另外就是手动average可能会有精度损失。
#可以看下这个
#https://www.cnblogs.com/carekee/articles/3629964.html
plt.imshow(np.average(sample_image, axis=2), cmap='gray')

# In[85]:
# # HOG特征第一步：转灰度图
sample_image = cv2.cvtColor(sample_image,cv2.COLOR_RGB2GRAY) # # HOG特征第一步：转灰度图

# In[87]:
plt.imshow(sample_image,cmap='gray')


# In[88]:
sample_image.reshape(-1).shape

# In[90]:
# # HOG特征第二步：图片数值取平方根, will increase robustness in image recongnition when lighter color
sample_image = sample_image ** 0.5

# In[92]:
plt.imshow(sample_image,cmap='gray')

# In[94]:
# # HOG特征第二步（1）：计算X和Y方向梯度
import pandas as pd
plt.imshow(translate(sample_image, -20, 0))  # image, x move, y move
# In[100]:
# 平移做kernal, 就不用做捲積了
# -1 , 0, 1
GX = translate(sample_image, 1, 0) - translate(sample_image, -1, 0)
# -1
# 0
# 1
GY = translate(sample_image, 0, 1) - translate(sample_image, 0, -1)
# In[101]:
plt.subplot(1,2,1)
plt.imshow(GX,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(GY,cmap='gray') #頭髮是縱向的，所以縱向梯度大


# # HOG特征第二步（2）：计算每个点梯度的角度和大小

# In[102]:
delta_G = np.sqrt(GX ** 2 + GY ** 2)   # 梯度大小
angle = np.arctan(GY / GX) / np.pi * 180 # 每個點梯度的方向


# In[103]:
plt.imshow(delta_G,cmap='gray')
plt.imshow(angle)

# In[104]:
delta_G.shape,angle.shape

# In[107]:
# # HOG特征第四步： 计算特征直方图
# The 100 x 100 picture is separated into 4 x 4 areas (16 areas), each area has 9 features (bins in histogram).
# put each 25 x 25 into 9 features.
features = np.zeros((4,4,9))
for i in range(0, 100, 25):
    for j in range(0, 100, 25):
        features[i // 25,j // 25] = calculate_histogram(angle[i:i + 25, j:j + 25], delta_G[i:i + 25, j:j + 25])


# In[109]:
# 9 bins
features.shape

# In[114]:
features[0,0]

# In[115]:
# # HOG特征第五步： 归一化
# after this 2nd sliding window 2 x 2, we have 3 x 3, and retain each small block's 9 feature, so a block in 3 x 3 has 2*2*9 features.
one_feature = []
for i in range(3):
    for j in range(3):
        mat_norm = features[i:i + 2,j:j + 2] # (2 * 2 * 9) - > (36)
        mag = np.linalg.norm(mat_norm)
        arr_list = (mat_norm / mag).flatten().tolist()  # flatten == reshape(-1)
        one_feature += arr_list

# In[116]:

# 3 x 3 sliding windows
len(one_feature)


# In[117]:
# # 整个过程review
plt.figure(figsize=(20,10))

sample_image = IM_train[-200]
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

# step 5 calcalute features, histogram, 4x4小圖, each 圖求個bin9的直方圖
features = np.zeros((4,4,9))
for i in range(0,100,25):
    for j in range(0,100,25):
        features[i // 25, j // 25] = calculate_histogram(angle[i:i + 25, j:j + 25],delta_G[i:i + 25, j:j + 25])

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
plt.imshow(np.asarray(one_feature).reshape(9, -1))

# In[118]:
# # 定义抽取特征函数
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


# In[119]:
#histr = cv2.calcHist([IM_train[0]],[1],None,[256],[0,256])
plt.subplot(2,1,1)
plt.imshow(IM_train[0])
plt.subplot(2,1,2)
plt.plot(extrace_feature(IM_train[0]))


# In[120]:
# # 抽取特征
from collections import Counter

# In[121]:
Counter(y_train)

# In[122]:
# pb = python_utils.ProgressBar(worksum=len(IM_train))
# pb.startjob()
features = []
targets = []
for one_image, one_target in zip(IM_train, y_train):
    one_feature = []
    if len(one_image.shape) != 3:
        continue
    if one_target == 0 and random.random() > 0.2:
        # pb.complete(1)
        continue
    features.append(extrace_feature(one_image))
    targets.append(one_target)
    # pb.complete(1)


# In[123]:
len(features),len(targets)

# In[124]:
Counter(targets)

# In[125]:
train_x,val_x = features[:-2000], features[-2000:]
train_y,val_y = targets[:-2000], targets[-2000:]

# In[126]:
import sklearn
from sklearn.svm import SVC

# In[127]:
svm = SVC(probability=True)

# In[128]:
train_x = np.nan_to_num(train_x, 0)

# In[129]:
svm_model = svm.fit(train_x,train_y)

# In[130]:
val_x = np.nan_to_num(val_x,0)

# In[131]:
val_pred = svm_model.predict_proba(val_x)[:,1]

# In[132]:
import pandas as pd

# In[133]:
pd.DataFrame(val_pred).hist(bins=50)



# In[137]:
# # 查看AUC
from sklearn import metrics

# In[138]:
fpr, tpr, thresholds = metrics.roc_curve(val_y, val_pred, pos_label = 1)

# In[139]:
metrics.auc(fpr, tpr) # a lot better!

# In[140]:
sample_image = np.copy(IM_test[0])
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
# need to do NMS to suppress other rectangles

# In[68]:
result
