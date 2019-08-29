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
import skimage
from skimage.feature import local_binary_pattern


# In[2]:


import pickle


# In[18]:


import pickle
with open('./xy.pkl','rb') as whdl:
    (IM_train,y_train,IM_test) = pickle.load(whdl)


# In[19]:


len(IM_train),len(y_train)


# In[20]:


from collections import Counter


# In[21]:


Counter(y_train)


# In[22]:



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
    features.append(one_image)
    targets.append(one_target)
    pb.complete(1)


# In[23]:


Counter(targets)


# In[24]:


len(features),len(targets)


# In[25]:


train_x,val_x = features[:-1000], features[-1000:]
train_y,val_y = targets[:-1000], targets[-1000:]


# In[26]:


train_x[0].shape


# In[16]:


plt.imshow(train_x[0])


# In[28]:


len(train_x[0].reshape(-1))


# In[33]:


plt.imshow(np.asarray(train_x[-i] * 0.5,np.uint8)) 


# In[15]:


plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(4,5,i + 1)
    plt.imshow(train_x[-i])
    plt.title(train_y[-i])


# In[13]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# In[14]:


import keras


# In[15]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import merge
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model
from keras.models import Sequential


# In[16]:



model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=train_x[0].shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[17]:


model.summary()


# In[18]:


#opt = keras.optimizers.SGD(lr=0.01, decay=1e-6)
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


# In[19]:


# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[20]:


train_x = np.asarray(train_x)


# In[21]:


val_x = np.asarray(val_x)


# In[22]:


train_x.shape,val_x.shape


# In[23]:


model.fit(train_x, train_y,
              batch_size=128,
              epochs=7,
              validation_data=(val_x, val_y),
              shuffle=True)


# In[37]:


model.save_weights('../data/keras_model')


# In[38]:


get_ipython().system('ls -l ../data/keras_model')


# In[24]:


val_pred = model.predict_proba(val_x)


# In[25]:


val_pred


# In[26]:


from sklearn import metrics


# In[27]:


fpr, tpr, thresholds = metrics.roc_curve(val_y, val_pred[:,0], pos_label = 1)


# In[28]:


metrics.auc(fpr, tpr)


# In[36]:



sample_image = np.copy(IM_test[23])
plt.subplot(1,2,1)
plt.imshow(sample_image)
for i in range(0,250 - 100,10):
    for j in range(0,250 - 100,10):
        one_feature = sample_image[i:i + 100,j:j + 100]
        result = model.predict_proba(np.asarray([one_feature]),verbose=False)
        #print(result)
        if result[0][0] > 0.9:
            cv2.rectangle(sample_image,(i,j),(i + 100,j + 100),(0,0,0),3)
plt.subplot(1,2,2)
plt.imshow(sample_image)


# In[34]:


result


# In[ ]:




