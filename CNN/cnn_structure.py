#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras


# In[5]:


vgg = keras.applications.vgg16.VGG16(weights=None)


# In[6]:


vgg.summary()


# In[9]:


resnet = keras.applications.resnet50.ResNet50(weights=None)


# In[10]:


resnet.summary()


# In[ ]:




