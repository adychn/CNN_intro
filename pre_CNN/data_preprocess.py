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


# In[60]:


import os
import sys
sys.path.append('../common/')
import utils


# In[3]:


dirs = os.listdir('../data/CASIA-WebFace/')


# In[4]:


valid_dirs = random.sample(dirs,30)


# In[5]:


valid_dirs = ['../data/CASIA-WebFace/{}'.format(x) for x in valid_dirs]


# In[6]:


valid_dirs


# In[7]:


picurls = []
for one_dir in valid_dirs:
    for one_file in os.listdir(one_dir):
        picurls.append("{}/{}".format(one_dir,one_file))


# In[8]:


len(picurls)


# In[9]:


facecascade = cv2.CascadeClassifier('../data/haar_cascade_frontalface_default.xml')


# In[56]:


sample_image = misc.imread(picurls[19])
plt.imshow(sample_image)


# In[57]:


x,y,w,h = facecascade.detectMultiScale(sample_image,scaleFactor=1.1,minNeighbors=5)[0]


# In[58]:


plt.imshow(cv2.rectangle(sample_image,(x,y),(x+w,y+h),(0,0,0),4))


# In[73]:


failcount = 0
pb = utils.ProgressBar(worksum=len(picurls))
pb.startjob()

resultdic = {}

for one_url in picurls:
    one_img = misc.imread(one_url)
    try:
        x,y,w,h = facecascade.detectMultiScale(one_img,scaleFactor=1.1,minNeighbors=5)[0]
        xm = x + int(w / 2)
        ym = y + int(h / 2)
        resultdic[one_url] = (xm - 50,ym - 50,xm + 50,ym + 50)
        pb.complete(1)
    except:
        failcount += 1
        pb.complete(1)
        continue


# In[74]:


plt.imshow(one_img)


# In[75]:


plt.imshow(cv2.rectangle(one_img,(xm-50,ym-50),(xm+50,ym+50),(0,0,0),4))


# In[76]:


import pickle


# In[77]:


with open('training_data.pkl','wb') as whdl:
    pickle.dump(resultdic,whdl)


# In[78]:


resultdic


# In[ ]:




