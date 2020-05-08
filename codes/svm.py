#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import math
import pandas as pd
import os
import wave
import pylab
from os import listdir
from os.path import isfile,join
import random
import ntpath
from scipy.io.wavfile import write
from sklearn.svm import SVC
from sklearn import svm
import pickle


# In[2]:


spec_path = "train_spectrogram/spec/"
labels = ["zero","one","two","three","four","five","six","seven","eight","nine"]
spec_folders = [spec_path+l+"/" for l in labels]
spec_files = [[i+f for f in listdir(i) if isfile(join(i,f))] for i in spec_folders]


# In[3]:


data = []


# In[4]:


for i in range(len(spec_files)):
    print("FOLDER", i)
    for j in range(len(spec_files[i])):
        data.append(np.asarray(pd.read_csv(spec_files[i][j],header=None)).flatten().tolist() + [i])
        if j%100==0:
            print(j/10,"%","completed")
#         break
#     break


# In[5]:


data = np.asarray(data)
np.random.shuffle(data)


# In[7]:


np.savetxt("all_spec.csv",data,delimiter=',')


# In[8]:


backup_data = data


# In[9]:


data = backup_data


# In[10]:


data.shape


# In[11]:


data = data[~np.isinf(data).any(axis=1)]


# In[12]:


train_data = data[:,:4864]
train_labels = data[:,4864].astype(int)


# In[13]:


train_data.max()


# In[14]:


lin_clf = svm.LinearSVC()
lin_clf.fit(train_data,train_labels)


# In[ ]:


fn = "svm_model.sav"
pickle.dump(lin_clf,open(fn,'wb'))


# In[24]:


val_path = "test_spectrogram/spec/"
val_folders = [val_path+v+"/" for v in listdir(val_path)]
val_files = [[v+f for f in listdir(v) if isfile(join(v,f))] for v in val_folders]


# In[37]:


val_data = []
for i in range(len(val_files)):
    print("FOLDER", i)
    for j in range(len(val_files[i])):
        val_data.append(np.asarray(pd.read_csv(val_files[i][j],header=None)).flatten().tolist() + [i])
        if j%100==0:
            print(j/10,"%","completed")
#         break
#     break


# In[38]:


val_data = np.asarray(val_data)
np.random.shuffle(val_data)


# In[43]:


val_data = val_data[~np.isinf(val_data).any(axis=1)]


# In[44]:


val_X = val_data[:,:4864]
val_y = val_data[:,4864].astype(int)


# In[46]:


val_data.shape


# In[45]:


lin_clf.score(val_X,val_y)


# In[82]:


train_labels = train_labels.reshape(10000,1)


# In[51]:


clf = SVC(kernel='rbf',decision_function_shape='ovo')
clf.fit(train_data,train_labels)


# In[52]:


clf.score(val_X,val_y)


# In[53]:


clf2 = SVC(kernel='poly',decision_function_shape='ovr')
clf2.fit(train_data,train_labels)


# In[54]:


clf2.score(val_X,val_y)


# In[95]:


data[~np.isinf(data).any(axis=1)].shape


# In[15]:





# In[17]:





# In[19]:


lin_clf.score(train_data,train_labels)


# In[ ]:




