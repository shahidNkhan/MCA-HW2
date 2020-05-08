#!/usr/bin/env python
# coding: utf-8

# In[62]:


import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import math
import os
import wave
import pylab
from os import listdir
from os.path import isfile,join
import random
import ntpath
from scipy.io.wavfile import write
# from scikits.audiolab import wavread


# In[63]:


def get_xns(ts,n):
    mag = []
    lim = n//2
    for iteration in range(0,lim): # Nyquest Limit
        ks = np.zeros((n)).astype(int)
        for i in range(n):
            ks[i]=i
        xn = np.abs(np.sum(ts*np.exp((1j*2*np.pi*ks*iteration)/n))/n)
        mag.append(xn*2)
    return(mag)


# In[64]:


def binary_search(starts,n,NFFT):
    low = 0
    high = len(starts)-1
    ans = -1
    while(low<=high):
        mid = low + (high-low)//2
        if(starts[mid]<n-NFFT):
            low = mid + 1
        else:
            high = mid - 1
            ans = mid
    return ans


# In[65]:


def create_spectrogram(ts,n,NFFT,noverlap):
    win = NFFT - noverlap
    starts = np.zeros((n//win+1)).astype(int)
    track = 0
    for i in range(n):
        if i*win > n: break
        starts[i] = i*win
    end_i = binary_search(starts,n,NFFT)
    starts  = starts[:end_i]
    xns = []
    for start in starts:
        xns.append(get_xns(ts[start:start + NFFT],NFFT))
    specX = np.array(xns)
    # rescale the absolute value of the spectrogram as rescaling is standard
    spec = 10*np.log10(specX.T)
    return(starts,spec)


# In[66]:


def plot_spectrogram(path,filename,label,spec,ks,sample_rate, L, starts,n):
    fg = plt.figure(figsize=(24,10))
    plt_spec = plt.imshow(spec,origin='lower')
    spath = path+"plot/"+label+"/" + ntpath.basename(filename)[:-4]+".png"
    ## create ylim
    increments = spec.shape[0]/(10-1)
    ks = np.zeros(10)
    for i in range(10):
        ks[i] = i*increments
    ksHz = np.asarray(ks*sample_rate/n).astype(int)
    plt.yticks(ks,ksHz)
    plt.ylabel("Frequency (Hz)")
    total_ts_sec = n/16000
    ## create xlim
    increments = spec.shape[1]/(10-1)
    ts_spec = np.zeros(10)
    for i in range(10):
        ts_spec[i] = i*increments
    x_increment = total_ts_sec/9
    x_vals = []
    for i in range(10):
        x_vals.append(str(i*x_increment)[:4])
    plt.xticks(ts_spec,x_vals)
    plt.xlabel("Time (sec)")
    plt_title = "Spectrogram"
    plt.title(plt_title)
    plt.colorbar(None,use_gridspec=True)
#     plt.show()
    fg.savefig(spath,bbox_inches='tight')
    return(plt_spec)


# In[67]:


def get_spec_plot(filename,path,label,plot=False):
    wav = wave.open(filename, 'r')
    frames = wav.readframes(-1)
    ts = np.asarray(pylab.fromstring(frames, 'Int16'))
    sample_rate = wav.getframerate()
    wav.close()
    n = len(ts)
    ts=ts[n-6688:]
    n=6688
    total_ts_sec = n/sample_rate
    mag = get_xns(ts,n)
    Nxlim = 10
    temp = len(mag)/(Nxlim-1)
    ks = np.asarray([temp*i for i in range(Nxlim)])
    ksHz = np.asarray(ks*sample_rate/n).astype(int)
    L = 256
    noverlap = 84
    starts, spec = create_spectrogram(ts,n,L,noverlap)
    spath = path+"spec/"+label+"/" + ntpath.basename(filename)[:-4]+".csv"
    np.savetxt(spath,spec,delimiter=',')
    plot_spectrogram(path,filename,label,spec,ks,sample_rate,L, starts,n)


# In[68]:


dataset_path = "Dataset/training/"
validation_path = "Dataset/validation/"
noise_path = "Dataset/_background_noise_/"
label_folders = ["zero","one","two","three","four","five","six","seven","eight","nine"]
folder_paths = [dataset_path+i+"/" for i in label_folders]
validation_folder_paths = [validation_path+i+"/" for i in label_folders]
audio_files = [[join(fp,f) for f in listdir(fp) if isfile(join(fp,f))] for fp in folder_paths]
validation_files = [[join(fp,f) for f in listdir(fp) if isfile(join(fp,f))] for fp in validation_folder_paths]


# In[48]:


noise_files = [join(noise_path,f) for f in listdir(noise_path) if isfile(join(noise_path,f))]


# In[54]:


noises = []
for i in range(6):
    wav = wave.open(noise_files[i], 'r')
    frames = wav.readframes(-1)
    ts = np.asarray(pylab.fromstring(frames, 'Int16'))
    noises.append(ts)
    wav.close()


# In[ ]:





# In[10]:


lowest = 1000000
for i in range(10):
    print(i)
    for j in range(len(validation[i])):
        wav = wave.open(validation[i][j], 'r')
        frames = wav.readframes(-1)
        ts = np.asarray(pylab.fromstring(frames, 'Int16'))
        wav.close()
        n = len(ts)
        if(n<lowest):
            lowest = n


# In[69]:


train_x_spec = []
train_y_spec = []
for j in range(len(validation_files[0])):
    rno = random.uniform(0,1)
    filename = validation_files[0][j]
    get_spec_plot(filename,"test_spectrogram/","zero")
    if(j%100==0):
        print(j)


# In[13]:


wav = wave.open("Dataset/_background_noise_/doing_the_dishes.wav", 'r')
frames = wav.readframes(-1)
ts = np.asarray(pylab.fromstring(frames, 'Int16'))
wav.close()
n = len(ts)


# In[16]:


wav = wave.open("check.wav", 'r')
frames = wav.readframes(-1)
ts2 = np.asarray(pylab.fromstring(frames, 'Int16'))
wav.close()


# In[20]:


d = 0.1*ts[:6688]+0.5*ts2[len(ts2)-6688:]


# In[47]:


from IPython.display import Audio
Audio(0.3*ts[:6688]+0.7*ts2[len(ts2)-6688:],rate=16000)


# In[60]:


Audio(noises[5],rate=16000)


# In[ ]:




