#!/usr/bin/env python
# coding: utf-8

# In[146]:


import os
import numpy as np
import scipy
import wave
import pylab
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import matplotlib.pyplot as plt
from numpy.core.multiarray import normalize_axis_index
# import scipy.signal as ss
from os import listdir
from os.path import isfile,join
from numpy.core import swapaxes
from sys import executable
from subprocess import call
from scipy.fftpack import dct as dct_inbuilt
import random
import ntpath
from sklearn.svm import SVC
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def frame_audio(audio, FFT_size,pad_len):
    
    audio = audio.tolist()
    extra_pad1 = audio[1:pad_len+1]
    extra_pad1.reverse()
    
    frame_len = (16000 * 15) // 1000
    
    extra_pad2 = audio[-(pad_len+1):-1]
    extra_pad2 = extra_pad2[::-1]
    
    extra_pad1.extend(audio)
    extra_pad1.extend(extra_pad2)
    
    audio = np.asarray(extra_pad1)
    
    frame_num = (len(audio) - 2048) // frame_len
    frame_num += 1
    frames = np.ones((frame_num,FFT_size))
    
    for i in range(frame_num):
        start_i = i*frame_len
        end_i = start_i + FFT_size
        frames[i] = audio[start_i:end_i]
    
    return frames


# In[3]:


def create_hanning_window(FFT_size):
    FFT_size = int(FFT_size)
    a=[0.5,0.5]
    M, needs_trunc = FFT_size+1,True
    fac = np.linspace(-np.pi, np.pi, M)
    w = np.zeros(M)
    for k in range(len(a)):
        w += a[k] * np.cos(k * fac)

    window = w[:-1]
    return window


# In[4]:


def do_fft(audio_win,FFT_size,dim):
    audio_winT = audio_win.T
    cols = dim[1]
    rows = dim[0]
    end_i = rows
    audio_fft = np.ones((rows,cols),dtype=np.complex64,order='F')
    for n in range(cols):
        x = fft.fft(audio_winT[:, n])
        audio_fft[:, n] = x[:end_i]
    return audio_fft.T


# In[74]:


def get_filter_points( fmax, mel_filter_num, FFT_size):
    fmin_mel = 0
    ins = 1.0 + fmax / 700.0
    fmax_mel = 2595.0 * np.log10(ins)
    inc_i = fmax_mel/11
    mels = np.arange(0,fmax_mel+inc_i/2,inc_i)
    freqs = 700.0 * (10.0**(mels / 2595.0) - 1.0)
    return freqs


# In[75]:


def get_filters(filter_points, FFT_size):
    rows = len(filter_points)-2
    cols = FFT_size//2
    cols += 1
    filters = np.zeros((rows,cols))
    
    start_i = filter_points[0]
    end_i = filter_points[1]
    counter=0
    for i in filter_points[2:]:
        filters[counter, start_i : end_i] = np.linspace(0, 1, end_i - start_i)
        filters[counter, end_i : i] = np.linspace(1, 0, i - end_i)
        counter += 1
        start_i = end_i
        end_i = i
    return filters


# In[97]:


def get_label(i):
    if i==0:
        return "zero"
    if i==1:
        return "one"
    if i==2:
        return "two"
    if i==3:
        return "three"
    if i==4:
        return "four"
    if i==5:
        return "five"
    if i==6:
        return "six"
    if i==7:
        return "seven"
    if i==8:
        return "eight"
    if i==9:
        return "nine"


# In[141]:


def get_mfcc(filename,path,label):
    wav = wave.open(filename, 'r')
    frames = wav.readframes(-1)
    audio = np.asarray(pylab.fromstring(frames, 'Int16'))/np.max(np.abs(np.asarray(pylab.fromstring(frames, 'Int16'))))
    sample_rate = wav.getframerate()
    wav.close()
    n = len(audio)
    audio = audio[n-6688:]
    n=6688
    
    total_ts_sec = n/sample_rate

    FFT_size = 2048

    audio_framed = frame_audio(audio, FFT_size, FFT_size//2)
    window = create_hanning_window(FFT_size)
    audio_win = audio_framed * window

    dim = (FFT_size // 2 + 1,audio_win.T.shape[1])
    audio_power = np.abs(do_fft(audio_win,FFT_size,dim))
    audio_power = audio_power*audio_power

    mel_freqs = get_filter_points(8000, 10, FFT_size)
    x = (1+FFT_size) / 16000
    filter_points = np.floor(x*mel_freqs).astype(int)

    filters = get_filters(filter_points, FFT_size)

    enorm = 2.0 / (mel_freqs[2:10+2] - mel_freqs[:10])
    filters *= enorm[:, np.newaxis]

    audio_powerT = audio_power.T
    audio_filtered = filters.dot(audio_powerT)
    audio_log = 10.0 * np.log10(audio_filtered)
    audio_log.shape

    dct_filter_num = 40
    samples = np.arange(1, 2 * 10, 2) * np.pi / (2.0 * 10)
    # dct_filters = dct(samples)

    dct_filter = dct_inbuilt(audio_log)
    cepstral_coefficients = dct_filter.T.dot(audio_log)
    print("label:",label)
    spath = path+"mfcc/"+label+"/" + ntpath.basename(filename)[:-4]+".csv"
#     np.savetxt(spath,cepstral_coefficients,delimiter=',')
    
    
    
    fg = plt.figure(figsize=(15,5))
    ender = len(audio)/sample_rate
    incer = ender/(len(audio)-1)
    xps = np.arange(0, ender+incer/2, incer)
    plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
#     plt.imshow(cepstral_coefficients, aspect='auto', origin='lower');
    spath = path+"plot/"+label+"/" + ntpath.basename(filename)[:-4]+".png"
#     fg.savefig(spath,bbox_inches='tight')
    return cepstral_coefficients


# In[139]:


dataset_path = "Dataset/training/"
validation_path = "Dataset/validation/"
noise_path = "Dataset/_background_noise_/"
label_folders = ["zero","one","two","three","four","five","six","seven","eight","nine"]
folder_paths = [dataset_path+i+"/" for i in label_folders]
validation_folder_paths = [validation_path+i+"/" for i in label_folders]
audio_files = [[join(fp,f) for f in listdir(fp) if isfile(join(fp,f))] for fp in folder_paths]
validation_files = [[join(fp,f) for f in listdir(fp) if isfile(join(fp,f))] for fp in validation_folder_paths]


# In[140]:


for i in range(len(audio_files)):
    for j in audio_files[i]:
        rno = random.uniform(0,1)
        if rno<=0.1:
            label=get_label(i)
            path = "train_mfcc/"
            x = get_mfcc(j,path,label).flatten().tolist()
            x.append(i)
            break


# In[127]:


backup = np.asarray(data)


# In[131]:


data=backup


# In[142]:


np.random.shuffle(data)


# In[143]:


data.shape


# In[144]:


X = data[:,:1120]
y = data[:,1120]


# In[177]:


train_X = X[:925]
train_y = y[:925]

test_X = X[925:]
text_y = y[925:]


# In[178]:


test_X


# In[179]:


clf2 = SVC(kernel='poly',decision_function_shape='ovr')
clf2.fit(train_X,train_y)


# In[180]:


clf2.score(test_X,text_y)


# In[ ]:




