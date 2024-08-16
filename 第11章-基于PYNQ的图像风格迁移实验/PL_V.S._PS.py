#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
from pynq import Overlay
import numpy as np
# from pynq import Xlnk
from pynq import allocate
import struct
from imageio import imread
import cv2
def RunConv(conv,Kx,Ky,Sx,Sy,mode,relu_en,feature_in,W,bias,feature_out):
    write_time_s = time.time()
    conv.write(0x10,feature_in.shape[2]);
    conv.write(0x18,feature_in.shape[0]);
    conv.write(0x20,feature_in.shape[1]);
    conv.write(0x28,feature_out.shape[2]);
    write_time_2 = time.time()
    print("time 2",write_time_2-write_time_s)
    conv.write(0x30,Kx);
    conv.write(0x38,Ky);
    conv.write(0x40,Sx);
    conv.write(0x48,Sy);
    write_time_3 = time.time()
    print("time 3",write_time_3-write_time_2)
    conv.write(0x50,mode);
    conv.write(0x58,relu_en);
    conv.write(0x60,feature_in.physical_address);
    conv.write(0x68,W.physical_address);
    write_time_4 = time.time()
    print("time 4",write_time_4-write_time_3)
    conv.write(0x70,bias.physical_address);
    conv.write(0x78,feature_out.physical_address);
    conv.write(0, (conv.read(0)&0x80)|0x01 );
    write_time_5 = time.time()
    print("time 5",write_time_5-write_time_4)
    tp=conv.read(0)
    print(tp);
    
    while not ((tp>>1)&0x1):
        tp=conv.read(0);
        print(tp);
    write_time_6 = time.time()
    print("time 6",write_time_6-write_time_5)
    #print(tp);



# In[31]:


import torch
import torch.nn as nn
i = 1
x = torch.randn(1,3,32*i,32*i)
# print(x[0,:,:])
a = nn.ReflectionPad2d(padding=(3))
b = a(x)


# In[3]:


ol=Overlay("/home/xilinx/jupyter_notebooks/mnist/ai.bit")
ol.ip_dict
ol.download()
conv=ol.Conv_0
pool=ol.Pool_0
print("Overlay download finish");


# In[29]:


import time
# i= 20
W = np.arange(9).reshape(3,3,1,1)
size = 10
X = np.arange(size*size).reshape(size,size,1,1)
Bias = np.zeros(1)
# feature_out = allocate(shape=(640,640,32),cacheable=0,dtype=np.float32)
x = allocate(shape=(size,size,1,1),cacheable=0,dtype=np.float32)
w = allocate(shape=(3,3,1,1),cacheable=0,dtype=np.float32)
bias = allocate(shape=(1,),cacheable=0,dtype=np.float32)
out_size = int(size-3+1)
feature_out = allocate(shape=(out_size,out_size,1),cacheable=0,dtype=np.float32)
np.copyto(w,W)
np.copyto(x,X)
np.copyto(bias,Bias)
# pic_2 = pic
mode = 0
relu_en = 0
s = time.time()
RunConv(conv,3,3,1,1,mode,relu_en,x,w,bias,feature_out)
e = time.time()
print("When run in PL with input size ={},take {:.4f}s.".format(X.shape,e-s))


# In[30]:


feature_out


# In[32]:


conv_ps = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, groups=1, bias=False)
conv_ps.weight.data = torch.from_numpy(np.arange(9).reshape(1,1,3,3)*1.0)
x = torch.from_numpy(np.arange(size*size).reshape(1,1,size,size)*1.0)
s = time.time()
b = conv_ps(x)
e = time.time()
print("When run in PS with input size ={},take {:.4f}s.".format(X.shape,e-s))
# print(e-s)

