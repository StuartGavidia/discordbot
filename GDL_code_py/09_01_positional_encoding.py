#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import matplotlib.pyplot as plt


# In[40]:


seq_len = 128
d_model = 512


# In[41]:


pe = np.zeros((seq_len, d_model))


# In[42]:


for pos in range(seq_len):
    for i in range(int(d_model / 2)):
        pe[pos,2*i] = np.sin(pos/(pow(10000,((2*i)/d_model))))
        pe[pos,2*i+1] = np.cos(pos/(pow(10000,((2*i)/d_model))))
        


# In[52]:


fig= plt.figure(figsize=(20,5))
# plt.xlabel('2i (d_model = 512)')
# plt.ylabel('pos')
plt.imshow(pe, cmap = 'Greys')


# In[ ]:


enc = np.zeros((seq_len, d_model))


# In[46]:


enc = np.random.rand(seq_len, d_model)


# In[49]:


fig= plt.figure(figsize=(20,5))
plt.imshow(enc, cmap = 'Greys')


# In[50]:


out = enc + pe


# In[51]:


fig= plt.figure(figsize=(20,5))
plt.imshow(out, cmap = 'Greys')


# In[ ]:




