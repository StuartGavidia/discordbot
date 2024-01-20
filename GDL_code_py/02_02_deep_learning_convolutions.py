#!/usr/bin/env python
# coding: utf-8

# # imports

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import rescale,resize


# # original image input

# In[ ]:


im = rgb2gray(data.coffee())
im = resize(im, (64,64))
print(im.shape)

plt.axis('off')
plt.imshow(im, cmap = 'gray');


# # horizontal edge filter

# In[ ]:


filter1 = np.array([
    [ 1,  1,  1],
    [ 0,  0,  0],
    [-1, -1, -1]
])

new_image = np.zeros(im.shape)

im_pad = np.pad(im, 1, 'constant')

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        try:
            new_image[i,j] =             im_pad[i-1,j-1] * filter1[0,0] +             im_pad[i-1,j] * filter1[0,1] +             im_pad[i-1,j+1] * filter1[0,2] +             im_pad[i,j-1] * filter1[1,0] +             im_pad[i,j] * filter1[1,1] +             im_pad[i,j+1] * filter1[1,2] +            im_pad[i+1,j-1] * filter1[2,0] +             im_pad[i+1,j] * filter1[2,1] +             im_pad[i+1,j+1] * filter1[2,2] 
        except:
            pass

plt.axis('off')
plt.imshow(new_image, cmap='Greys');


# # vertical edge filter

# In[ ]:


filter2 = np.array([
    [ -1,  0,  1],
    [ -1,  0,  1],
    [ -1,  0,  1]
])

new_image = np.zeros(im.shape)

im_pad = np.pad(im,1, 'constant')

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        try:
            new_image[i,j] =             im_pad[i-1,j-1] * filter2[0,0] +             im_pad[i-1,j] * filter2[0,1] +             im_pad[i-1,j+1] * filter2[0,2] +             im_pad[i,j-1] * filter2[1,0] +             im_pad[i,j] * filter2[1,1] +             im_pad[i,j+1] * filter2[1,2] +            im_pad[i+1,j-1] * filter2[2,0] +             im_pad[i+1,j] * filter2[2,1] +             im_pad[i+1,j+1] * filter2[2,2] 
        except:
            pass

plt.axis('off')
plt.imshow(new_image, cmap='Greys');


# # horizontal edge filter with stride 2

# In[ ]:


filter1 = np.array([
    [ 1,  1,  1],
    [ 0,  0,  0],
    [-1, -1, -1]
])

stride = 2

new_image = np.zeros((int(im.shape[0] / stride), int(im.shape[1] / stride)))

im_pad = np.pad(im,1, 'constant')

for i in range(0,im.shape[0],stride):
    for j in range(0,im.shape[1],stride):
        try:
            new_image[int(i/stride),int(j/stride)] =             im_pad[i-1,j-1] * filter1[0,0] +             im_pad[i-1,j] * filter1[0,1] +             im_pad[i-1,j+1] * filter1[0,2] +             im_pad[i,j-1] * filter1[1,0] +             im_pad[i,j] * filter1[1,1] +             im_pad[i,j+1] * filter1[1,2] +            im_pad[i+1,j-1] * filter1[2,0] +             im_pad[i+1,j] * filter1[2,1] +             im_pad[i+1,j+1] * filter1[2,2] 
        except:
            pass

plt.axis('off')
plt.imshow(new_image, cmap='Greys');


# # vertical edge filter with stride 2

# In[ ]:


filter2 = np.array([
    [ -1,  0,  1],
    [ -1,  0,  1],
    [ -1,  0,  1]
])

stride = 2

new_image = np.zeros((int(im.shape[0] / stride), int(im.shape[1] / stride)))

im_pad = np.pad(im,1, 'constant')

for i in range(0,im.shape[0],stride):
    for j in range(0,im.shape[1],stride):
        try:
            new_image[int(i/stride),int(j/stride)] =             im_pad[i-1,j-1] * filter2[0,0] +             im_pad[i-1,j] * filter2[0,1] +             im_pad[i-1,j+1] * filter2[0,2] +             im_pad[i,j-1] * filter2[1,0] +             im_pad[i,j] * filter2[1,1] +             im_pad[i,j+1] * filter2[1,2] +            im_pad[i+1,j-1] * filter2[2,0] +             im_pad[i+1,j] * filter2[2,1] +             im_pad[i+1,j+1] * filter2[2,2] 
        except:
            pass

plt.axis('off')
plt.imshow(new_image, cmap='Greys');


# In[ ]:




