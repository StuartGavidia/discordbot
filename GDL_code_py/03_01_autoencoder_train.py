#!/usr/bin/env python
# coding: utf-8

# # Autoencoder

# In[ ]:


import os

from utils.loaders import load_mnist
from models.AE import Autoencoder


# ## Set parameters

# In[ ]:


# run params
SECTION = 'vae'
RUN_ID = '0001'
DATA_NAME = 'digits'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

MODE =  'build' #'load' #


# ## Load the data

# In[ ]:


(x_train, y_train), (x_test, y_test) = load_mnist()


# ## Define the structure of the neural network

# In[ ]:


AE = Autoencoder(
    input_dim = (28,28,1)
    , encoder_conv_filters = [32,64,64, 64]
    , encoder_conv_kernel_size = [3,3,3,3]
    , encoder_conv_strides = [1,2,2,1]
    , decoder_conv_t_filters = [64,64,32,1]
    , decoder_conv_t_kernel_size = [3,3,3,3]
    , decoder_conv_t_strides = [1,2,2,1]
    , z_dim = 2
)

if MODE == 'build':
    AE.save(RUN_FOLDER)
else:
    AE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))


# In[ ]:


AE.encoder.summary()


# In[ ]:


AE.decoder.summary()


# ## Train the autoencoder

# In[ ]:


LEARNING_RATE = 0.0005
BATCH_SIZE = 32
INITIAL_EPOCH = 0


# In[ ]:


AE.compile(LEARNING_RATE)


# In[ ]:


AE.train(     
    x_train[:1000]
    , batch_size = BATCH_SIZE
    , epochs = 200
    , run_folder = RUN_FOLDER
    , initial_epoch = INITIAL_EPOCH
)


# In[ ]:





# In[ ]:




