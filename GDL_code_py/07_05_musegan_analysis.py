#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
import numpy as np

from music21 import midi
from music21 import note, stream, duration
from music21 import converter

from models.MuseGAN import MuseGAN

from utils.loaders import load_music

from keras.models import load_model


# In[2]:


# run params
SECTION = 'compose'
RUN_ID = '0016'
DATA_NAME = 'chorales'
FILENAME = 'Jsb16thSeparated.npz'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])


# ## data

# In[3]:


BATCH_SIZE = 64
n_bars = 2
n_steps_per_bar = 16
n_pitches = 84
n_tracks = 4

data_binary, data_ints, raw_data = load_music(DATA_NAME, FILENAME, n_bars, n_steps_per_bar)
# data_binary = np.squeeze(data_binary)


# In[4]:


gan = MuseGAN(input_dim = data_binary.shape[1:]
        , critic_learning_rate = 0.001
        , generator_learning_rate = 0.001
        , optimiser = 'adam'
        , grad_weight = 10
        , z_dim = 32
        , batch_size = BATCH_SIZE
        , n_tracks = n_tracks
        , n_bars = n_bars
        , n_steps_per_bar = n_steps_per_bar
        , n_pitches = n_pitches
        )


# In[5]:


gan.load_weights(RUN_FOLDER, None)


# In[6]:


gan.generator.summary()


# In[7]:


gan.critic.summary()


# # view sample score

# In[51]:


chords_noise = np.random.normal(0, 1, (1, gan.z_dim))
style_noise = np.random.normal(0, 1, (1, gan.z_dim))
melody_noise = np.random.normal(0, 1, (1, gan.n_tracks, gan.z_dim))
groove_noise = np.random.normal(0, 1, (1, gan.n_tracks, gan.z_dim))


# In[52]:


gen_scores = gan.generator.predict([chords_noise, style_noise, melody_noise, groove_noise])


# In[77]:


np.argmax(gen_scores[0,0,0:4,:,3], axis = 1)


# In[78]:


gen_scores[0,0,0:4,60,3] = 0.02347812


# In[79]:


filename = 'example'
gan.notes_to_midi(RUN_FOLDER, gen_scores, filename)
gen_score = converter.parse(os.path.join(RUN_FOLDER, 'samples/{}.midi'.format(filename)))
gen_score.show()


# In[11]:


gan.draw_score(gen_scores, 0)


# # find the closest match

# In[80]:


def find_closest(data_binary, score):
    current_dist = 99999999
    current_i = -1
    for i, d in enumerate(data_binary):
        dist = np.sqrt(np.sum(pow((d - score),2)))
        if dist < current_dist:
            current_i = i
            current_dist = dist
        
    return current_i
    


# In[81]:


closest_idx = find_closest(data_binary, gen_scores[0])
closest_data = data_binary[[closest_idx]]
print(closest_idx)


# In[82]:


filename = 'closest'
gan.notes_to_midi(RUN_FOLDER, closest_data,filename)
closest_score = converter.parse(os.path.join(RUN_FOLDER, 'samples/{}.midi'.format(filename)))
print('original')
gen_score.show()
print('closest')
closest_score.show()


# # changing the chords noise

# In[104]:


chords_noise_2 = 5 * np.ones((1, gan.z_dim))


# In[105]:


chords_scores = gan.generator.predict([chords_noise_2, style_noise, melody_noise, groove_noise])


# In[106]:


filename = 'changing_chords'
gan.notes_to_midi(RUN_FOLDER, chords_scores, filename)
chords_score = converter.parse(os.path.join(RUN_FOLDER, 'samples/{}.midi'.format(filename)))
print('original')
gen_score.show()
print('chords noise changed')
chords_score.show()


# # changing the style noise

# In[111]:


style_noise_2 = 5 * np.ones((1, gan.z_dim))


# In[112]:


style_scores = gan.generator.predict([chords_noise, style_noise_2, melody_noise, groove_noise])


# In[113]:


filename = 'changing_style'
gan.notes_to_midi(RUN_FOLDER, style_scores, filename)
style_score = converter.parse(os.path.join(RUN_FOLDER, 'samples/{}.midi'.format(filename)))
print('original')
gen_score.show()
print('style noise changed')
style_score.show()


# # changing the melody noise

# In[114]:


melody_noise_2 = np.copy(melody_noise)
melody_noise_2[0,0,:] = 5 * np.ones(gan.z_dim) 


# In[115]:


melody_scores = gan.generator.predict([chords_noise, style_noise, melody_noise_2, groove_noise])


# In[116]:


filename = 'changing_melody'
gan.notes_to_midi(RUN_FOLDER, melody_scores, filename)
melody_score = converter.parse(os.path.join(RUN_FOLDER, 'samples/{}.midi'.format(filename)))
print('original')
gen_score.show()
print('melody noise changed')
melody_score.show()


# # changing the groove noise

# In[120]:


groove_noise_2 = np.copy(groove_noise)
groove_noise_2[0,3,:] = 5 * np.ones(gan.z_dim)


# In[121]:


groove_scores = gan.generator.predict([chords_noise, style_noise, melody_noise, groove_noise_2])


# In[122]:


filename = 'changing_groove'
gan.notes_to_midi(RUN_FOLDER, groove_scores, filename)
groove_score = converter.parse(os.path.join(RUN_FOLDER, 'samples/{}.midi'.format(filename)))
print('original')
gen_score.show()
print('groove noise changed')
groove_score.show()


# In[ ]:





# In[ ]:




