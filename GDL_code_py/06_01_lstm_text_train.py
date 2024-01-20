#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import re
from IPython.display import clear_output

from keras.layers import Dense, LSTM, Input, Embedding, Dropout
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import LambdaCallback


# In[2]:


load_saved_model = False
train_model = False


# In[3]:


token_type = 'word'


# In[4]:


#load in the text and perform some cleanup

seq_length = 20

filename = "./data/aesop/data.txt"

with open(filename, encoding='utf-8-sig') as f:
    text = f.read()
    
    
#removing text before and after the main stories
start = text.find("THE FOX AND THE GRAPES\n\n\n")
end = text.find("ILLUSTRATIONS\n\n\n[")
text = text[start:end]


# In[5]:


start_story = '| ' * seq_length
    
text = start_story + text
text = text.lower()
text = text.replace('\n\n\n\n\n', start_story)
text = text.replace('\n', ' ')
text = re.sub('  +', '. ', text).strip()
text = text.replace('..', '.')

text = re.sub('([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])', r' \1 ', text)
text = re.sub('\s{2,}', ' ', text)


# In[6]:


len(text)


# In[7]:


text


# In[8]:



if token_type == 'word':
    tokenizer = Tokenizer(char_level = False, filters = '')
else:
    tokenizer = Tokenizer(char_level = True, filters = '', lower = False)
    
    
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

token_list = tokenizer.texts_to_sequences([text])[0]


# In[9]:


total_words


# In[10]:


print(tokenizer.word_index)
print(token_list)


# In[11]:


def generate_sequences(token_list, step):
    
    X = []
    y = []

    for i in range(0, len(token_list) - seq_length, step):
        X.append(token_list[i: i + seq_length])
        y.append(token_list[i + seq_length])
    

    y = np_utils.to_categorical(y, num_classes = total_words)
    
    num_seq = len(X)
    print('Number of sequences:', num_seq, "\n")
    
    return X, y, num_seq

step = 1
seq_length = 20

X, y, num_seq = generate_sequences(token_list, step)

X = np.array(X)
y = np.array(y)


# In[12]:


X.shape


# In[46]:


y.shape


# ## Define the LSTM model

# In[47]:


if load_saved_model:
    # model = load_model('./saved_models/lstm_aesop_1.h5')
    model = load_model('./saved_models/aesop_dropout_100.h5')

else:

    n_units = 256
    embedding_size = 100

    text_in = Input(shape = (None,))
    embedding = Embedding(total_words, embedding_size)
    x = embedding(text_in)
    x = LSTM(n_units)(x)
    # x = Dropout(0.2)(x)
    text_out = Dense(total_words, activation = 'softmax')(x)

    model = Model(text_in, text_out)

    opti = RMSprop(lr = 0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opti)


# In[48]:


model.summary()


# In[49]:


def sample_with_temp(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)



def generate_text(seed_text, next_words, model, max_sequence_len, temp):
    output_text = seed_text
    
    seed_text = start_story + seed_text
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-max_sequence_len:]
        token_list = np.reshape(token_list, (1, max_sequence_len))
        
        probs = model.predict(token_list, verbose=0)[0]
        y_class = sample_with_temp(probs, temperature = temp)
        
        if y_class == 0:
            output_word = ''
        else:
            output_word = tokenizer.index_word[y_class]
            
        if output_word == "|":
            break
            
        if token_type == 'word':
            output_text += output_word + ' '
            seed_text += output_word + ' '
        else:
            output_text += output_word + ' '
            seed_text += output_word + ' '
            
            
    return output_text


# In[50]:


def on_epoch_end(epoch, logs):
    seed_text = ""
    gen_words = 500

    print('Temp 0.2')
    print (generate_text(seed_text, gen_words, model, seq_length, temp = 0.2))
    print('Temp 0.33')
    print (generate_text(seed_text, gen_words, model, seq_length, temp = 0.33))
    print('Temp 0.5')
    print (generate_text(seed_text, gen_words, model, seq_length, temp = 0.5))
    print('Temp 1.0')
    print (generate_text(seed_text, gen_words, model, seq_length, temp = 1))

    
    
if train_model:
    epochs = 1000
    batch_size = 32
    num_batches = int(len(X) / batch_size)
    callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks = [callback], shuffle = True)



# In[51]:


model.summary()


# In[52]:


seed_text = "the frog and the snake . "
gen_words = 500
temp = 0.1

print (generate_text(seed_text, gen_words, model, seq_length, temp))


# In[53]:


def generate_human_led_text(model, max_sequence_len):
    
    output_text = ''
    seed_text = start_story
    
    while 1:
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-max_sequence_len:]
        token_list = np.reshape(token_list, (1, max_sequence_len))
        
        probs = model.predict(token_list, verbose=0)[0]

        top_10_idx = np.flip(np.argsort(probs)[-10:])
        top_10_probs = [probs[x] for x in top_10_idx]
        top_10_words = tokenizer.sequences_to_texts([[x] for x in top_10_idx])
        
        for prob, word in zip(top_10_probs, top_10_words):
            print('{:<6.1%} : {}'.format(prob, word))

        chosen_word = input()
                
        if chosen_word == '|':
            break
            
        
        seed_text += chosen_word + ' '
        output_text += chosen_word + ' '
        
        clear_output()

        print (output_text)
            
    
    


# In[54]:


generate_human_led_text(model, 20)


# In[20]:


# model.save('./saved_models/aesop_no_dropout_100.h5')

