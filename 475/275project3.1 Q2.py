#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import numpy as np
import pandas as pd
import unicodedata
import string
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt


# In[3]:


def findFiles(path): 
    return glob.glob(path)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# In[4]:


names = {}
languages = []


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# (TO DO:) CHANGE FILE PATH AS NECESSARY
for filename in findFiles('/Users/xiaoshiya/Desktop/data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    languages.append(category)
    lines = readLines(filename)
    names[category] = lines


# In[5]:


n_categories = len(languages)

def letterToIndex(letter):
    return all_letters.find(letter)


def nameToTensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)
    for li, letter in enumerate(name):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


# In[6]:


class RNN(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, N_LAYERS,OUTPUT_SIZE):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE, # number of hidden units
            num_layers = N_LAYERS, # number of layers
            batch_first = True)
        self.out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
            
    def forward(self, x):
        r_out, h = self.rnn(x, None) # None represents zero initial hidden state           
        out = self.out(r_out[:, -1, :])
        return out


# In[48]:


n_hidden = 128

allnames = [] # Create list of all names and corresponding output language
for language in list(names.keys()):
    for name in names[language]:
        allnames.append([name, language])
        
## (TO DO:) Determine Padding length (this is the length of the longest string) 

maxlen = 0
for i in range(len(allnames)):
    length = len(allnames[i][0])
    if length > maxlen:
        maxlen = length
print (maxlen)
                
n_letters = len(all_letters)
n_categories = len(languages)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i.item()
    return languages[category_i], category_i


# In[49]:


def categoryToTensor(category):
    return torch.tensor([languages.index(category)], dtype=torch.long)

allnames_tensor = pd.DataFrame(np.array(allnames), columns = ["name", "language"])
allnames_tensor["name"] = allnames_tensor["name"].apply(nameToTensor)
allnames_tensor["language"] = allnames_tensor["language"].apply(categoryToTensor)


# In[50]:


import torch

learning_rate = 0.005
rnn = RNN(n_letters, 128, 1, n_categories)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)   # optimize all rnn parameters
loss_func = nn.CrossEntropyLoss()  

for epoch in range(5):  
    batch_size = len(allnames)
    padded_length = 0
    random.shuffle(allnames)
    
    # if "b_in" and "b_out" are the variable names for input and output tensors, you need to create those
    b_in = torch.zeros(batch_size, padded_length, n_letters)
    b_out = torch.zeros(batch_size, n_categories)
    
    # (TO DO:) Populate "b_in" tensor
    b_in = torch.nn.utils.rnn.pad_sequence(allnames_tensor["name"].tolist(), batch_first=True)
    b_in = b_in.reshape([batch_size, maxlen, n_letters])
    
    # (TO DO:) Populate "b_out" tensor
    b_out = torch.nn.utils.rnn.pad_sequence(allnames_tensor["language"].tolist(), batch_first=True)
    b_out = b_out.reshape(batch_size)
       
    output = rnn(b_in)                               # rnn output
    
    loss = loss_func(output, b_out)   # (TO DO:) Fill "...." to calculate the cross entropy loss
    optimizer.zero_grad()                           # clear gradients for this training step
    loss.backward()                                 # backpropagation, compute gradients
    optimizer.step()                                # apply gradients
        
    # Print accuracy
    test_output = rnn(b_in)                   # 
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    test_y = b_out.data.numpy().squeeze()
    accuracy = sum(pred_y == test_y)/batch_size
    print("Epoch: ", epoch, "| train loss: %.4f" % loss.item(), '| accuracy: %.2f' % accuracy)


# In[ ]:




