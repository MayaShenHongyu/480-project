#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.utils import murmurhash3_32
import random
import csv
import pandas as pd
import numpy as np
import heapq
import pickle
import json
from datetime import datetime


# In[2]:


data = pd.read_csv("user-ct-test-collection-02.txt", sep="\t")


# In[3]:


size = len(data.index)
train_set = []
valid_set = []
train_limit = datetime(2006, 3, 5, 23, 59, 59)
valid_limit = datetime(2006, 3, 6, 23, 59, 59)
for i in range(size):
    item = data.iloc[i]
    date = datetime.strptime(item.QueryTime, "%Y-%m-%d %H:%M:%S")
    if (date <= train_limit):
        train_set.append(item)
    elif (date <= valid_limit):
        valid_set.append(item)


# In[5]:


length = len(subset)
count = {}
for item in train_set
    query = item.Query
    if url in count:
        count[url] += 1
    else:
        count[url] = 1


# In[4]:


training = []
validation = []
for item in train_set:
    query = item.Query
    if pd.isna(query):
        continue
    training.append(query)
for item in valid_set:
    query = item.Query
    if pd.isna(query):
        continue
    validation.append(query)


# In[5]:


with open("training.txt", "wb") as f1:
    pickle.dump(training, f1)
    
with open("validation.txt", "wb") as f2:
    pickle.dump(validation, f2)
