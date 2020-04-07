from sklearn.utils import murmurhash3_32
import random
import csv
import pandas as pd
import numpy as np
import heapq
import pickle
from datetime import datetime


data = pd.read_csv("user-ct-test-collection-02.txt", sep="\t")


size = len(data.index)
subset = []
subsize = 0
limit = datetime(2006, 3, 5, 0, 0, 0)
for i in range(size):
    item = data.iloc[i]
    if item.ClickURL == None:
        continue
    date = datetime.strptime(item.QueryTime, "%Y-%m-%d %H:%M:%S")
    if (date < limit):
        subset.append(item)


length = len(subset)
count = {}
for item in subset[:int(length * 5 / 6)]:
    query = item.Query
    if url in count:
        count[url] += 1
    else:
        count[url] = 1


training = []
validation = []
for item in subset[:int(length * 5 / 6)]:
    query = item.Query
    if pd.isna(query):
        continue
    training.append(query)
for item in subset[int(length * 5 / 6):]:
    query = item.Query
    if pd.isna(query):
        continue
    validation.append(query)



with open("training.txt", "wb") as f1:
    pickle.dump(training, f1)
    
with open("validation.txt", "wb") as f2:
    pickle.dump(validation, f2)

