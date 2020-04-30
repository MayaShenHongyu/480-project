import numpy as np
from sklearn.utils import murmurhash3_32
import random
import pandas as pd
from heapq import nlargest, nsmallest, heappush, heapreplace
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

def load_keyword_dictionary(keywords):
	keyword_dict = {}
	for kw in keywords:
		if kw in keyword_dict:
			keyword_dict[kw] += 1
		else:
			keyword_dict[kw] = 1
	return keyword_dict


with open("test.txt", "rb") as f2:
	keywords = pickle.load(f2)


keyword_dict = load_keyword_dictionary(keywords)

Freq_100 = nlargest(100, keyword_dict, key = keyword_dict.get) 
Infreq_100 = nsmallest(100, keyword_dict, key = keyword_dict.get)
Rand_100 = nlargest(100, random.sample(keyword_dict.keys(), 100), key=keyword_dict.get)


queries = {"Freq100": Freq_100, "Infreq100": Infreq_100, "Rand100": Rand_100}
Freq_100_dict = {}
Infreq_100_dict = {}
Rand_100_dict = {}

dictionaries = {"Freq100": Freq_100_dict, "Infreq100": Infreq_100_dict, "Rand100": Rand_100_dict}

for setname in queries:
	for q in queries[setname]:
		dictionaries[setname][q] = keyword_dict[q]

for setname in dictionaries:
	print(dictionaries[setname])
with open("freq-infreq-rand-counts.txt", "wb") as f1:
    pickle.dump(dictionaries, f1)
