from keras.preprocessing import sequence
from heapq import nlargest, nsmallest
import numpy as np
import sys


### Helper functions for model data preparing

def map_char_to_id(c):
	return ord(c)

def map_to_X(raw_X, maxlen=30):
	X = [ [ map_char_to_id(c) for c in x  ] for x in raw_X]
	return sequence.pad_sequences(X, maxlen=maxlen, padding='post')

def map_to_Y(raw_X, percent = 0.05):
	kw_dict = load_keyword_dictionary(raw_X)
	heavy_hitters = nlargest(int(len(kw_dict.keys()) * percent), kw_dict, key = kw_dict.get)
	Y = []
	for x in raw_X:
		if x in heavy_hitters:
			Y.append(1)
		else:
			Y.append(0)
	return Y

def load_keyword_dictionary(keywords):
	keyword_dict = {}
	for kw in keywords:
		if kw in keyword_dict:
			keyword_dict[kw] += 1
		else:
			keyword_dict[kw] = 1
	return keyword_dict


### Helper functions for assesing results

def error(estimated, actual):
	return abs(estimated - actual) * actual

def MSE(estimate, actual):
	return np.sqrt((estimate - actual) ** 2) / actual

### Helper functions for computing memory

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def byte_to_mb(byte):
	return byte * 1e-6
