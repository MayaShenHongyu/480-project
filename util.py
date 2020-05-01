from keras.preprocessing import sequence
from heapq import nlargest, nsmallest
import numpy as np


### Helper functions for model data preparing

def map_char_to_id(c):
	return ord(c)

def map_to_X(raw_X, maxlen=30):
	X = [ [ map_char_to_id(c) for c in x  ] for x in raw_X]
	return sequence.pad_sequences(X, maxlen=maxlen, padding='post')

def map_to_Y(raw_X, dataset, percent = 0.01):
	kw_dict = load_keyword_dictionary(dataset)
	heavy_hitters = nlargest(int(len(dataset) * percent), kw_dict, key = kw_dict.get)
	Y = []
	for x in raw_X:
		if x in heavy_hitters:
			Y.append(1)
		else:
			Y.append(0)
	return Y

# def map_to_X_for_algo(raw_X, maxlen=30):
# 	X = [ [ map_char_to_id(c) for c in x  ] for x in raw_X]
# 	return sequence.pad_sequences(X, maxlen=maxlen, padding='post')
# def map_to_Y_for_algo(raw_X, dataset, percent = 0.01):
# 	kw_dict = load_keyword_dictionary(dataset)
# 	heavy_hitters = nlargest(int(len(dataset) * percent), kw_dict, key = kw_dict.get)
# 	Y = []
# 	for x in raw_X:
# 		if x in heavy_hitters:
# 			Y.append(1)
# 		else:
# 			Y.append(0)
# 	return Y

def load_keyword_dictionary(keywords):
	keyword_dict = {}
	for kw in keywords:
		if kw in keyword_dict:
			keyword_dict[kw] += 1
		else:
			keyword_dict[kw] = 1
	return keyword_dict


### Helper functions for assesing results

def MSE(estimate, actual):
	return np.sqrt((estimate - actual) ** 2) / actual
