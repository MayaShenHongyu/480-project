import numpy as np
import pickle
from keras.models import load_model
import sys
from heapq import nlargest, nsmallest
import matplotlib.pyplot as plt

from Algorithms import CountMinSketch, CountSketch, SingleHash

from util import map_to_X, MSE, load_keyword_dictionary, error, get_size, byte_to_mb



## Load data and model

with open("test.txt", "rb") as f1:
	queries = pickle.load(f1)

model = load_model("model.h5")

## Input: model - trained model, queries - list of all test queries, sketches - list of sketch objects
## Return: heavy_hitter_buckets - dictionary of heavy hitter counts, sketches - sketch objects with test queries inserted
def learning_algo(model, queries, sketches):
	heavy_hitter_buckets = {}
	X = map_to_X(queries)
	predictions = model.predict(X)
	for i in range(len(queries)):
		if predictions[i] > 0.5:
			if queries[i] in heavy_hitter_buckets:
				heavy_hitter_buckets[queries[i]] += 1
			else:
				heavy_hitter_buckets[queries[i]] = 1
		else:
			for sketch in sketches:
				sketch.insert(queries[i])
	return heavy_hitter_buckets

def non_learning_algo(queries, sketches):
	for sketch in sketches:
		for query in queries:
			sketch.insert(query)


## Return the mean error of a sketch's estimation of test queries. This is a single "y" value in the graph.
## test_queries_and_counts is a dictionary of the test query set (e.g. Rand100) with value being their actual counts.
def compute_learning_sketch_accuracy(test_queries_and_counts, heavy_hitter_buckets, sketch):
	error_values = []
	N = sum(test_queries_and_counts.values())
	for q in test_queries_and_counts:
		if q in heavy_hitter_buckets:
			estimate = heavy_hitter_buckets[q]
		else:
			estimate = sketch.query(q)
		error_values.append(error(estimate, test_queries_and_counts[q]))
	return float(sum(error_values)) / N

def compute_non_learning_sketch_accuracy(test_queries_and_counts, sketch):
	error_values = []
	N = sum(test_queries_and_counts.values())
	for q in test_queries_and_counts:
		error_values.append(error(sketch.query(q), test_queries_and_counts[q]))
	return float(sum(error_values)) / N



# R values for learning and non learning sketches
R_values_non_learning = [int(2 ** i) for i in np.arange(14, 19, 0.1)]
R_values_learning = [int(2 ** i) for i in np.arange(11, 19, 0.1)]

# Initialize sketches: SingleHash, CountMinSketch, or CountSketch
learning_sketches = [ SingleHash(R) for R in R_values_learning ]
non_learning_sketches = [ SingleHash(R) for R in R_values_non_learning ]

# Insert queries
heavy_hitter_buckets = learning_algo(model, queries, learning_sketches)
non_learning_algo(queries, non_learning_sketches)

# Reference solution for actual count
kw_dict = load_keyword_dictionary(queries)

# Compute errors
learning_sketches_x = [byte_to_mb(get_size(sketch.hash_arrays) + get_size(heavy_hitter_buckets) + sys.getsizeof(model)) for sketch in learning_sketches]
learning_sketches_y = [compute_learning_sketch_accuracy(kw_dict, heavy_hitter_buckets, sketch) for sketch in learning_sketches]
non_learning_sketches_x = [byte_to_mb(get_size(sketch.hash_arrays)) for sketch in non_learning_sketches]
non_learning_sketches_y = [compute_non_learning_sketch_accuracy(kw_dict, sketch) for sketch in non_learning_sketches]


plt.plot(learning_sketches_x, learning_sketches_y, label="Learning")
plt.plot(non_learning_sketches_x, non_learning_sketches_y, label="Non-learning")


plt.ylim(0, 12) # this may vary
plt.xlabel("Space (MB)")
plt.ylabel("Average error per query")
plt.title("Single Hash Function")
plt.legend(loc="upper right")
plt.show()
