import numpy as np
import pickle
import sys
from heapq import nlargest, nsmallest
import matplotlib.pyplot as plt

from Algorithms import CountMinSketch, CountSketch

from util import map_to_X, MSE



# 5 day test data
with open("test.txt", "rb") as f1:
	queries = pickle.load(f1)

with open("freq-infreq-rand-counts.txt", "rb") as f2:
	test_queries_sets = pickle.load(f2)

with open("model-repetitive.obj", "rb") as f3:
	model = pickle.load(f3)


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


## Return the mean MSE of a sketch's estimation of test queries. This is a single "y" value in the graph.
## test_queries_and_counts is a dictionary of the test query set (e.g. Rand100) with value being their actual counts.
def compute_learning_sketch_accuracy(test_queries_and_counts, heavy_hitter_buckets, sketch):
	MSE_values = []
	for q in test_queries_and_counts:
		if q in heavy_hitter_buckets:
			estimate = heavy_hitter_buckets[q]
		else:
			estimate = sketch.query(q)

		MSE_values.append(MSE(estimate, test_queries_and_counts[q]))
	return np.mean(MSE_values)

def compute_non_learning_sketch_accuracy(test_queries_and_counts, sketch):
	MSE_values = []
	for q in test_queries_and_counts:
		MSE_values.append(MSE(sketch.query(q), test_queries_and_counts[q]))
	return np.mean(MSE_values)

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

## Input: 
##   sketch_name - e.g. Count-Min Sketch
##   test_queries_sets - a dictonary with key being the set name (e.g. Rand100) and value being the counts dictionary (key: query, value: actual count)
##
## Effect: plot a single graph, each test query set has two lines: learning and non-learning
def plot_graph(sketch_name, test_queries_sets, learning_sketches, heavy_hitter_buckets, model, non_learning_sketches, test_query_names={"Freq100":"Freq100", "Infreq100":"Infreq100", "Rand100":"Rand100"}):

	# fig, axes = plt.subplots(1, 3)

	i = 0

	for query_set_name in test_queries_sets:
		test_queries_and_counts = test_queries_sets[query_set_name]
		learning_sketches_x = [byte_to_mb(get_size(sketch.hash_arrays) + get_size(heavy_hitter_buckets) + sys.getsizeof(model)) for sketch in learning_sketches]

		learning_sketches_y = [compute_learning_sketch_accuracy(test_queries_and_counts, heavy_hitter_buckets, sketch) for sketch in learning_sketches]
		non_learning_sketches_x = [byte_to_mb(get_size(sketch.hash_arrays)) for sketch in non_learning_sketches]
		non_learning_sketches_y = [compute_non_learning_sketch_accuracy(test_queries_and_counts, sketch) for sketch in non_learning_sketches]

		print(query_set_name)
		# axes[i].plot(learning_sketches_x, learning_sketches_y, label="Learning")
		# axes[i].plot(non_learning_sketches_x, non_learning_sketches_y, label="Non-learning")
		# axes[i].set_xlabel("Space (MB)")
		# axes[i].set_ylabel("MSE")
		# axes[i].legend(loc="upper right")
		# axes[i].set_title(test_query_names[query_set_name])
		# print(query_set_name)
		# i+=1
		print("learning")
		print(learning_sketches_x)
		print(learning_sketches_y)
		print("non-learning")
		print(non_learning_sketches_x)
		print(non_learning_sketches_y)

	
	

	# fig.suptitle(sketch_name)
	# fig.tight_layout(pad=0.5)
	# plt.show()



R_values = [int(2 ** i) for i in np.arange(15, 25, 0.2)] # 10 x values
# R_values = [ 2 ** 10, 2 ** 15, 2 ** 20]
learning_sketches = [ CountSketch(R) for R in R_values ]
non_learning_sketches = [ CountSketch(R) for R in R_values ]
heavy_hitter_buckets = learning_algo(model, queries, learning_sketches)
print("learning sketch")
non_learning_algo(queries, non_learning_sketches)
print("non-learning sketch")
plot_graph("Count Sketch", test_queries_sets, learning_sketches, heavy_hitter_buckets, model, non_learning_sketches)



