import numpy as np
from sklearn.utils import murmurhash3_32
from heapq import heappush, heapreplace
import pickle
from util import map_to_X, MSE, byte_to_mb, get_size, load_keyword_dictionary
from Algorithms import SpaceSaving
import matplotlib.pyplot as plt
import sys


with open("test.txt", "rb") as f1:
	queries = pickle.load(f1)
with open("model.obj", "rb") as f3:
	model = pickle.load(f3)

test_counts = load_keyword_dictionary(queries)


def learning_algo(model, queries, sketches):
	heavy_hitter_buckets = {}
	X = map_to_X(queries)
	predictions = model.predict(X)
	for i in range(len(queries)):
		if predictions[i] > 0.7:
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

def compute_learning_freq(func, heavy_hitter_buckets, m, freq, total):
	results = set(func.QueryFrequent(m, freq, total)[1])
	limit = freq * total
	for q, count in heavy_hitter_buckets.items():
		if count >= limit:
			results.add(q)
	return results

def compute_nonlearning_freq(func, m, freq, total):
	return set(func.QueryFrequent(m, freq, total)[1])


learning_values = [int(2 ** i) for i in np.arange(5, 14, 0.12)]
non_learning_values = [int(2 ** i) for i in np.arange(12, 15, 0.04)]
learning = [SpaceSaving(m) for m in learning_values]
non_learning = [SpaceSaving(m) for m in non_learning_values]
heavy_hitter_buckets = learning_algo(model, queries, learning)
non_learning_algo(queries, non_learning)


def main():
	args = sys.argv
	if len(args) == 1:
		print("No Frequency Threshold Specified\n")
		return
	freq = float(args[1])

	length = 75
	learning_x = []
	learning_y = []
	non_learning_x = []
	non_learning_y = []
	total = len(queries)
	limit = freq * total
	real_results = set()
	for q in queries:
		if test_counts[q] >= limit:
			real_results.add(q)
	real_num = len(real_results)

	for i in range(length):
		learning_results = compute_learning_freq(learning[i], heavy_hitter_buckets, learning_values[i], freq, total)
		non_learning_results = compute_nonlearning_freq(non_learning[i], non_learning_values[i], freq, total)
		
		learning_intersection = len(real_results.intersection(learning_results))
		non_learning_intersection = len(real_results.intersection(non_learning_results))

		learning_mem = byte_to_mb(learning[i].ComputeMem() + get_size(learning[i].error) + sys.getsizeof(model) + get_size(heavy_hitter_buckets))
		non_learning_mem = byte_to_mb(non_learning[i].ComputeMem() + get_size(non_learning[i].error))
	
		learning_x.append(learning_mem)
		learning_y.append(learning_intersection / real_num)

		non_learning_x.append(non_learning_mem)
		non_learning_y.append(non_learning_intersection / real_num)

	print(learning_x)
	print(learning_y)
	print(non_learning_x)
	print(non_learning_y)

	plt.plot(learning_x, learning_y, label = "learning")
	plt.plot(non_learning_x, non_learning_y, label = "non learning")
	plt.xlabel("Space (MB)")
	plt.ylabel("Accuracy with Threshold = " + str(freq) + " %")
	plt.title("Space-Saving Algorithm")
	plt.legend()
	plt.show()



if __name__== "__main__":
  main()
