import numpy as np
import pandas as pd
import pickle
import sys

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from heapq import nlargest, nsmallest

from Algorithms import CountMinSketch, CountSketch, MSE

np.random.seed(7)

def map_to_X_for_algo(raw_X, maxlen=20):
	X = [ [ map_char_to_id(c) for c in x  ] for x in raw_X]
	return sequence.pad_sequences(X, maxlen=maxlen, padding='post')

def map_char_to_id(c):
	return ord(c)

def map_to_X(raw_X, maxlen=20):
	X = [ [ map_char_to_id(c) for c in x  ] for x in set(raw_X)]
	return sequence.pad_sequences(X, maxlen=maxlen, padding='post')

def map_to_Y(raw_X, dataset, percent = 0.01):
	kw_dict = load_keyword_dictionary(dataset)
	heavy_hitters = nlargest(int(len(dataset) * percent), kw_dict, key = kw_dict.get)
	Y = []
	for x in kw_dict:
		if x in heavy_hitters:
			Y.append(1)
		else:
			Y.append(0)
	return Y


def process_query_keywords(queries):
	# data = pd.read_csv("user-ct-test-collection-01.txt", sep="\t")
	# queries = data.Query.dropna().values.tolist()
	query_keywords = []
	for query in queries:
		query_keywords.extend(query.split(' '))
	return query_keywords


def load_keyword_dictionary(keywords):
	keyword_dict = {}
	for kw in keywords:
		if kw in keyword_dict:
			keyword_dict[kw] += 1
		else:
			keyword_dict[kw] = 1
	return keyword_dict



# with open("training.txt", "rb") as f1:
#     train_queries = pickle.load(f1)

# with open("validation.txt", "rb") as f2:
# 	test_queries = pickle.load(f2)

# train_keywords = process_query_keywords(train_queries)
# test_keywords = process_query_keywords(test_queries)


with open("training.txt", "rb") as f1:
    train_keywords = pickle.load(f1, encoding='bytes')

with open("validation.txt", "rb") as f2:
	test_keywords = pickle.load(f2, encoding='bytes')


with open("test.txt", "rb") as f3:
	queries = pickle.load(f3)
with open("freq-infreq-rand-counts.txt", "rb") as f4:
	test_queries_sets = pickle.load(f4)

X_train = map_to_X(train_keywords)
y_train = map_to_Y(train_keywords, train_keywords)

X_test = map_to_X(test_keywords)
y_test = map_to_Y(test_keywords, test_keywords)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(300, embedding_vecor_length, input_length=20))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


## Input: model - trained model, queries - list of all test queries, sketches - list of sketch objects
## Return: heavy_hitter_buckets - dictionary of heavy hitter counts, sketches - sketch objects with test queries inserted
def learning_algo(model, queries, sketches):
	heavy_hitter_buckets = {}
	X = map_to_X_for_algo(queries)
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


## Input: 
##   sketch_name - e.g. Count-Min Sketch
##   test_queries_sets - a dictonary with key being the set name (e.g. Rand100) and value being the counts dictionary (key: query, value: actual count)
##
## Effect: plot a single graph, each test query set has two lines: learning and non-learning
def plot_graph(sketch_name, test_queries_sets, learning_sketches, heavy_hitter_buckets, model, non_learning_sketches):

	for query_set_name in test_queries_sets:
		test_queries_and_counts = test_queries_sets[query_set_name]
		learning_sketches_x = [sys.getsizeof(sketch) + sys.getsizeof(heavy_hitter_buckets) + sys.getsizeof(model) for sketch in learning_sketches]
		learning_sketches_y = [compute_learning_sketch_accuracy(test_queries_and_counts, heavy_hitter_buckets, sketch) for sketch in learning_sketches]
		non_learning_sketches_x = [sys.getsizeof(sketch) for sketch in non_learning_sketches]
		non_learning_sketches_y = [compute_non_learning_sketch_accuracy(test_queries_and_counts, sketch) for sketch in non_learning_sketches]

		plt.plot(learning_sketches_x, learning_sketches_y, label=str("Learning - " + query_set_name))
		plt.plot(non_learning_sketches_x, non_learning_sketches_y, label=str("Non-learning - " + query_set_name))

	plt.title(sketch_name)
	
	plt.xlabel("Memory usage")
	plt.ylabel("MSE")
	plt.legend(loc="upper right")
	plt.show()



R_values = [2 ** i for i in range(5, 6)] # 10 x values
learning_sketches = [ CountSketch(R) for R in R_values ]
non_learning_sketches = [ CountSketch(R) for R in R_values ]
heavy_hitter_buckets = learning_algo(model, queries, learning_sketches)
non_learning_algo(queries, non_learning_sketches)
plot_graph("Count Sketch", test_queries_sets, learning_sketches, heavy_hitter_buckets, model, non_learning_sketches)

