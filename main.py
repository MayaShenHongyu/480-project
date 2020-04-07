import numpy as np
import pandas as pd
from heapq import nlargest, nsmallest, heappush, heapreplace


from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

np.random.seed(7)

def map_char_to_id(c):
	return ord(c)

def map_to_X(raw_X, maxlen=30):
	X = [ [ map_char_to_id(c) for c in x  ] for x in raw_X]
	return sequence.pad_sequences(X, maxlen=maxlen, padding='post')


def split_data_to_train_and_test(data, ratio=0.2):
	l = int(len(data) * ratio)
	return data[:l], data[l:]

def map_to_Y(raw_X, dataset, percent = 0.05):
	kw_dict = load_keyword_dictionary(dataset)
	heavy_hitters = nlargest(int(len(dataset) * percent), kw_dict, key = kw_dict.get)
	Y = []
	for x in raw_X:
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




# keywords = process_query_keywords()[:100000]
# train_keywords, test_keywords = split_data_to_train_and_test(keywords)

with open("training.txt", "rb") as f1:
    train_queries = pickle.load(f1)

with open("validation.txt", "rb") as f2:
	test_queries = pickle.load(f2)

train_keywords = process_query_keywords(train_queries)
test_keywords = process_query_keywords(test_queries)

X_train = map_to_X(train_keywords)
y_train = map_to_Y(train_keywords, train_keywords)

X_test = map_to_X(test_keywords)
y_test = map_to_Y(test_keywords, test_keywords)




# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(300, embedding_vecor_length, input_length=30))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))




