import numpy as np
from sklearn.utils import murmurhash3_32
from heapq import heappush, heapreplace
import random

def MSE(estimate, actual):
	return np.sqrt((estimate - actual) ** 2) / actual

def hash_func(seed, R):
	s = random.randint(0, 1000)
	return lambda x: (murmurhash3_32(x, seed, True) % R)

def g_func(seed):
	s = random.randint(0, 1000)
	return lambda x: -1 if murmurhash3_32(x, s, positive=True) % 2 == 0 else 1

class CountMinSketch:
	def __init__(self, R):
		self.hash_functions = [hash_func(i, R) for i in range(4)]
		self.hash_arrays = [np.zeros((R,), dtype=int) for i in range(4)]
		self.freq_1000_heap = []

	def get_freq_1000(self):
		return map(lambda p, v: v, self.freq_1000_heap)

	def insert(self, x):
		for i in range(4):
			self.hash_arrays[i][self.hash_functions[i](x)] += 1
		self._heap_insert(x)
	
	def _heap_insert(self, x):
		new_count = self.query(x)
		for i in range(len(self.freq_1000_heap)):
			if self.freq_1000_heap[i][1] == x:
				self.freq_1000_heap.pop(i)
				heappush(self.freq_1000_heap, (new_count, x))
				return

		if len(self.freq_1000_heap) < 1000:
			heappush(self.freq_1000_heap, (new_count, x))
		else:
			if new_count > self.freq_1000_heap[0][0]:
				heapreplace(self.freq_1000_heap, (new_count, x))

	def query(self, x):
		return min([self.hash_arrays[i][self.hash_functions[i](x)] for i in range(4)])

class CountSketch:
	def __init__(self, R):
		self.hash_functions = [hash_func(i, R) for i in range(4)]
		self.g_functions = [g_func(i + 10) for i in range(4)]
		self.hash_arrays = [np.zeros((R,), dtype=int) for i in range(4)]
		self.freq_1000_heap = []

	def get_freq_1000(self):
		return map(lambda p, v: v, self.freq_1000_heap)


	def insert(self, x):
		for i in range(4):
			self.hash_arrays[i][self.hash_functions[i](x)] += self.g_functions[i](x)
		self._heap_insert(x)

	def _heap_insert(self, x):
		new_count = self.query(x)
		for i in range(len(self.freq_1000_heap)):
			if self.freq_1000_heap[i][1] == x:
				self.freq_1000_heap.pop(i)
				heappush(self.freq_1000_heap, (new_count, x))
				return

		if len(self.freq_1000_heap) < 1000:
			heappush(self.freq_1000_heap, (new_count, x))
		else:
			if new_count > self.freq_1000_heap[0][0]:
				heapreplace(self.freq_1000_heap, (new_count, x))

	def query(self, x):
		return int(np.median([ self.hash_arrays[i][self.hash_functions[i](x)] * self.g_functions[i](x) for i in range(4) ]))
