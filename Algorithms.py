import numpy as np
from sklearn.utils import murmurhash3_32
from heapq import heappush, heapreplace
import random
import sys


def hash_func(seed, R):
	s = random.randint(0, 1000)
	return lambda x: (murmurhash3_32(x, seed, True) % R)

def g_func(seed):
	s = random.randint(0, 1000)
	return lambda x: -1 if murmurhash3_32(x, s, positive=True) % 2 == 0 else 1

class SingleHash:
	def __init__(self, R):
		self.hash_func = hash_func(0, R);
		self.hash_arrays = np.zeros((R,), dtype=int)

	def insert(self, x):
		self.hash_arrays[self.hash_func(x)] += 1

	def query(self, x):
		return self.hash_arrays[self.hash_func(x)]


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
		# self._heap_insert(x)
	
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
		# self._heap_insert(x)

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


### Space Saving
class Counter:
	def __init__(self, count, q):
		self.query = q
		self.count = count
		self.next = None
		self.prev = None
		self.parent = None

	def update(self):
		self.count += 1
		bucket = self.parent.next
		if bucket and bucket.count == self.count:
			self.parent.remove(self)
			self.parent = bucket
			bucket.insert(self)
		else:
			new_bucket = StreamSummary(self.count)

			# Insert new bucket after parent bucket
			if bucket:
				bucket.prev = new_bucket
			new_bucket.next = bucket
			self.parent.next = new_bucket
			new_bucket.prev = self.parent

			self.parent.remove(self)
			self.prev = None
			self.next = None
			new_bucket.counter = self

			self.parent = new_bucket

class StreamSummary:

	def __init__(self, count):
		self.count = count
		self.next = None
		self.prev = None
		self.counter = None

	def insert(self, counter):
		if self.counter == None:
			self.counter = counter
		else:
			counter.prev = None
			counter.next = self.counter
			self.counter.prev = counter
			self.counter = counter

	def remove(self, counter):
		if self.count == 0:
			return
		if self.counter.query == counter.query:
			self.counter = self.counter.next
			if self.counter:
				self.counter.prev = None
		else:
			if counter.prev:
				counter.prev.next = counter.next
			if counter.next:
				counter.next.prev = counter.prev

		# Remove empty bucket
		if self.counter == None:
			if self.prev:
				self.prev.next = self.next
			if self.next:
				self.next.prev = self.prev

	def replace(self, q):
		old_query = self.counter.query
		self.counter.query = q
		return old_query, self.counter

class SpaceSaving:
	def __init__(self, R):
		self.zero = StreamSummary(0)
		self.R = R
		self.num = 0
		self.elems = {}
		self.error = {}
		self.total = 0

	def insert(self, q):
		self.total += 1
		if q in self.elems:
			self.elems[q].update()

		elif self.R > self.num:
			counter = Counter(0, q)
			counter.parent = self.zero
			counter.update()
			self.elems[q] = counter
			self.error[q] = 0
			self.num += 1

		else:
			least_bucket = self.zero.next
			self.error[q] = least_bucket.count
			oldq, counter = least_bucket.replace(q)
			self.elems[q] = counter
			self.elems.pop(oldq, None)
			self.error.pop(oldq, None)
			counter.update()

	def QueryFrequent(self, m, freq, total):
		current = self.zero.next
		while current.next:
			current = current.next
		i = 0
		result = []
		guaranteed = True
		limit = freq * total
		while current and current.count >= limit and i <= m:
			counter = current.counter
			while counter:
				i += 1
				result.append(counter.query)
				if current.count - self.error[counter.query] < limit:
					guaranteed = False
				counter = counter.next
			current = current.prev
		return guaranteed, result
	
	def ComputeMem(self):
		num_bytes = 0
		current = self.zero.next
		while current:
			num_bytes += sys.getsizeof(current)
			counter = current.counter
			while counter:
				num_bytes += sys.getsizeof(counter)
				counter = counter.next
			current = current.next
		return num_bytes