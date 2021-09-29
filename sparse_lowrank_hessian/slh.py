import tensorflow as tf
import numpy as np

class SparseLowrankHessian:
	def __init__(self, cooccurences, embedding):
		self.cooccurences = cooccurences
		assert len(embedding.shape) == 2
		self.v = embedding.shape[0] // 2
		self.d = embedding.shape[1]
		self.rhos = embedding[:self.v]
		self.alphas = embedding[self.v:]

	def multiply_with(self, x):
		y = np.zeros(x.shape)
		current_i = 0
		i_vec = []
		j_vec = []

		def flush_row(i_vec, j_vec):
			if len(i_vec) > 0:
				i_vec = tf.constant(i_vec, dtype=tf.int32)
				j_vec = tf.constant(j_vec, dtype=tf.int64)
				# Calculate row
				phis = tf.gather(self.rhos, i_vec)
				alphas = tf.gather(self.alphas, j_vec)

				print(phis)
				x_current = tf.constant(x[current_i], shape=[self.d, 1])
				print(x_current)

				row = tf.linalg.matmul(phis, x_current)
				row = tf.linalg.matmul(alphas, row, transpose_a=True)

				y[current_i] = row[:, 0]

		for coordinates, v1, v2 in self.cooccurences:
			i, j = coordinates
			if i != current_i:
				print(current_i, i_vec)
				flush_row(i_vec, j_vec)
				# Reset
				i_vec = []	
				j_vec = []
				current_i = i

			i_vec.append(i)
			j_vec.append(j)

		flush_row(i_vec, j_vec)

		return y