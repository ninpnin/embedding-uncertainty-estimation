import tensorflow as tf
import numpy as np

from embedding_uncertainty.slh import SparseLowrankHessian

def main():
	V = 7
	D = 3

	x = np.random.rand(V * 2, D)

	theta = np.random.rand(V * 2, D)
	indices = [((1,2), 1.0, 1.0),((1,3), 1.0, 1.0)]
	H = SparseLowrankHessian(indices, theta)

	y = H.multiply_with(x)
	print(y)

if __name__ == '__main__':
	main()