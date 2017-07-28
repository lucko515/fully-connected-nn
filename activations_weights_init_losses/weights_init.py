import numpy as np

####################
# WEIGHTS INIT     #
####################

def zero_init(input_size, output_size):
	'''
	Do not use this initialization. It is here just for an example.

	Input: sizes of weights matrix
	Output: initialized weights
	'''
	return np.zeros((input_size, output_size))

def random_normal_init(input_size, output_size):
	'''
	Most common way to initialize. Sometimes it is hard to converge. For the debugging purpose seed random generator.

	Problem with this init: Variance is growing as number of input_size is growing.

	Input: sizes of weights matrix
	Output: initialized weights
	'''
	return np.random.randn(input_size, output_size)

def xavier_init(input_size, output_size):
	'''

	This implementation is better than random init. With this implemenatation we are normalizing its variance. 

	Input: sizes of weights matrix
	Output: initialized weights
	'''
	return np.random.randn(input_size, output_size) / np.sqrt(input_size)

def small_init(input_size, output_size, alpha=0.01):
	'''
	Input: sizes of weights matrix
	Output: initialized weights
	'''
	return np.random.randn(input_size, output_size) * alpha

def he_init(input_size, output_size):
	'''

	This is good implementation for ReLu activation function.

	Input: sizes of weights matrix
	Output: initialized weights
	'''
	return np.random.randn(input_size, output_size) * np.sqrt(2.0/input_size)


def init_bias(output_size):
	'''
	Input: output_size - len of vector for biases used]
	Output: vector representing biases for particular layer
	'''
	return np.zeros((1, output_size))