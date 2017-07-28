import numpy as np



####################
# ACTIVATIONS      #
####################


def relu(x, der=False):
	
	if der:
		return 1. * (0 < x)

	return np.maximum(x, 0)


def sigmoid(x, der=False):
	
	if der:
		return (1 - x) * x

	return 1 / (1 + np.exp(-x))

def softmax(x, der=False):
	
	if der:
		return (1 - x) * x

	return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def tanh(x, der=False):
	
	if der:
		return 1.0 - np.power(np.tanh(x), 2)

	return np.tanh(x)

def leaky_relu(x, alpha=0.1, der=False):
	
	if der:
		gradients = 1.0 * (x > 0)
		gradients[gradients == 0] = alpha
		return gradients

	return np.maximum(x * alpha, x)


def sin_activation(x, der=False):

	if der:
		return np.cos(x)

	return np.sin(x)

def softplus(x, der=False):

	if der:
		return 1 / (1 + np.exp(-x))

	return np.log(x)


def gaussian(x, der=False):

	if der:
		return (-2*x) * np.exp(np.power((-x), 2))

	return np.exp(np.power((-x), 2))