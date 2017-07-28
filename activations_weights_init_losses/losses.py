import numpy as np


def softmax_loss(x, no_of_samples, y, reg):
	logs = -np.log(x[range(no_of_samples), y])
	data_loss = np.sum(logs)/no_of_samples
	reg_loss = 0.5*reg * np.sum(W1*W1) # ......... Do this term for all weights in network
	loss = data_loss + reg_loss
	return loss

def mean_squared_error(scores, y, der=False):

	if der:
		return y - scores

	return np.mean(np.squared(y - scores))