import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split


def prepare_dataset(dataset):
	''' 
	This fucnction is used when you have strings in dataframe.
	It is using LabelEncoder to transform columns from dataframe into columns encoded in specific order

	Input: Pandas dataframe
	Output: Pandas dataframe editetd
	'''
	new_dataset = dataset
	ch = list(dataset.columns.values)
	encoder = LabelEncoder()
	for i in ch:   
		column = new_dataset[i]
		column = encoder.fit_transform(column)
		new_dataset[i]=column

	return new_dataset



def labels_vs_features(dataset):
	''' 
	This fucnction is used specificly for problem with muschrooms
	This function splits dataset to X and y (features and labels)

	Input: Pandas dataframe
	Output: X - features for our dataset
	y - labels (classes) for the dataset
	'''


	X = []
	y = []
	y = dataset['class']
	ch = list(dataset.columns.values)
	for i in ch:
		if i == 'class':
			continue
		else:
			X.append(dataset[i])
	return np.array(X).T, np.array(y)



def data_splitter(X, y, test_size=0.2,verbose=True):
	''' 
	This fucnction is used to split dataset into training and testing parts

	Input: X- features
			y- labels
			test_size - how much samples from dataset you want to use for testing: Default is 20% - 0.2
			verbose - showing sizes of splited data or not
	Output: X_train, y_train, X_test, y_test - features split into train and test set
	'''

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	
	if verbose:
		print("X_train size -->", X_train.shape)
		print("y_train size -->", y_train.shape)
		print("X_test size -->", X_test.shape)
		print("y_test size -->", y_test.shape)

	return X_train, X_test, y_train, y_test