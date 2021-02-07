# cnn lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from matplotlib import pyplot

from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector


# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# acceleration
	filenames += ['sensor_1_x.txt', 'sensor_1_y.txt', 'sensor_1_z.txt']
	# gyroscope
	filenames += ['sensor_2_x.txt', 'sensor_2_y.txt', 'sensor_2_z.txt']
	# magnetometer
	filenames += ['sensor_3_x.txt', 'sensor_3_y.txt', 'sensor_3_z.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/output.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'OURDataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'OURDataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 1, 15, 20
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = 250, 4
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	# define model
	model = Sequential()
	print('shape of trainX:')
	print(trainX.shape)
	print('shape of trainy:')
	print(trainy.shape)
	model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, strides = 1, activation='relu'), input_shape=(None,n_length,n_features)))
	# model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, strides = 1, activation='relu')))
	# model.add(TimeDistributed(Dropout(0.5)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	# model.add(Bidirectional(LSTM(100, input_shape=(n_timesteps,n_features))))
	model.add(Bidirectional(LSTM(100)))
	# model.summary()
	# model.add(AttentionDecoder(50, n_features))
	# model.add(Dropout(0.5))
	model.summary()
	model.add(Dense(100, activation='relu'))
	model.summary()
	model.add(Dense(n_outputs, activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	print('Evaluating...')
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
	# print('Predicting...')
	# y_pred = model.predict_classes(testX[0:10], verbose = 1)
	# from sklearn.metrics import confusion_matrix
	# cm=confusion_matrix(testy,y_pred)
	# print(cm)
	import pandas
	from sklearn import model_selection
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import confusion_matrix
	predicted = model.predict(testX)
	matrix = confusion_matrix(testy.argmax(axis=1), predicted.argmax(axis=1))
	# predicted = (predicted > 0.5)
	# matrix = confusion_matrix(testy, predicted)
	# print(predicted)
	# print(testy)
	print(matrix)
	import seaborn as sns
	import numpy as np
	sns.heatmap(matrix, annot=True)
	# group_counts = ["{0:0.0f}".format(value) for value in matrix.flatten()]
	# group_percentages = ["{0:.2%}".format(value) for value in matrix.flatten()/np.sum(matrix)]
	# labels = [f"{v2}{v3}" for v2, v3 in zip(group_counts,group_percentages)]
	# labels = np.asarray(labels).reshape(6,6)
	# sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues')
	import matplotlib.pyplot as plt
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()
	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=1):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# run the experiment
run_experiment()
