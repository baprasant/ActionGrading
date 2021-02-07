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
import time
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.callbacks import Callback

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


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

def evaluate_cnn_srnn_model(trainX, trainy, testX, testy):
	cnn_filters = 128
	cnn_strides = 1
	srnn_layers = 100
	verbose, epochs, batch_size = 1, 15, 20
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	n_steps, n_length = 250, 4
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=cnn_filters, kernel_size=3, strides = cnn_strides, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(SimpleRNN(srnn_layers))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	time_callback = TimeHistory()
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback])
	print('Evaluating...')
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
	#############################################################
	import pandas
	from sklearn import model_selection
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import confusion_matrix
	predicted = model.predict(testX)
	matrix = confusion_matrix(testy.argmax(axis=1), predicted.argmax(axis=1))
	print(matrix)
	import seaborn as sns
	import numpy as np
	sns.heatmap(matrix, annot=True)
	import matplotlib.pyplot as plt
	plt.xlabel('Predicted Values')
	plt.ylabel('True Values')
	plt.savefig('cf_cnn_srnn.png')
	plt.close()
	training_time = sum(time_callback.times)
	print('training_time:')
	print(training_time)
	print('Test Data Validation...')
	_, accuracy_tdv = model.evaluate(testX, testy, batch_size=batch_size, verbose=2)
	print('accuracy_tdv:')
	print(accuracy_tdv)
	f = open("results.txt", "a")
	f.write("-------CNN + SRNN-------")
	f.write('\n')
	model.summary(print_fn=lambda x: f.write(x + '\n'))
	print('\n')
	f.write('CNN Filters:')
	f.write(str(cnn_filters))
	f.write('\n')
	f.write('SRNN Layers:')
	f.write(str(srnn_layers))
	f.write('\n')
	f.write('Strides:')
	f.write(str(cnn_strides))
	f.write('\n')
	f.write('epochs:')
	f.write(str(epochs))
	f.write('\n')
	f.write('training_time:')
	f.write(str(training_time)+'sec')
	f.write('\n')
	f.write('accuracy_tdv:')
	f.write(str(accuracy_tdv*100)+'%')
	f.write('\n')
	f.close()
	####################################################################


def evaluate_cnn_lstm_model(trainX, trainy, testX, testy):
	cnn_filters = 128
	cnn_strides = 1
	lstm_layers = 100
	verbose, epochs, batch_size = 1, 15, 20
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	n_steps, n_length = 250, 4
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=cnn_filters, kernel_size=3, strides = cnn_strides, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(lstm_layers))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	time_callback = TimeHistory()
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback])
	print('Evaluating...')
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
	#############################################################
	import pandas
	from sklearn import model_selection
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import confusion_matrix
	predicted = model.predict(testX)
	matrix = confusion_matrix(testy.argmax(axis=1), predicted.argmax(axis=1))
	print(matrix)
	import seaborn as sns
	import numpy as np
	sns.heatmap(matrix, annot=True)
	import matplotlib.pyplot as plt
	plt.xlabel('Predicted Values')
	plt.ylabel('True Values')
	plt.savefig('cf_cnn_lstm.png')
	plt.close()
	training_time = sum(time_callback.times)
	print('training_time:')
	print(training_time)
	print('Test Data Validation...')
	_, accuracy_tdv = model.evaluate(testX, testy, batch_size=batch_size, verbose=2)
	print('accuracy_tdv:')
	print(accuracy_tdv)
	f = open("results.txt", "a")
	f.write("-------CNN + LSTM-------")
	f.write('\n')
	model.summary(print_fn=lambda x: f.write(x + '\n'))
	print('\n')
	f.write('CNN Filters:')
	f.write(str(cnn_filters))
	f.write('\n')
	f.write('LSTM Layers:')
	f.write(str(lstm_layers))
	f.write('\n')
	f.write('Strides:')
	f.write(str(cnn_strides))
	f.write('\n')
	f.write('epochs:')
	f.write(str(epochs))
	f.write('\n')
	f.write('training_time:')
	f.write(str(training_time)+'sec')
	f.write('\n')
	f.write('accuracy_tdv:')
	f.write(str(accuracy_tdv*100)+'%')
	f.write('\n')
	f.close()
	####################################################################


def evaluate_cnn_bilstm_model(trainX, trainy, testX, testy):
	cnn_filters = 128
	cnn_strides = 1
	bilstm_layers = 100
	verbose, epochs, batch_size = 1, 15, 20
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	n_steps, n_length = 250, 4
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=cnn_filters, kernel_size=3, strides = cnn_strides, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(Bidirectional(LSTM(bilstm_layers)))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	time_callback = TimeHistory()
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback])
	print('Evaluating...')
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
	#############################################################
	import pandas
	from sklearn import model_selection
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import confusion_matrix
	predicted = model.predict(testX)
	matrix = confusion_matrix(testy.argmax(axis=1), predicted.argmax(axis=1))
	print(matrix)
	import seaborn as sns
	import numpy as np
	sns.heatmap(matrix, annot=True)
	import matplotlib.pyplot as plt
	plt.xlabel('Predicted Values')
	plt.ylabel('True Values')
	plt.savefig('cf_cnn_bilstm.png')
	plt.close()
	training_time = sum(time_callback.times)
	print('training_time:')
	print(training_time)
	print('Test Data Validation...')
	_, accuracy_tdv = model.evaluate(testX, testy, batch_size=batch_size, verbose=2)
	print('accuracy_tdv:')
	print(accuracy_tdv)
	f = open("results.txt", "a")
	f.write("-------CNN + Bi-LSTM-------")
	f.write('\n')
	model.summary(print_fn=lambda x: f.write(x + '\n'))
	print('\n')
	f.write('CNN Filters:')
	f.write(str(cnn_filters))
	f.write('\n')
	f.write('Bi-LSTM Layers:')
	f.write(str(bilstm_layers))
	f.write('\n')
	f.write('Strides:')
	f.write(str(cnn_strides))
	f.write('\n')
	f.write('epochs:')
	f.write(str(epochs))
	f.write('\n')
	f.write('training_time:')
	f.write(str(training_time)+'sec')
	f.write('\n')
	f.write('accuracy_tdv:')
	f.write(str(accuracy_tdv*100)+'%')
	f.write('\n')
	f.close()
	####################################################################


def evaluate_srnn_model(trainX, trainy, testX, testy):
	srnn_layers = 100
	verbose, epochs, batch_size = 1, 15, 20
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(SimpleRNN(srnn_layers, input_shape=(n_timesteps,n_features)))
	# model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	time_callback = TimeHistory()
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback])
	print('Evaluating...')
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
	#############################################################
	import pandas
	from sklearn import model_selection
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import confusion_matrix
	predicted = model.predict(testX)
	matrix = confusion_matrix(testy.argmax(axis=1), predicted.argmax(axis=1))
	print(matrix)
	import seaborn as sns
	import numpy as np
	sns.heatmap(matrix, annot=True)
	import matplotlib.pyplot as plt
	plt.xlabel('Predicted Values')
	plt.ylabel('True Values')
	plt.savefig('cf_srnn.png')
	plt.close()
	training_time = sum(time_callback.times)
	print('training_time:')
	print(training_time)
	print('Test Data Validation...')
	_, accuracy_tdv = model.evaluate(testX, testy, batch_size=batch_size, verbose=2)
	print('accuracy_tdv:')
	print(accuracy_tdv)
	f = open("results.txt", "a")
	f.write("-------SRNN-------")
	f.write('\n')
	model.summary(print_fn=lambda x: f.write(x + '\n'))
	print('\n')
	f.write('SRNN Layers:')
	f.write(str(srnn_layers))
	f.write('\n')
	f.write('epochs:')
	f.write(str(epochs))
	f.write('\n')
	f.write('training_time:')
	f.write(str(training_time)+'sec')
	f.write('\n')
	f.write('accuracy_tdv:')
	f.write(str(accuracy_tdv*100)+'%')
	f.write('\n')
	f.close()
	####################################################################


def evaluate_lstm_model(trainX, trainy, testX, testy):
	lstm_layers = 100
	verbose, epochs, batch_size = 1, 15, 20
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(lstm_layers, input_shape=(n_timesteps,n_features)))
	# model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	time_callback = TimeHistory()
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback])
	print('Evaluating...')
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
	#############################################################
	import pandas
	from sklearn import model_selection
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import confusion_matrix
	predicted = model.predict(testX)
	matrix = confusion_matrix(testy.argmax(axis=1), predicted.argmax(axis=1))
	print(matrix)
	import seaborn as sns
	import numpy as np
	sns.heatmap(matrix, annot=True)
	import matplotlib.pyplot as plt
	plt.xlabel('Predicted Values')
	plt.ylabel('True Values')
	plt.savefig('cf_lstm.png')
	plt.close()
	training_time = sum(time_callback.times)
	print('training_time:')
	print(training_time)
	print('Test Data Validation...')
	_, accuracy_tdv = model.evaluate(testX, testy, batch_size=batch_size, verbose=2)
	print('accuracy_tdv:')
	print(accuracy_tdv)
	f = open("results.txt", "a")
	f.write("-------LSTM-------")
	f.write('\n')
	model.summary(print_fn=lambda x: f.write(x + '\n'))
	print('\n')
	f.write('LSTM Layers:')
	f.write(str(lstm_layers))
	f.write('\n')
	f.write('epochs:')
	f.write(str(epochs))
	f.write('\n')
	f.write('training_time:')
	f.write(str(training_time)+'sec')
	f.write('\n')
	f.write('accuracy_tdv:')
	f.write(str(accuracy_tdv*100)+'%')
	f.write('\n')
	f.close()
	####################################################################

def evaluate_bilstm_model(trainX, trainy, testX, testy):
	bilstm_layers = 100
	verbose, epochs, batch_size = 1, 15, 20
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Bidirectional(LSTM(bilstm_layers, input_shape=(n_timesteps,n_features))))
	# model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	# model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	time_callback = TimeHistory()
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[time_callback])
	print('Evaluating...')
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
	#############################################################
	import pandas
	from sklearn import model_selection
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import confusion_matrix
	predicted = model.predict(testX)
	matrix = confusion_matrix(testy.argmax(axis=1), predicted.argmax(axis=1))
	print(matrix)
	import seaborn as sns
	import numpy as np
	sns.heatmap(matrix, annot=True)
	import matplotlib.pyplot as plt
	plt.xlabel('Predicted Values')
	plt.ylabel('True Values')
	plt.savefig('cf_bilstm.png')
	plt.close()
	training_time = sum(time_callback.times)
	print('training_time:')
	print(training_time)
	print('Test Data Validation...')
	_, accuracy_tdv = model.evaluate(testX, testy, batch_size=batch_size, verbose=2)
	print('accuracy_tdv:')
	print(accuracy_tdv)
	f = open("results.txt", "a")
	f.write("-------BI-LSTM-------")
	f.write('\n')
	model.summary(print_fn=lambda x: f.write(x + '\n'))
	print('\n')
	f.write('BI-LSTM Layers:')
	f.write(str(bilstm_layers))
	f.write('\n')
	f.write('epochs:')
	f.write(str(epochs))
	f.write('\n')
	f.write('training_time:')
	f.write(str(training_time)+'sec')
	f.write('\n')
	f.write('accuracy_tdv:')
	f.write(str(accuracy_tdv*100)+'%')
	f.write('\n')
	f.close()
	####################################################################


def run_all():
	# load data
	trainX, trainy, testX, testy = load_dataset()
	evaluate_srnn_model(trainX, trainy, testX, testy)
	evaluate_lstm_model(trainX, trainy, testX, testy)
	evaluate_bilstm_model(trainX, trainy, testX, testy)
	evaluate_cnn_srnn_model(trainX, trainy, testX, testy)
	evaluate_cnn_lstm_model(trainX, trainy, testX, testy)
	evaluate_cnn_bilstm_model(trainX, trainy, testX, testy)

run_all()
