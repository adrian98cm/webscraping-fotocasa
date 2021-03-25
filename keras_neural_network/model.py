from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from io import open
import csv
import numpy as np
import pandas

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# Charge the csv
dataNaN = pandas.read_csv('buildings_information.csv', header=0, encoding='latin1')
data = dataNaN.fillna(value=1).drop(['Distrito','Tipo','URL'],axis=1).to_numpy()
X = data[:,1:9]
Y = data[:,0]
print(X)
print(Y)
model = Sequential()
model.add(Dense(12, input_dim=5, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

# Compile the keras model 
#  'mean_absolute_error': Computes the mean absolute error between the labels and predictions.
# https://www.tensorflow.org/api_docs/python/tf/keras/Model

model.compile(loss='mean_squared_logarithmic_error', optimizer='sgd', metrics=['mean_absolute_percentage_error'])

# fit the keras model on the dataset
model.fit(X, Y, epochs=150, batch_size=10)

accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))
