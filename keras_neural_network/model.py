from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from io import open
import csv
import numpy as np
import pandas
import sys

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# Charge the csv
np.set_printoptions(threshold=sys.maxsize)
csv_route = 'finalDataset.csv'
dataNaN = pandas.read_csv(csv_route, header=0, encoding='latin1')
data = dataNaN.fillna(value=1).drop(['fullDataset.ID','fullDataset.Precio_m2','fullDataset.Distrito','fullDataset.Planta','fullDataset.Tipo'],axis=1).to_numpy()
X = data[:,2:7]
Y = data[:,1]
# print(Y)

model = Sequential()
model.add(Dense(12, input_dim=5, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

# Compile the keras model 
#  'mean_absolute_error': Computes the mean absolute error between the labels and predictions.
# optimizer='sgd','rmsprop'
# https://www.tensorflow.org/api_docs/python/tf/keras/Model

model.compile(loss='mean_squared_logarithmic_error', optimizer='sgd', metrics=['mean_absolute_percentage_error'])

# fit the keras model on the dataset
model.fit(X, Y, epochs=150, batch_size=10)


predictions = model.predict(X)
for i in range(20):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
