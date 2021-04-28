from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras as keras
from io import open
import csv
import numpy as np
import pandas
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(optimizer='adam'):
    # create model
    model = Sequential()
    model.add(Dense(6, input_dim=6, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='relu'))
    # Compile model
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy','mse'])
    return model

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# Charge the csv
np.set_printoptions(threshold=sys.maxsize)

csv_route = 'finalDataset3.csv'

# csv_route = 'pisos.csv'
dataNaN = pandas.read_csv(csv_route, header=0, encoding='latin1')
data = dataNaN.fillna(value=1).drop(['Tipo','Distrito'],axis=1).to_numpy()
# data = dataNaN.fillna(value=1).drop(['Tipo','Distrito','Precio_m2','Habitaciones','Parking','Colegios'],axis=1).to_numpy()

# 6 variables 
X = data[:,2:8]
# X = data[:,2:4]
Y = data[:,1]


# Compile the keras model 
#  'mean_absolute_error': Computes the mean absolute error between the labels and predictions.
# optimizer='sgd','rmsprop','adam'
# loss = 'mean_squared_logarithmic_error', 'huber', 'log_cosh', 'mean_squared_error','mean_absolute_error'.
# https://www.tensorflow.org/api_docs/python/tf/keras/Model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))