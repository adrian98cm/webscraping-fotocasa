from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from io import open
import csv
import numpy as np
import pandas
import sys
import matplotlib.pyplot as plt

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# Charge the csv
np.set_printoptions(threshold=sys.maxsize)

csv_route = 'finalDataset3.csv'

dataNaN = pandas.read_csv(csv_route, header=0, encoding='latin1')
data = dataNaN.fillna(value=1).drop(['Tipo','Distrito'],axis=1).to_numpy()

# 6 variables 
X = data[:,2:8]
Y = data[:,1]

model = Sequential()
# Numero de capas, la ultima siempre tiene que ser 1, las funciones de activacion son relu, el primer numero es el numero de neuronas en la capa
model.add(Dense(6, input_dim=6, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(1, activation='relu'))

# Compile the keras model 
#  'mean_absolute_error': Computes the mean absolute error between the labels and predictions.
# optimizer='sgd','rmsprop'
# loss = 'huber','mean_squared_error','mean_absolute_error','log_cosh', 'mean_squared_logarithmic_error'
# https://www.tensorflow.org/api_docs/python/tf/keras/Model

model.compile(loss='mean_squared_logarithmic_error', optimizer='sgd', metrics=['msle'])

# fit the keras model on the dataset
history = model.fit(X, Y, validation_split=0.30, epochs=50, batch_size=10)
# summarize history for accuracy
plt.plot(history.history['msle'])
plt.plot(history.history['val_msle'])
plt.title('Model msle')
plt.ylabel('msle')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# for i in range(20):
# 	print('Valores %s => Precio estimado: %d (Precio de venta: %d)' % (X[i].tolist(), predictions[i], Y[i]))

# while (True):
#     print('Introduce los datos de tu piso')
#     precio_m2  = float(input("Precio_m2: "))
#     habs = float(input("Habitaciones: "))
#     aseos = float(input("Aseos: "))
#     superficie = float(input("Superficie: "))
#     parking = float(input("Parkings: "))
#     colegios = float(input("Colegios: "))
#     precio = float(input("Precio Real: "))
#     piso = [[precio_m2, habs, aseos, superficie ,parking ,colegios]]
#     prediction = model.predict(piso)
#     print('Precio estimado: %d (Precio de venta: %d)' % (prediction, precio))