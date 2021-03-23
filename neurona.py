from funcionRelu import relu
from capas import capa
from io import open
import csv
import numpy as np
import pandas

filename = 'buildings_information.csv'

data = pandas.read_csv(filename, header=0)



print(data.shape)

print (data.head(10))

# Numero de neuronas en cada capa. 
# El primer valor es el numero de columnas de la capa de entrada.

# Lectura de pisos
with open('buildings_information.csv', newline ='') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    print(row)

# neuronas = [2,4,8,1] 
# # Funciones de activacion. 
# funciones_activacion = [relu,relu,relu]
# red_neuronal = []

# for paso in list(range(len(neuronas)-1)):
#   x = capa(neuronas[paso],neuronas[paso+1],funciones_activacion[paso])
#   red_neuronal.append(x)


# error = []
# predicciones = []

# for epoch in range(0,1000):
#   ronda = entrenamiento(X = X ,Y = Y ,red_neuronal = red_neuronal, lr = 0.001)
#   predicciones.append(ronda)
#   temp = mse(np.round(predicciones[-1]),Y)[0]
#   error.append(temp)