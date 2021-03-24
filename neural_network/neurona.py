from funcionRelu import relu
from capas import capa
from entrenamiento import entrenamiento
from entrenamiento import mse
from io import open
import csv
import numpy as np
import pandas
import sys
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=sys.maxsize)
# https://anderfernandez.com/blog/como-programar-una-red-neuronal-desde-0-en-python/
dataNaN = pandas.read_csv('buildings_information.csv', header=0, encoding='latin1')
data = dataNaN.fillna(value=1).drop(['Distrito','Tipo','URL'],axis=1)
YTemp = data["Precio"].to_numpy()
Y = YTemp.reshape(len(YTemp),1)
# X = np.vstack((data["Habitaciones"].to_numpy(),data["Aseos"].to_numpy()))
X = data.loc[:,["Habitaciones", "Aseos"]].to_numpy()
# x1=np.transpose(X)
# print(X)
# print(Y)
# X = np.stack((data["Habitaciones"].to_numpy(),data["Aseos"].to_numpy()))


# Lectura de pisos
# with open('buildings_information.csv', newline ='', encoding='latin1') as csvfile:
#   reader = csv.DictReader(csvfile)
#   for row in reader:
#     print(row)


# Numero de neuronas en cada capa. 
# El primer valor es el numero de columnas de la capa de entrada. Tenemos 4 ya que usamos 4 variables ahora
# Tenemos dos capas ocultas de 2 y 2 neuronas 
# Tenenos una de salida que predecira el precio
neuronas = [2,2,1]
# Funciones de activacion. 
funciones_activacion = [relu,relu,relu,relu]
red_neuronal = []

# Inicializar la neurona
for paso in list(range(len(neuronas)-1)):
  x = capa(neuronas[paso],neuronas[paso+1],funciones_activacion[paso])
  red_neuronal.append(x)

# Nos quedamos con los errores y predicciones
error = []
predicciones = []

# Entrenamos la neurona
for epoch in range(0,1500):
  print('epoch',epoch)
  ronda = entrenamiento(X = X ,Y = Y ,red_neuronal = red_neuronal, lr = 0.001)
  predicciones.append(ronda)
  temp = mse(np.round(predicciones[-1]),Y)[0]
  error.append(temp)

rare = list(range(0,1500))
plt.plot(rare, error)
# plt.show()

# Casos de prueba
output = [X]

for num_capa in range(len(red_neuronal)):
  z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b
  # print(z)
  a = red_neuronal[num_capa].funcion_act[0](z)
  output.append(a)

print(output[-1])
