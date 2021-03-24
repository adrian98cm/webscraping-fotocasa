import numpy as np
import math
import matplotlib.pyplot as plt

rango = np.linspace(-10,10).reshape([50,1])

def derivada_relu(x):
  x[x<=0] = 0
  x[x>0] = 1
  return x

relu = (
  lambda x: x * (x > 0),
  lambda x:derivada_relu(x)
  )

# datos_relu = relu[0](rango)
# datos_relu_derivada = relu[1](rango)


# # Volvemos a definir rango que ha sido cambiado
# rango = np.linspace(-10,10).reshape([50,1])

# # Cremos los graficos
# plt.cla()
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize =(15,5))
# axes[0].plot(rango, datos_relu[:,0])
# axes[1].plot(rango, datos_relu_derivada[:,0])
# plt.show()