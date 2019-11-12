import numpy as np
import sys 
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import randint
from sklearn import metrics # para revisar el error y precision del modelo

from sklearn.model_selection import KFold # use for cross validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler # para normalizacion
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline # pipeline: aplica una transformacion a los datos


# Deep-learing:
import itertools
import keras
from keras.models import Sequential  #pila lineal de capas
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD 
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D


dataset = pd.read_csv('DATA1.csv', sep=',',
                  parse_dates={'dt':['Fecha']},infer_datetime_format=True, 
                  low_memory=False, index_col='dt')



# 1) Se convirtio la data a tipo serie-temporal, tomando el tiempo como indice

#Verificacion de datos
print("encabezados")
print(dataset.head())

print("info")
print(dataset.info())

print("dtypes")
print(dataset.dtypes)

print("shape") #(78,7)
print(dataset.shape)

print("describe") #count mean std min 25% 50% 75% max
print(dataset.describe())

print("columnas")
print(dataset.columns)

dataset.plot('Nro_Clientes',title='Parametros de Facturación', kind='kde') 
plt.tight_layout()
plt.show()


dataset.EA_.resample('M').max().plot(title='EA_ por mes',kind='kde') 
plt.tight_layout()
plt.show()

def series_controladas(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	nuevo_dataset = pd.DataFrame(data)
	cols, names = list(), list()
	# secuencia de entrada (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(nuevo_dataset.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# secuencia de predicción (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(nuevo_dataset.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	# juntando los datos
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	return agg


dataset_resample = dataset.resample('m').mean() 
dataset_resample.shape

#Normaliza caracteristicas [0,1]
values = dataset_resample.values 

## datos completos sin remuestreo
values = dataset.values

# normalizar funciones
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# aprendizaje supervisado
reframed = series_controladas(scaled, 1, 1)

values = reframed.values

n_train_time = 365*24
train = values[:n_train_time, :]
test = values[n_train_time:, :]


# particion entre entradas y salidas
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape entrada para ser 3D [muestras, timesteps, carcteristicas]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 


# Modelacion de red neuronal LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# ajustamos la red
history = model.fit(train_X, train_y, epochs=20, batch_size=12, validation_data=(test_X, test_y), verbose=2, shuffle=False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# prediccion
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 7))

# inversion de escala para prediccion
inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# inversion de escala para data Actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculo de RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


## timesteps de 1 mes
## para un proposito de demostracion, solo comparo las predicciones en 200 series.

aa=[x for x in range(12)]
plt.plot(aa, inv_y[:12], marker='.', label="actual")
plt.plot(aa, inv_yhat[:12], 'r', label="prediction")
plt.ylabel('EA_', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()

