import numpy as np
import matplotlib.pyplot as plt #
import pandas as pd


from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler

import itertools
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM

dataset = pd.read_csv('data.txt', sep=';', parse_dates={'dt' : ['Fecha', 'Hora']}, infer_datetime_format=True,
                  low_memory=False, na_values=['nan','?'], index_col='dt')

def ea_grafico(dataset):
	# EA por mes
	dataset.EA.resample('M').sum().plot(title='Energia activa por mes')
	plt.tight_layout()
	plt.show()

	#  ER por mes (mean, std) #comportamiento similar
	r2 = dataset.ER.resample('M').agg(['mean', 'std'])
	r2.plot(subplots = True, title='ER por mes', color='red')
	plt.show()





print("Encabezados:")
print(dataset.head())

print("Informacion de archivo de entrada:")
print(dataset.info())

print("Tipos de dato por columna:")
print(dataset.dtypes)

print("Dimensiones matriz:")
print(dataset.shape)

print("Descripción")
print(dataset.describe())

print("Columnas:")
print(dataset.columns)


ea_grafico(dataset)



# A continuación comparo la media de las diferentes características remuestreadas durante el día.
# especificar columnas para plotear
cols = [0, 1, 2, 3, 5, 6]
i = 1
groups=cols
values = dataset.resample('D').mean().values


print("AQUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")

print("paso 1")
## remuestreando por semana y calculando la media
#dataset.ER.resample('W').mean().plot(color='y', legend=True)
#dataset.EA.resample('W').mean().plot(color='r', legend=True)
#plt.show()




# Se ve arriba que con las técnicas de remuestreo uno puede cambiar las correlaciones entre las características. Esto es importante para la ingeniería de características.

# Machine-Learning: preparación de datos LSTM e ingeniería de características

# * Aplicaré la red neuronal recurrente (LSTM) que es más adecuada para series de tiempo y problemas secuenciales. Este enfoque es el mejor si tenemos datos de gran tamaño.
# * Enmarcaré el problema de aprendizaje supervisado como predicción de EA en el momento actual (t) dada la medición EA y otras características en el paso de tiempo anterior.

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	nuevo_dataset = pd.DataFrame(data)
	cols, names = list(), list()
	# secuencia de entrada (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(nuevo_dataset.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# secuencia de pronóstico (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(nuevo_dataset.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# juntando todos los datos
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# elimina filas con vlores NaN
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# * Para reducir el tiempo de cálculo, y también obtener un resultado rápido para probar el modelo. Se puede volver a remuestrear los datos por hora (los datos originales se dan en minutos). Esto reducirá el tamaño de los datos de 2075259 a 34589, pero mantendrá la estructura general de los datos como se muestra en la figura anterior.

## remuestreo de datos por horas
dataset_resample = dataset.resample('h').mean()
dataset_resample.shape

## * Nota: Escalo todas las caracteristicas en el rango de [0,1].

## Si desea entrenar según los datos remuestreados (por hora), utilice a continuación
values = dataset_resample.values

## datos completos sin remuestreo
#values = dataset.values

# codificar los datos integer
# garantizar que todos los datos sean float
#values = values.astype('float32')

# normalizar caracteristicas
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

# marco como aprendizaje supervisado
reframed = series_to_supervised(scaled, 1, 1)

# eliminamos que no queremos predecir
reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
print(reframed.head())


# * Más arriba mostré 7 variables de entrada (serie de entrada) y la variable de salida 1 para 'EA' en el tiempo actual en horas (dependiendo del remuestreo).
# Particionar el resto de los datos para entrenar y validar conjuntos
# * Primero, dividí el conjunto de datos preparado en conjuntos de entrenamiento y prueba. Para acelerar el entrenamiento del modelo (por el bien de la demostración), solo entrenaremos el modelo en el primer año de datos, luego lo evaluaremos en los próximos 3 años de datos.

# particion en conjuntos de entrenmiento y prueba
values = reframed.values

n_train_time = 365*24*4# 35040 Conjunto de entrenamiento
train = values[:n_train_time, :]
test = values[n_train_time:, :]
##test = values[n_train_time:n_test_time, :]
# particion entre entradas y salidas
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape entrada para ser 3D [muestras, pasos de tiempo, carcteristicas]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Arquitectura de modelo
# 1) LSTM con 100 neuronas en la primera capa visible
# 3) dropout 20%
# 4) 1 neurona en la capa de salida para predecir EA.
# 5) La forma de entrada será 1 paso de tiempo con 7 caracteristicsa.
# 6) Uso la función de pérdida Mean Absolute Error (MAE) y la versión eficiente de Adam de gradiente descendiente.
# 7) El modelo será apto para 20 iteraciones con un batch de 70.

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# ajustamos la red
history = model.fit(train_X, train_y, epochs=25, batch_size=70, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# resumir el historial de pérdida
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Error')
plt.ylabel('error')
plt.xlabel('iteracion')
plt.legend(['entrenamiento', 'prueba'], loc='upper right')
plt.show()


yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 7))

inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('RMSE: %.3f' % rmse)
#49537
# 35040 de entrenamiento
# 14495 prediccion
aa=[x for x in range(400)] #40000
bb=[x for x in range(340)] #34000
plt.plot(bb, inv_y[:340], marker='.', label="actual")
plt.plot(aa, inv_yhat[:400], 'r', label="prediccion")
plt.ylabel('Energia Activa', size=15)
plt.xlabel('Tiempo', size=15)
plt.legend(fontsize=15)
plt.show()
