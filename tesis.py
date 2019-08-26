import numpy as np
import sys 
import matplotlib.pyplot as plt #
import pandas as pd

from scipy.stats import randint
from sklearn import metrics # para revisar el error y precision del modelo

from sklearn.cross_validation import KFold # use for cross validation
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


dataset = pd.read_csv('consolidado_final.csv', sep=';', 
                  parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                  low_memory=False, na_values=['nan','?'], index_col='dt')


# 1) En la fase de lectura de datos, convertir los valores NULL(strings) a valores numpy nan 
# 2) En database combinar las columnas de FEcha y tiempo'
# 3) Convertir la data a tipo serie-temporal, tomando el tiempo como indice
