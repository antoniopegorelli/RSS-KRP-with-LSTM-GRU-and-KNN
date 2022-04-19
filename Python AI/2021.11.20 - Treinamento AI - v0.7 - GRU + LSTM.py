# -*- coding: utf-8 -*-
"""
Created on Mon Jun  21 21:47:45 2021

@author: anton
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K

from tensorflow.python.client import device_lib


def rmse(y_pred, y_true):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

print("-----------------------------------------------------------------------------------------")
print(device_lib.list_local_devices())
print("-----------------------------------------------------------------------------------------")

start_time = time.time()


# Carrega os dados de simulação para treinamento ------------------------------------------------------
simdBm = '1'
simAT = '20'
simTipo = '63'
#simPrepDados = 'Pack 30s'
simPrepDados = 'Stream'
simAI = 'GRU'
# simAI = 'LSTM'

simPath = 'DADOS SIMULAÇÃO/' + simTipo + ' - AT' + simAT + ' - ' + simdBm + 'dBm - ' + simPrepDados + '/'
modPath = simPath + simAI + '/'

X_train = np.load(simPath + 'X_train.npy')
y_train = np.load(simPath + 'y_train.npy')

X_val = np.load(simPath + 'X_val.npy')
y_val = np.load(simPath + 'y_val.npy')

X_test = np.load(simPath + 'X_test.npy')
y_test = np.load(simPath + 'y_test.npy')


print(X_train)

print("Time Stamp = {}s". format(round(time.time() - start_time, 0)))

# Limpa a sessão do keras para inicialização ----------------------------------------------------------
K.clear_session()


# Inicia construção do modelo --------------------------------------------------------------------
modelo = Sequential()

if simAI == 'GRU':
    modelo.add(GRU(units = 100, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    modelo.add(Dropout(0.2))
    
    modelo.add(GRU(units = 100, return_sequences = True))
    modelo.add(Dropout(0.2))
    
  
elif simAI == 'LSTM':
    modelo.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    modelo.add(Dropout(0.2))
    
    modelo.add(LSTM(units = 100, return_sequences = True))
    modelo.add(Dropout(0.2))

modelo.add(Dense(units = 2))    

# Compila o modelo --------------------------------------------------------------------
modelo.compile(optimizer = 'adam', loss = rmse, metrics=['accuracy'])

modelo.summary()

# Treinamento do modelo --------------------------------------------------------------------
history = modelo.fit(X_train, y_train, epochs = 1000, validation_data=(X_val, y_val))


if not os.path.exists(modPath):
    os.makedirs(modPath)

if not os.path.exists(modPath + 'modelo_' + simAI + '/variables'):
    os.makedirs(modPath + 'modelo_' + simAI + '/variables')

print("Saving model to: {}". format(modPath + 'modelo'))

modelo.save(modPath + 'modelo_' + simAI + '.h5')

modelo.save(modPath + 'modelo_' + simAI)

y_predict = modelo.predict(X_test)

np.save(modPath + 'y_predict', y_predict)

#print(y_predict)

plt.figure(dpi=600)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(modPath + 'grafico_acuracia.png')
plt.show()

plt.figure(dpi=600)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(modPath + 'grafico_perda.png')
plt.show()


print("Time Stamp = {}s". format(round(time.time() - start_time, 0)))
