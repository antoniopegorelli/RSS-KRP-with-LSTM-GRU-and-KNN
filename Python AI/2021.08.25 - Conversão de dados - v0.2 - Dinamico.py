# -*- coding: utf-8 -*-
"""
Created on Mon Jun  21 21:47:45 2021

@author: anton
"""

import numpy as np
import pandas as pd
import time
import os

from sklearn.preprocessing import MinMaxScaler

# Subrotina para exibição de uma barra de progresso durante o processamento de cada etapa para acompanhamento
def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        try:
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r{prefix:<100} |{bar}| {percent}% {suffix}', end = printEnd)
        except:
            print(f'\r{prefix:<100} -------------------  SEM DADOS  -------------------', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
            
    # Print New Line on Complete
    print()


start_time = time.time()

# Carrega os dados de simulação para realização da conversão ------------------------------------------------------
simdBm = '1'
simAT = '10'
simTipo = '1'
#simPrepDados = 'Pack 30s'
simPrepDados = 'Stream'
simPath = 'DADOS SIMULAÇÃO/' + simTipo + ' - AT' + simAT + ' - ' + simdBm + 'dBm - ' + simPrepDados + '/'

# Abre os dados de simulação para conversão ------------------------------------------------------
dataset = pd.read_csv('test ' + simTipo + '.csv', delimiter=";")
sc = MinMaxScaler(feature_range = (0, 1))

print(dataset)

X_train = []
y_train = []

X_val = []
y_val = []

X_test = []
y_test = []


# Prepara os dados para segmentação tipo Pacotes de 30 segundos

if simPrepDados == 'Pack 30s':
    for i in progressBar(range(1, 10001), prefix = 'Preparando dados de treinamento: ', decimals = 0, length = 50):
        test_data = dataset.loc[dataset['test'] == i]
        test_data_outputs = test_data.iloc[:, 5:11].values
        test_data_scaled = sc.fit_transform(test_data_outputs)
        X_train.append(test_data_scaled)
        y_train.append(test_data.iloc[:, 2:4].values)
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 6))
    
    
    for i in progressBar(range(10001, 12501), prefix = 'Preparando dados de validação: ', decimals = 0, length = 50):
        test_data = dataset.loc[dataset['test'] == i]
        test_data_outputs = test_data.iloc[:, 5:11].values
        test_data_scaled = sc.fit_transform(test_data_outputs)
        X_val.append(test_data_scaled)
        y_val.append(test_data.iloc[:, 2:4].values)
    
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 6))
    
    
    for i in progressBar(range(12501, 15001), prefix = 'Preparando dados de teste: ', decimals = 0, length = 50):
        test_data = dataset.loc[dataset['test'] == i]
        test_data_outputs = test_data.iloc[:, 5:11].values
        test_data_scaled = sc.fit_transform(test_data_outputs)
        X_test.append(test_data_scaled)
        y_test.append(test_data.iloc[:, 2:4].values)    
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 6))


# Prepara os dados para segmentação tipo stream

if simPrepDados == 'Stream':
    train_inputs = dataset.iloc[:, 5:11].values
    train_outputs = dataset.iloc[:, 2:4].values
    
    train_inputs_scaled = sc.fit_transform(train_inputs)
    train_outputs_scaled = sc.fit_transform(train_outputs)
    
    for i in progressBar(range(10, 10011), prefix = 'Preparando dados de treinamento: ', decimals = 0, length = 50):
        X_train.append(train_inputs_scaled[i-10:i, :6])
        y_train.append(train_outputs[i-10:i, :2])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 6))
    
    
    for i in progressBar(range(10011, 12511), prefix = 'Preparando dados de validação: ', decimals = 0, length = 50):
        X_val.append(train_inputs_scaled[i-10:i, :6])
        y_val.append(train_outputs[i-10:i, :2])
    
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 6))
    
    
    for i in progressBar(range(12511, 15011), prefix = 'Preparando dados de teste: ', decimals = 0, length = 50):
        X_test.append(train_inputs_scaled[i-10:i, :6])
        y_test.append(train_outputs[i-10:i, :2])

    
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 6))


print(X_train)

if not os.path.exists(simPath):
    os.makedirs(simPath)

print("Saving to: {}". format(simPath))

np.save(simPath + 'X_train', X_train)
np.save(simPath + 'y_train', y_train)

np.save(simPath + 'X_test', X_test)
np.save(simPath + 'y_test', y_test)

np.save(simPath + 'X_val', X_val)
np.save(simPath + 'y_val', y_val)

print("Time Stamp = {}s". format(round(time.time() - start_time, 0)))
