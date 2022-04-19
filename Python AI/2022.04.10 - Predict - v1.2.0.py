# -*- coding: utf-8 -*-
"""
Created on Mon Jun  21 21:47:45 2021

@author: anton
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import time
import os
from random import randint

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from sklearn.neighbors import KNeighborsRegressor

from scipy.stats import f_oneway

dist_y = []
hist_bins = []
dist_y_LSTM = []
dist_y_GRU = []
dist_y_KNN = []
dist_y_LSTM_O = []
dist_y_GRU_O = []
dist_y_KNN_O = []

global cdf_LSTM
global cdf_GRU
global cdf_KNN


# Subrotina para avaliar resultados variando a quantidade de vizinhos da KNN
def teste_KNN_neigh(X_train_L, y_train_L, X_test_L, y_test_L):
    
    cdf_L = []
    K_L = []
    
    for i in range(2,6):
        classifier_L = KNeighborsRegressor(n_neighbors = i, metric = 'minkowski', p = 2)
        classifier_L.fit(X_train_L, y_train_L)
        
        # Predicting the Test set results
        y_predict_L = classifier_L.predict(X_test_L)
        
        for j in range(25000):
                dist_y.append(dist_euclideana(y_test_L[j,:], y_predict_L[j,:]))
        
        # Calcula a FDA  --------------------------------------------------------------------
        count, bins_count = np.histogram(dist_y, bins=hist_bins)
        pdf = count / sum(count)
        cdf_L.append(np.cumsum(pdf))
        K_L.append(i)
        
    # Gera o gráfico único de FDA --------------------------------------------------------------------
    plt.figure(dpi=600)
    plt.xlim([0,500])
    plt.ylim([0,1])
    plt.xlabel("Distância (cm)")
    plt.ylabel("FDA")
    plt.title('FDA comparação KNN Vizinhos ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + ' - ' + simPrepDados)
    for i in cdf_L:
        plt.plot(i)
    plt.legend(K_L, loc='upper right')
    plt.savefig(simPath + 'KNN/FDA - KNN_NEIGH ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.png', bbox_inches='tight')
    plt.show()         
        

# Subrotina para calcular distância Euclideana entre teste e predição
def dist_euclideana(teste, pred):
    return np.sqrt(np.sum(np.square(teste-pred)))


# Subrotina de teste
def teste(simAI, base, origem, grafs):
    
    print("Iniciando Teste " + simAI + " - Time Stamp = {}". format(time.time() - start_time))
    
    if simAI == 'LSTM' or simAI == 'GRU':
        # Limpa a sessão do keras para inicialização ----------------------------------------------------------
        K.clear_session()
        
        # Carrega modelo construído --------------------------------------------------------------------
        if origem:
            modelo = load_model(simPath_original + simAI + '/modelo_' + simAI +'.h5', compile=False)
            savePath = '/Plots_ORIGINAL/'
        else:
            modelo = load_model(simPath + simAI + '/modelo_' + simAI +'.h5', compile=False)
            savePath = '/Plots/'
        
        print("Gerando resultados " + simAI + " - Time Stamp = {}". format(time.time() - start_time))
        
        # Calcula os dados de predição --------------------------------------------------------------------
        y_predict = modelo.predict(X_test)
    
    elif simAI == 'KNN':
        # Realiza a construção do teste KNN --------------------------------------------------------------------   
        if origem:
            savePath = '/Plots_ORIGINAL/'
            X_train = np.load(simPath_original + 'X_train.npy')
            y_train = np.load(simPath_original + 'y_train.npy')
        else:
            savePath = '/Plots/'    
            X_train = np.load(simPath + 'X_train.npy')
            y_train = np.load(simPath + 'y_train.npy')

        # Construindo base de dados para realização do treinamento e teste do KNN
        X_train_KNN = []
        y_train_KNN = []
        
        X_test_KNN = []
        y_test_KNN = []
        
        
        for i in range(10000):
            for j in range(sample_size):
                X_train_KNN.append([X_train[i,j,0], X_train[i,j,1], X_train[i,j,2], X_train[i,j,3], X_train[i,j,4], X_train[i,j,5]])
                y_train_KNN.append([y_train[i,j,0], y_train[i,j,1]])

        for i in range(2500):
            for j in range(sample_size):
                X_test_KNN.append([X_test[i,j,0], X_test[i,j,1], X_test[i,j,2], X_test[i,j,3], X_test[i,j,4], X_test[i,j,5]])
                y_test_KNN.append([y_test[i,j,0], y_test[i,j,1]])

        
        X_test_KNN = np.array(X_test_KNN)
        y_test_KNN = np.array(y_test_KNN)

        X_train_KNN = np.array(X_train_KNN)
        y_train_KNN = np.array(y_train_KNN)
        
        # Teste de avaliação da quantidade de vizinhos para a KNN
        # teste_KNN_neigh(X_train_KNN, y_train_KNN, X_test_KNN, y_test_KNN)
        
        # Treinando a KNN e gerando os resultados de predição
        classifier = KNeighborsRegressor(n_neighbors = 2, metric = 'minkowski', p = 2)
        classifier.fit(X_train_KNN, y_train_KNN)
        y_predict = classifier.predict(X_test_KNN)

    
    # print("X=%s, Predicted=%s" % (y_predict[:,:,0], y_test[:,:,0]))
    
    if simAI == 'LSTM' or simAI == 'GRU':    
        # Gera os gráficos de predição vs localização real --------------------------------------------------------------------
        if grafs:
            for i in range(50):
                plt.figure(dpi=200)
                plt.xlim([0,10])
                plt.ylim([0,10])
                plt.xlabel("Instante (s)")
                plt.ylabel("Posição (m)")
                plt.title('Predição ' + simAI + ' ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + ' - teste ' + str(i + 1))
                plt.plot(y_predict[base + i,:,0])
                plt.plot(y_predict[base + i,:,1])
                plt.plot(y_test[base + i,:,0])
                plt.plot(y_test[base + i,:,1])
                plt.legend(['predict y1', 'predict y2', 'test y1', 'test y2'], loc='upper right')
                plt.savefig(simPath + simAI + savePath + 'plot base ' + str(base) + ' - ' + simAI + ' ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + ' - teste ' + str(i + 1) + '.png', bbox_inches='tight')
                # plt.show()    
        
        
        # Limpa a variável de distâncias ----------------------------------------------------------
        dist_y.clear()
        
        # Calcula os dados de distâncias euclideanas --------------------------------------------------------------------
        for i in range(2500):
            for j in range(sample_size):
                dist_y.append(dist_euclideana(y_test[i,j,:], y_predict[i,j,:]))
            

    elif simAI == 'KNN':
        # Gera os gráficos de predição vs localização real --------------------------------------------------------------------
        if grafs:
            for i in range(50):
                plt.figure(dpi=200)
                plt.xlim([0,10])
                plt.ylim([0,10])
                plt.xlabel("Instante (s)")
                plt.ylabel("Posição (m)")
                plt.title('Predição ' + simAI + ' ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + ' - teste ' + str(i + 1))
                plt.plot(y_predict[base+i-10:base+i,0])
                plt.plot(y_predict[base+i-10:base+i,1])
                plt.plot(y_test_KNN[base+i-10:base+i,0])
                plt.plot(y_test_KNN[base+i-10:base+i,1])
                plt.legend(['predict y1', 'predict y2', 'test y1', 'test y2'], loc='upper right')
                plt.savefig(simPath + simAI + savePath + 'plot base ' + str(base) + ' - ' + simAI + ' ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + ' - teste ' + str(i + 1) + '.png', bbox_inches='tight')
                # plt.show()    
        
        # Limpa a variável de distâncias ----------------------------------------------------------
        dist_y.clear()
        
        # Calcula os dados de distâncias euclideanas --------------------------------------------------------------------
        for i in range(25000):
                dist_y.append(dist_euclideana(y_test_KNN[i,:], y_predict[i,:]))   
    
    
    # Calcula a FDA  --------------------------------------------------------------------
    count, bins_count = np.histogram(dist_y, bins=hist_bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    
    # Gera o gráfico único de FDA --------------------------------------------------------------------
    if salvarResultadosFDA:
        plt.figure(dpi=600)
        plt.xlim([0,500])
        plt.ylim([0,1])
        plt.xlabel("Distância (cm)")
        plt.ylabel("FDA")
        plt.title('FDA ' + simAI + ' ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + ' - ' + simPrepDados)
        plt.plot(cdf)
        # plt.legend([simAI], loc='upper right')
        plt.savefig(simPath + simAI + savePath + 'FDA/FDA - ' + simAI + ' ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.png', bbox_inches='tight')
        plt.show() 

    return cdf, dist_y



# Define início do processamento para avaliação do tempo percorrido ------------------------------------------------------------
start_time = time.time()

# Limpa memória dos gráficos gerandos ateriormente ------------------------------------------------------------
plt.close()
plt.cla()
plt.clf()

# Carrega os dados de simulação para realização dos testes -------------------------*****************************************************************
simdBm = '1'
# simdBm = '2'
# simdBm = '10'

simAT = '10'
# simAT = '20'
simAT = '30'

simTipo = '63' # Todos os APs
# simTipo = '37' # Triangularização
# simTipo = '8' # AP Central
# simTipo = '4' # AP Lateral
# simTipo = '1' # AP Lateral

# simPrepDados = 'Pack 30s'
simPrepDados = 'Stream'

# Define quais resultados serão emitidos ----------------------------******************************************************************
analise_graf = False
salvarResultadosFDA = False
salvarDpResultados = False
gerarGraficos = False
salvarDistancias = False
anovaTest = True


simPath = 'DADOS SIMULAÇÃO/' + simTipo + ' - AT' + simAT + ' - ' + simdBm + 'dBm - ' + simPrepDados + '/'
simPath_original = 'DADOS SIMULAÇÃO/63 - AT10 - 1dBm - Stream/'

X_test = np.load(simPath + 'X_test.npy')
y_test = np.load(simPath + 'y_test.npy')

if simPrepDados == 'Pack 30s':
    sample_size = 30
elif simPrepDados == 'Stream':
    sample_size = 10

# Determina o teste base para geração dos gráficos ---------------------------------------------------------
base = randint(0, 2399)
# base = 1500
print('Base de avaliação: ' + str(base))

# Determina segmentação do gráfico de FDA
for i in range(900):
    hist_bins.append(i/100)


# Prepara as pastas para salvar os resultados ----------------------------------------------------------
if not os.path.exists(simPath + 'LSTM'):
    os.makedirs(simPath + 'LSTM')
if not os.path.exists(simPath + 'GRU'):
    os.makedirs(simPath + 'GRU')
if not os.path.exists(simPath + 'KNN'):
    os.makedirs(simPath + 'KNN')

if not os.path.exists(simPath + 'LSTM/Plots'):
    os.makedirs(simPath + 'LSTM/Plots')
if not os.path.exists(simPath + 'GRU/Plots'):
    os.makedirs(simPath + 'GRU/Plots')
if not os.path.exists(simPath + 'KNN/Plots'):
    os.makedirs(simPath + 'KNN/Plots')
if not os.path.exists(simPath + 'LSTM/Plots_ORIGINAL'):
    os.makedirs(simPath + 'LSTM/Plots_ORIGINAL')
if not os.path.exists(simPath + 'GRU/Plots_ORIGINAL'):
    os.makedirs(simPath + 'GRU/Plots_ORIGINAL')
if not os.path.exists(simPath + 'KNN/Plots_ORIGINAL'):
    os.makedirs(simPath + 'KNN/Plots_ORIGINAL')

if not os.path.exists(simPath + 'LSTM/Plots/FDA'):
    os.makedirs(simPath + 'LSTM/Plots/FDA')
if not os.path.exists(simPath + 'GRU/Plots/FDA'):
    os.makedirs(simPath + 'GRU/Plots/FDA')
if not os.path.exists(simPath + 'KNN/Plots/FDA'):
    os.makedirs(simPath + 'KNN/Plots/FDA')
if not os.path.exists(simPath + 'LSTM/Plots_ORIGINAL/FDA'):
    os.makedirs(simPath + 'LSTM/Plots_ORIGINAL/FDA')
if not os.path.exists(simPath + 'GRU/Plots_ORIGINAL/FDA'):
    os.makedirs(simPath + 'GRU/Plots_ORIGINAL/FDA')
if not os.path.exists(simPath + 'KNN/Plots_ORIGINAL/FDA'):
    os.makedirs(simPath + 'KNN/Plots_ORIGINAL/FDA')


# Chama as rotinas de testes  --------------------------------------------------------------------
dist_y_LSTM.clear()
cdf_LSTM, dist_y = teste('LSTM', base, False, analise_graf)
for dist in dist_y:
    dist_y_LSTM.append(dist)
cdf_LSTM_M = np.mean(dist_y_LSTM)
cdf_LSTM_std = np.std(dist_y_LSTM)

dist_y_GRU.clear()
cdf_GRU, dist_y = teste('GRU', base, False, analise_graf)
for dist in dist_y:
    dist_y_GRU.append(dist)
cdf_GRU_M = np.mean(dist_y_GRU)
cdf_GRU_std = np.std(dist_y_GRU)

dist_y_KNN.clear()
cdf_KNN, dist_y = teste('KNN', base, False, analise_graf)
for dist in dist_y:
    dist_y_KNN.append(dist)
cdf_KNN_M = np.mean(dist_y_KNN)
cdf_KNN_std = np.std(dist_y_KNN)

if salvarResultadosFDA:
    np.save(simPath + 'LSTM/Plots/FDA/FDA - LSTM ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', cdf_LSTM)
    np.save(simPath + 'GRU/Plots/FDA/FDA - GRU ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', cdf_GRU)
    np.save(simPath + 'KNN/Plots/FDA/FDA - KNN ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', cdf_KNN)

if salvarDistancias:
    np.save(simPath + 'LSTM/y_dist - LSTM ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', dist_y_LSTM)
    np.save(simPath + 'GRU/y_dist - GRU ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', dist_y_GRU)
    np.save(simPath + 'KNN/y_dist - KNN ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', dist_y_KNN)
    
if anovaTest:
    boxplotData = [dist_y_LSTM, dist_y_GRU, dist_y_KNN]
    plt.figure(dpi=600)
    plt.ylim([0,12])
    plt.ylabel("Erro de predição (m)")
    plt.boxplot(boxplotData, showfliers=False)
    plt.xticks([1, 2, 3], ['LSTM', 'GRU', 'KNN'])
    plt.savefig(simPath + 'Boxplots_' + simdBm + 'dBm_AT' + simAT + '_tipo_' + simTipo + '.png', bbox_inches='tight')
    plt.show()
    
    F, p = f_oneway(dist_y_LSTM, dist_y_GRU, dist_y_KNN)
    print('Resultado ANOVA:')
    print('F = ' + str(F))
    print('p = ' + str(p))


print('LSTM Mean = {} | STD = {}'. format(cdf_LSTM_M,cdf_LSTM_std))
print('GRU Mean = {} | STD = {}'. format(cdf_GRU_M,cdf_GRU_std))
print('KNN Mean = {} | STD = {}'. format(cdf_KNN_M,cdf_KNN_std))

# Chama a rotina de testes com treinamento base em 1dBm, 10% de atenuação e todos os APs disponíveis --------------------------------------------------------------------
if simPrepDados == 'Stream' and (simTipo != '63' or simAT != '10' or simdBm != '1'):
    dist_y_LSTM_O.clear()
    cdf_LSTM_ORIGINAL, dist_y = teste('LSTM', base, True, analise_graf)
    for dist in dist_y:
        dist_y_LSTM_O.append(dist)
    cdf_LSTM_O_M = np.mean(dist_y_LSTM_O)
    cdf_LSTM_O_std = np.std(dist_y_LSTM_O)
    
    dist_y_GRU_O.clear()
    cdf_GRU_ORIGINAL, dist_y = teste('GRU', base, True, analise_graf)
    for dist in dist_y:
        dist_y_GRU_O.append(dist)
    cdf_GRU_O_M = np.mean(dist_y_GRU_O)
    cdf_GRU_O_std = np.std(dist_y_GRU_O)
    
    dist_y_KNN_O.clear()
    cdf_KNN_ORIGINAL, dist_y = teste('KNN', base, True, analise_graf)
    for dist in dist_y:
        dist_y_KNN_O.append(dist)
    cdf_KNN_O_M = np.mean(dist_y_KNN_O)
    cdf_KNN_O_std = np.std(dist_y_KNN_O)
    
    if salvarResultadosFDA:
        np.save(simPath + 'LSTM/Plots_ORIGINAL/FDA/FDA - LSTM ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', cdf_LSTM_ORIGINAL)
        np.save(simPath + 'GRU/Plots_ORIGINAL/FDA/FDA - GRU ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', cdf_GRU_ORIGINAL)
        np.save(simPath + 'KNN/Plots_ORIGINAL/FDA/FDA - KNN ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', cdf_KNN_ORIGINAL)

    if salvarDistancias:
        np.save(simPath + 'LSTM/y_dist - LSTM_O ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', dist_y_LSTM_O)
        np.save(simPath + 'GRU/y_dist - GRU_O ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', dist_y_GRU_O)
        np.save(simPath + 'KNN/y_dist - KNN_O ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.npy', dist_y_KNN_O)
    
    if anovaTest:
        boxplotData = [dist_y_LSTM_O, dist_y_GRU_O, dist_y_KNN_O]
        plt.figure(dpi=600)
        plt.ylim([0,12])
        plt.ylabel("Erro de predição (m)")
        plt.boxplot(boxplotData, showfliers=False)
        plt.xticks([1, 2, 3], ['LSTM', 'GRU', 'KNN'])
        plt.savefig(simPath + 'Boxplots_O_' + simdBm + 'dBm_AT' + simAT + '_tipo_' + simTipo + '.png', bbox_inches='tight')
        plt.show()
        
        F_O, p_O = f_oneway(dist_y_LSTM_O, dist_y_GRU_O, dist_y_KNN_O)
        print('Resultado ANOVA:')
        print('F = ' + str(F))
        print('p = ' + str(p))
        
    
    print('LSTM_O Mean = {} | STD = {}'. format(cdf_LSTM_O_M,cdf_LSTM_O_std))
    print('GRU_O Mean = {} | STD = {}'. format(cdf_GRU_O_M,cdf_GRU_O_std))
    print('KNN_O Mean = {} | STD = {}'. format(cdf_KNN_O_M,cdf_KNN_O_std))

    ORIGIN = True
else:    
    ORIGIN = False

# Salva os dados de média e desvio padrão no arquivo de texto
if salvarDpResultados:
    with open('resultados.txt', "a") as file_object:
        file_object.write('\n')
        file_object.write('Teste '+ simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + ' - ' + simPrepDados + '\n')
        file_object.write('LSTM Mean = {} | STD = {}\n'. format(cdf_LSTM_M,cdf_LSTM_std))
        file_object.write('GRU Mean = {} | STD = {}\n'. format(cdf_GRU_M,cdf_GRU_std))
        file_object.write('KNN Mean = {} | STD = {}\n'. format(cdf_KNN_M,cdf_KNN_std))
        file_object.write('ANOVA TEST: F = {} | p = {}\n'. format(F,p))
        file_object.write('\n')
        if ORIGIN:
            file_object.write('LSTM_O Mean = {} | STD = {}\n'. format(cdf_LSTM_O_M,cdf_LSTM_O_std))
            file_object.write('GRU_O Mean = {} | STD = {}\n'. format(cdf_GRU_O_M,cdf_GRU_O_std))
            file_object.write('KNN_O Mean = {} | STD = {}\n'. format(cdf_KNN_O_M,cdf_KNN_O_std))
            file_object.write('ANOVA TEST_O: F = {} | p = {}\n'. format(F_O,p_O))
            file_object.write('\n')
        file_object.close()


# Gera o gráfico geral de FDA --------------------------------------------------------------------
if gerarGraficos:
    # fig, ax = plt.subplots()
    plt.figure(dpi=600)
    plt.xlim([0,500])
    plt.ylim([0,1])
    plt.xlabel("Distância (cm)")
    plt.ylabel("FDA")
    plt.title('FDA GERAL ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + ' - ' + simPrepDados)
    plt.plot(cdf_LSTM)
    plt.plot(cdf_GRU)
    plt.plot(cdf_KNN)
    # # legend_extra = plt.legend(["LSTM", "GRU", "KNN"], [cdf_LSTM_M + r'$ \pm $' + cdf_LSTM_std, '', ''], loc=4)
    # legend_extra = plt.legend(["LSTM", "GRU", "KNN"], ['', '', ''], loc='lower right')
    plt.legend(['LSTM', 'GRU', 'KNN'], loc='upper right')
    # ax.add_artist(legend_extra)
    plt.savefig(simPath + 'FDA - GERAL ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.png', bbox_inches='tight')
    plt.show()  
    
    if ORIGIN:
        # Gera o gráfico ORIGNAL de FDA --------------------------------------------------------------------
        plt.figure(dpi=600)
        plt.xlim([0,500])
        plt.ylim([0,1])
        plt.xlabel("Distância (cm)")
        plt.ylabel("FDA")
        plt.title('FDA GERAL ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + ' - ' + simPrepDados)
        plt.plot(cdf_LSTM_ORIGINAL)
        plt.plot(cdf_GRU_ORIGINAL)
        plt.plot(cdf_KNN_ORIGINAL)
        plt.legend(['LSTM', 'GRU', 'KNN'], loc='upper right')
        plt.savefig(simPath + 'FDA - ORIGINAL ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.png', bbox_inches='tight')
        plt.show()  
        
        
        # Gera o gráfico geral + ORIGINAL de FDA --------------------------------------------------------------------
        plt.figure(dpi=600)
        plt.xlim([0,500])
        plt.ylim([0,1])
        plt.xlabel("Distância (cm)")
        plt.ylabel("FDA")
        plt.title('FDA GERAL ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + ' - ' + simPrepDados)
        plt.plot(cdf_LSTM)
        plt.plot(cdf_GRU)
        plt.plot(cdf_KNN)
        plt.plot(cdf_LSTM_ORIGINAL)
        plt.plot(cdf_GRU_ORIGINAL)
        plt.plot(cdf_KNN_ORIGINAL)
        plt.legend(['LSTM', 'GRU', 'KNN', 'LSTM_O', 'GRU_O', 'KNN_O'], loc='upper right')
        plt.savefig(simPath + 'FDA - GERAL + ORIGINAL ' + simdBm + 'dBm AT' + simAT + '% tipo ' + simTipo + '.png', bbox_inches='tight')
        plt.show()
    