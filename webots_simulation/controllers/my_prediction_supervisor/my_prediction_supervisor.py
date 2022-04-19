# -*- coding: utf-8 -*-

from controller import Supervisor, Node
import sys
import random
import time
import pandas as pd
import numpy as np
import os.path


# do this once only

print(" ")
print(" ")
print(" ------------------------ WiFi - Robot ------------------------ ")
print(" ")
print(" ")
print("                         `.....::.....         ")
print("                      `..``````::``````..`     ")
print("                    `..```.``````````.```..`-/ ")
print("                   ..``   .    ``    .   ``--  ")
print("                  ..``    . ``.``.`` .    ``.. ")
print("                  - `     . ``.``.`` .     ` - ")
print("                 - `      .          .      ` -")
print("                 :`.      .          .      .`:")
print("                 - `      .          .      ` -")
print("                  - `      `` -++- ``      ` - ")
print("                  ..``       ``````       ``.. ")
print("                   .- `                  ` -.  ")
print("                     ..```            ```..    ")
print("                       ...````````````...      ")
print("                          ............         ")
print(" ")
print(" ")



TIME_STEP = 32

supervisor = Supervisor()

# Abre os arquivos de referência para gerar os dados de simulação e KRP
KRP_valid_positions = pd.read_csv('valid_krp.csv', header=None, sep=';')
ap1 = pd.read_csv('ap1.csv', header=None, sep=';')
ap2 = pd.read_csv('ap2.csv', header=None, sep=';')
ap3 = pd.read_csv('ap3.csv', header=None, sep=';')
ap4 = pd.read_csv('ap4.csv', header=None, sep=';')
ap5 = pd.read_csv('ap5.csv', header=None, sep=';')
ap6 = pd.read_csv('ap6.csv', header=None, sep=';')

# Tipos de teste a serem considerados
test_type_list = [63]
#test_type_list = [63, 52, 41, 37, 32, 22, 16, 11, 8, 4, 2, 1, 0]

# Define o dataframe para receber os dados da simulação
colunas_dados = ['test', 'time', 'posX', 'posY', 'KRP_period', 'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6']
data_result = pd.DataFrame(columns=colunas_dados)

# Define o dataframe para receber os log de simulação
colunas_log = ['Time of Dump', 'Simulation Time', 'test_type', 'test']
data_log = pd.DataFrame(columns=colunas_log)

# Inicia as variáveis de teste
test = 1
test_type_count = 0
test_type = test_type_list[test_type_count]

# Busca o robô e seus dados na simulação
robot_node = supervisor.getFromDef("WIFI_ROBOT")
if robot_node is None:
    sys.stderr.write("No DEF WIFI_ROBOT node found in the current world file\n")
    sys.exit(1)
trans_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

# Inicia contadores de tempo
start_time = time.time()
test_time = supervisor.getTime()
read_time = supervisor.getTime()

# Define tempo do KRP e as variáveis de controle para o KRP
KRP_time = random.randint(10, 20)
KRP_ready = False
KRP_period = False


# ------------------------------------------------------------------------------------ Subrotina para verificar se existe dados salvos e onde parou
def file_last_test():
    # Busca variáveis globais para uso na subortina
    global test_type
    
    # Define local do arquivo de testes e testa sua existência
    PATH = './Tests/test ' + str(test_type) + '.csv'
    if os.path.isfile(PATH):
        # Retorna o número do último teste no arquivo
        test_file = pd.read_csv(PATH, sep=';')
        print("Test: {} / Test start number: {}".format(test_type, test_file['test'].iloc[-1]))
        return int(int(test_file['test'].iloc[-1]) + 1)
    else:
        # Retorna teste como 1
        print("Test: {} / Test start number: 1". format(test_type))
        return 1

# ------------------------------------------------------------------------------------ Subrotina para realizar o KRP
def teleport_rand():
    # Define variável de controle e inicia o loop de verificação de localização válida
    test_pos = True
    while test_pos:
        # Define uma posição aleatória
        posX = random.uniform(0.5, 9.49)
        posY = random.uniform(0.5, 8.49)
        # Calcula quadrante de verificação
        test_posX = int((posX // 0.5))
        test_posY = int(17 - (posY // 0.5))
        
        try:
            # Testa se a posição é válida e sair do loop
            if int(KRP_valid_positions.iloc[test_posY, test_posX]) == 2:
                test_pos = False
            # Testa se a posição é marginalmente válida e sair do loop
            elif float(KRP_valid_positions.iloc[test_posY, test_posX]) != 0:
                if ((posX - float(KRP_valid_positions.iloc[test_posY, test_posX])) // 0.5) == test_posX:
                    test_pos = False

        except:
            # Posição inválida por erro na validação, reiniciar do loop
            test_pos = True

    # Teleporta o robô para a nova posição válida e reinicia o sistema de simulação física
    INITIAL = [posY, 0.042, posX]
    trans_field.setSFVec3f(INITIAL)
    robot_node.resetPhysics()


# ------------------------------------------------------------------------------------ Subrotina para adicionar os dados
def add_data():
    # Busca variáveis globais para uso na subortina
    global data_result
    global test_time

    # Obtêm os dados de posição do robô na simulação
    trans_values = trans_field.getSFVec3f()
    rotation_values = rotation_field.getSFRotation()
    
    # Calculando os quadrantes referentes a posição do robô
    ap_posX = int((trans_values[2] // 0.5) + 1)
    ap_posY = int(18 - (trans_values[0] // 0.5))
    
    # Obtém os dados do fingerprinting
    ap_root = pd.Series([ap1.iloc[ap_posY, ap_posX], ap2.iloc[ap_posY, ap_posX], ap3.iloc[ap_posY, ap_posX], ap4.iloc[ap_posY, ap_posX], ap5.iloc[ap_posY, ap_posX], ap6.iloc[ap_posY, ap_posX]])
    # Define máscara de tipo de teste
    ap_test_type = pd.Series([int(x) for x in list('{0:06b}'.format(test_type))])
    # Define vetor de ruído Gaussiano
    ap_noise = pd.Series(np.random.normal(0, 2, 6))
    # Define máscara de atenuação
    ap_mask = pd.Series(random.choices(population=[0, 1], weights=[0.2, 0.8], k=6))

    # Aplica o ruído Gaussiano, a máscara de atenuação e a máscara de teste nos dados base
    ap_signals = ap_root + ap_noise
    ap_signals = ap_signals * ap_mask
    ap_signals = ap_signals * ap_test_type
    
    #print("Time test {}: {} / {}". format((supervisor.getTime() // 1 - test_time // 1), supervisor.getTime(), test_time))
    # Inclui no dataframe os dados simulados
    test_instant = (supervisor.getTime() // 1 - test_time // 1)
    
    if test_instant > 0 and test_instant < 31:
        # Inclui no dataframe os dados simulados
        data_result = data_result.append({'test' : test, 'time' : test_instant, 'posX' : trans_values[2], 'posY' : trans_values[0], 'KRP_period' : KRP_period, 'ap1' : ap_signals.iloc[0], 'ap2' : ap_signals.iloc[1], 'ap3' : ap_signals.iloc[2], 'ap4' : ap_signals.iloc[3], 'ap5' : ap_signals.iloc[4], 'ap6' : ap_signals.iloc[5]}, ignore_index=True)
        
        
# ------------------------------------------------------------------------------------ Subrotina para salvar os dados no arquivo de teste
def dump_tests():
    # Busca variável global de dados para atualização
    global data_result
    global test_time
    
    # Indica no console o progresso do teste
    print(" ---------------------  Dumping data....  --------------------- ")

    # Define local do arquivo de testes e testa sua existência
    PATH = './Tests/test ' + str(test_type) + '.csv'
    if os.path.isfile(PATH):
        # Incorpora os testes no arquivo exixtente
        data_result.to_csv('Tests\\test ' + str(test_type) + '.csv',  sep=';', index=False, mode='a', header=False)
    else:
        # Cria um novo arquivo de testes
        data_result.to_csv('Tests\\test ' + str(test_type) + '.csv',  sep=';', index=False)
    
    # Limpa o dataframe de testes
    data_result = data_result[0:0]
    
    # Reinicia o simulador, teleporta o robô e reinicia o controlador
    supervisor.simulationSetMode(supervisor.SIMULATION_MODE_PAUSE)
    supervisor.simulationReset()
    robot_node.restartController()
    teleport_rand() 
    supervisor.simulationSetMode(supervisor.SIMULATION_MODE_FAST)
    
    # Reinicia tempo de referência do teste
    test_time = 0


# ------------------------------------------------------------------------------------ Subrotina para salvar os dados no arquivo de log
def dump_log():
    # Busca variável global de dados para atualização
    global data_log
    global test_time
    
    # Prepara os dados de log
    data_log = data_log.append({'Time of Dump' : ((time.time() / 60 / 60 / 24) + 25569 - 0.125), 'Simulation Time' : int(time.time() - start_time), 'test_type' : int(test_type), 'test' : int(test)}, ignore_index=True)
    # Define local do arquivo de log e testa sua existência
    PATH = './Tests/sim_log.csv'
    if os.path.isfile(PATH):
        # Incorpora o log no arquivo exixtente
        data_log.to_csv('Tests\\sim_log.csv',  sep=';', index=False, mode='a', header=False)
    else:
        # Cria um novo arquivo de log
        data_log.to_csv('Tests\\sim_log.csv',  sep=';', index=False)
    # Limpa o dataframe de log
    data_log = data_log[0:0]


# Define o último teste salvo no arquivo de testes
print(" ---------------------  Evaluating saved  --------------------- ")
print(" ")
next_test = True
while next_test:
    test = file_last_test()
    # Se o teste salvo estiver completo, segue para o próximo até o final
    if test > 15000:
        test_type_count = test_type_count + 1
        if test_type_count < len(test_type_list):
            test_type = test_type_list[test_type_count]
        else:
            test_type = 0
            next_text = False
    else:
        next_test = False
print(" ")
print(" ------------------------  Test start  ------------------------ ")
print(" ")

teleport_rand()

# ------------------------------------------------------------------------------------ Loop principal de teste
while supervisor.step(TIME_STEP) != -1:
    # this is done repeatedly    
    
    # Testa se a simulação encerrou na última etapa
    if test_type == 0:
        # Encerra a simulação pois chegou no fim
        supervisor.simulationSetMode(supervisor.SIMULATION_MODE_PAUSE)
    
    # Testa se passou 1 segundo de simulação
    if (supervisor.getTime() // 1 != read_time // 1 or supervisor.getTime() == 0):
        # Cataloga os novos dados de simulação
        add_data()
        # Reinia a contagem de tempo de leitura
        read_time = supervisor.getTime()

    # Testa se está no momento de realizar o KRP
    if (supervisor.getTime() - test_time) // 1 == KRP_time:
        # Testa se está pronto para realizar o KRP
        if KRP_ready:
            # Teleporta o robô
            teleport_rand()
            
            # Indica que o KRP já foi realizado neste teste
            KRP_ready = False
            KRP_period = True
    
    # Testa se chegou ao final do teste
    elif (supervisor.getTime() - test_time) // 1 == 30:
        # Testa se está pronto para realizar o KRP
        if KRP_ready:
            # Teleporta o robô e indica que a finalização já foi realizada neste passo do simulador
            teleport_rand()
            KRP_ready = False
            
            # Reinicia tempo de referência do teste
            test_time = supervisor.getTime()
            
            # Atualiza status da simulação no console e no log
            if test % 10 == 0:
                cur_time = int(round(time.time() - start_time, 0))
                cur_est = int(round((30 / 50) * (15000 - test), 0))
                print("Test type {:>2} - test {:>5} ({:03.1%}) -/- TS=>{:02d}:{:02d}:{:02d} -/- Est.=>{:02d}:{:02d}:{:02d}". format(test_type, test, test / 15000, cur_time // (60 * 60), (cur_time // 60) % 60, cur_time % 60, cur_est // (60 * 60), (cur_est // 60) % 60, cur_est % 60))
                dump_log()
            
            # Verifica se já juntou uma quantidade pré-definida para salvar no arquivo de testes
            if test % 200 == 0:
                dump_tests()
            
            # Reinicia as variáveis de número do teste, instante do KRP e KRP ocorrido para um novo teste
            test = test + 1
            KRP_time = random.randint(10, 20)
            KRP_period = False

            # Testa se completou a quantidade de testes a configuração atual
            if test > 15000:
                # Salva os dados no arquivo de testes
                dump_tests()

                # Limpa o dataframe de testes
                data_result = data_result[0:0]
                
                # Reinicia as variáveis de tipo de teste e número do teste para um novo tipo de teste. Se for o último teste prepara para finalizar a simulação
                test_type_count = test_type_count + 1
                test_type = test_type_list[test_type_count]
                test = 1
                
                # Testa se chegou ao final dos testes
                if test_type == 0:
                    print(" ")
                    print(" ")
                    print(" ------------------------  Test  over  ------------------------ ")
                    print(" ")
                    print(" ")

    else:
        # Indica que o teste está pronto para um novo KRP
        KRP_ready = True
        


