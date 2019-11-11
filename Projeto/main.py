import librosa
import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris
import numpy as np

#para compilar instale as bibliotecas pandas, librosa, matplotlib, tensorflow e atualize o colorama, não esquecer das bibliotecas importadas (pip3 install ...)
#python3 main.py

class Tratando_Dados:

    contador = 0
    dados_agrupado = []
    dados_mfccsscaled = []

    def salvarConsolidado(self,letra):
        for i in range(0,len(self.dados_agrupado)):
            if(self.dados_agrupado[i][0][1] == letra):
                #print(Dados.dados_agrupado[i][0][1])
                pd.Series(self.dados_agrupado[i][0][0]).plot()

        plt.savefig('ENTENDIMENTO/' + letra + '.png')
        plt.close()

    def salvarPNGeWAV(self,dados,letra_lida,dados_p_seg,fs):
        dados_p_seg[dados].plot().get_figure().savefig('EXPERIMENTO/' + letra_lida + '-' + str(self.contador) + '.png')
        plt.close()
        librosa.output.write_wav('EXPERIMENTO/' + letra_lida + '-' + str(self.contador) + '.wav', dados_p_seg[dados].to_numpy(),fs)
        self.contador = self.contador +1

    def separandoArquivos(self, arquivo):
        data , fs = librosa.load (arquivo, None)
        duracao_total = data.shape[0] / fs
        intervalo = duracao_total/4
        if (duracao_total < 7.0):
            #print('erro encontrado no arquivo' + arquivo + ' com duracao de ' + str(duracao_total) + ' letra com '+ str(intervalo))
            return
        quebra_arquivo = int(fs*intervalo)
        dados_p_seg = []

        texto = arquivo
        letras = texto.split("/")[1].split(".wav")

        for i,ini in enumerate(range(0, data.shape[0], quebra_arquivo)):
            if(i < 4):
                dados_p_seg.append([pd.Series(data[ini:(ini + quebra_arquivo)]),str(letras[0][i])])
                mfccs = librosa.feature.mfcc(y=data[ini:(ini + quebra_arquivo)], sr=fs, n_mfcc=40)
                mfccsscaled = np.mean(mfccs.T,axis=0)
                self.dados_mfccsscaled.append([mfccsscaled, str(letras[0][i])])
                #self.salvarPNGeWAV(i,letras[0][i],dados_p_seg,fs)

        self.dados_agrupado.append(dados_p_seg)


Dados = Tratando_Dados()
files = os.listdir(r'TREINAMENTO/')
for f in files:
    #print(r'TREINAMENTO/' + f)
    Dados.separandoArquivos(r'TREINAMENTO/' + f)


'''
Dados.salvarConsolidado('6')
Dados.salvarConsolidado('7')
Dados.salvarConsolidado('a')
Dados.salvarConsolidado('b')
Dados.salvarConsolidado('c')
Dados.salvarConsolidado('d')
Dados.salvarConsolidado('h')
Dados.salvarConsolidado('m')
Dados.salvarConsolidado('n')
Dados.salvarConsolidado('x')
'''

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

#from sklearn.datasets import load_iris
'''
X = []
Y = []
for i in range(0,len(Dados.dados_agrupado)):
    x_temp, y_temp = [pd.Series(Dados.dados_agrupado[i][0][0])], pd.Series(Dados.dados_agrupado[i][0][1])
    X.append(x_temp)
    Y.append(y_temp)

X = np.asarray(X)
Y = np.asarray(Y)

print('tamanho de X ' + str(len(X)))
print('tamanho de Y ' + str(len(Y)))

print(X)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X.reshape(1,10), Y.reshape(1,10), test_size=0.4,random_state=0)

clf = LinearDiscriminantAnalysis()
y_pred = clf.fit(X_train,y_train).predict(X_test)
print(sum(y_pred == y_test)/len(y_pred))
'''

featuresdf = pd.DataFrame(Dados.dados_mfccsscaled, columns=['Audio','Letra'])
#print(featuresdf)

X = np.array(featuresdf.Audio.tolist())
y = np.array(featuresdf.Letra.tolist())

for i in range(0, len(y)):
    if y[i] == '6':
        y[i] = 1
    elif y[i] == '7':
        y[i] = 2
    elif y[i] == 'a':
        y[i] = 3
    elif y[i] == 'b':
        y[i] = 4
    elif y[i] == 'c':
        y[i] = 5
    elif y[i] == 'd':
        y[i] = 6
    elif y[i] == 'h':
        y[i] = 7
    elif y[i] == 'm':
        y[i] = 8
    elif y[i] == 'n':
        y[i] = 9
    elif y[i] == 'x':
        y[i] = 10

# Encode the classification labels
#le = LabelEncoder()
#yy = to_categorical(le.fit_transform(y)) 

# split the dataset 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(x_train)  
X_test = sc.transform(x_test)  

X_train = x_train
X_test = x_test

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)  

from sklearn.metrics import r2_score, mean_squared_error
# Valor de R2 perto de 1 nos diz que é um bom modelo
#print(f"R2 score: {r2_score(y_test, y_pred)}")
# MSE Score perto de 0 é um bom modelo
#print(f"MSE score: {mean_squared_error(y_test, y_pred)}")
#acuracia prof
#print('Acuracia: ' + str(sum(y_pred == y_test)/len(y_pred)))
#entendendo
j = 0
for i in range(0, len(y_pred)):
    print('Predito: ' + str(int(round(y_pred[i],0))) + ' Verdadeiro: ' + str(int(y_test[i])))
    if(int(round(y_pred[i],0)) == int(y_test[i])):
        j = j + 1

print('Taxa de acerto: ' + str((j/len(y_pred))))
