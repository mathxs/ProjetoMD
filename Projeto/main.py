'''
pickle.dump(featuresdf,open('train', 'wb'))
pickle.dump(featuresdf_teste,open('teste', 'wb'))

featuresdf = pickle.load(open('train', 'rb'))
featuresdf_teste = pickle.load(open('teste', 'rb'))
'''
import librosa
import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris
import numpy as np


#treinando
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing as pp
import pickle

#para compilar instale as bibliotecas pandas, librosa, matplotlib, tensorflow e atualize o colorama, não esquecer das bibliotecas importadas (pip3 install ...)
#python3 main.py

class Tratando_Dados:

    contador = 0
    dados_agrupado = []
    dados_mfccsscaled = []
    dados_mfccsscaled_teste = []

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
    
    def salvarPNG(self,letra_lida,dados_p_seg):
        plt.imshow(dados_p_seg)
        plt.savefig('ENTENDIMENTO/' + letra_lida + '-' + str(self.contador) + '.png')
        plt.close()
        self.contador = self.contador +1

    def separandoArquivos_treino(self, arquivo):
        data , fs = librosa.load (arquivo, None)
        duracao_total = data.shape[0] / fs
        intervalo = duracao_total/4
        if (duracao_total < 7.0):
            #print('erro encontrado no arquivo' + arquivo + ' com duracao de ' + str(duracao_total) + ' letra com '+ str(intervalo))
            return
        quebra_arquivo = int(fs*1.9)
        dados_p_seg = []

        texto = arquivo
        letras = texto.split("/")[1].split(".wav")
        
        for i,ini in enumerate(range(0, data.shape[0], quebra_arquivo)):
            if(i < 4):
                #dados_p_seg.append([pd.Series(data[ini:(ini + quebra_arquivo)]),str(letras[0][i])])
                mfccs = librosa.feature.mfcc(y=data[ini:(ini + quebra_arquivo)], sr=fs,n_mfcc=40)                
                filtrando = librosa.decompose.nn_filter(mfccs, aggregate=np.median)#, metric='cosine', width=int(librosa.time_to_frames(2, sr=fs)))
                zero_rate = librosa.feature.zero_crossing_rate(data[ini:(ini + quebra_arquivo)])
                banda = librosa.feature.spectral_bandwidth(y=data[ini:(ini + quebra_arquivo)],sr=fs)                
                centro = librosa.feature.spectral_centroid(y=data[ini:(ini + quebra_arquivo)],sr=fs) 
                chroma_cq = librosa.feature.chroma_stft(y=data[ini:(ini + quebra_arquivo)],sr=fs) 
                mel = librosa.feature.melspectrogram(y=data[ini:(ini + quebra_arquivo)],sr=fs)
                contraste = librosa.feature.spectral_contrast(y=data[ini:(ini + quebra_arquivo)],sr=fs)
                tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(data[ini:(ini + quebra_arquivo)]),sr=fs)
                
                mfccsscaled = np.mean(filtrando.T,axis=0)                                
                zero_rate = np.mean(zero_rate.T,axis=0)
                banda = np.mean(banda.T,axis=0)
                centro = np.mean(centro.T,axis=0)                
                chroma_cq = np.mean(chroma_cq.T,axis=0)   
                mel = np.mean(mel.T,axis=0)   
                contraste = np.mean(contraste.T,axis=0)   
                tonnetz = np.mean(tonnetz.T,axis=0)                   
                
                self.contador += 1
                print("teste" + str(self.contador))                                            

                self.dados_mfccsscaled.append([mfccsscaled, zero_rate, banda, centro, chroma_cq, mel, contraste, tonnetz, str(letras[0][i])])


                #self.salvarPNGeWAV(i,letras[0][i],dados_p_seg,fs)
                #self.salvarPNG(str(letras[0][i]), mfccs)

        self.dados_agrupado.append(dados_p_seg)

    def separandoArquivos_teste(self, arquivo):
        data , fs = librosa.load (arquivo, None)
        duracao_total = data.shape[0] / fs
        intervalo = duracao_total/4
        if (duracao_total < 7.0):
            #print('erro encontrado no arquivo' + arquivo + ' com duracao de ' + str(duracao_total) + ' letra com '+ str(intervalo))
            return
        quebra_arquivo = int(fs*1.9)
        dados_p_seg = []

        texto = arquivo
        letras = texto.split("/")[1].split(".wav")

        for i,ini in enumerate(range(0, data.shape[0], quebra_arquivo)):
            if(i < 4):
                #dados_p_seg.append([pd.Series(data[ini:(ini + quebra_arquivo)]),str(letras[0][i])])
                mfccs = librosa.feature.mfcc(y=data[ini:(ini + quebra_arquivo)], sr=fs,n_mfcc=40)                
                filtrando = librosa.decompose.nn_filter(mfccs, aggregate=np.median)#, metric='cosine', width=int(librosa.time_to_frames(2, sr=fs)))
                zero_rate = librosa.feature.zero_crossing_rate(data[ini:(ini + quebra_arquivo)])
                banda = librosa.feature.spectral_bandwidth(y=data[ini:(ini + quebra_arquivo)],sr=fs)                
                centro = librosa.feature.spectral_centroid(y=data[ini:(ini + quebra_arquivo)],sr=fs) 
                chroma_cq = librosa.feature.chroma_stft(y=data[ini:(ini + quebra_arquivo)],sr=fs) 
                mel = librosa.feature.melspectrogram(y=data[ini:(ini + quebra_arquivo)],sr=fs)
                contraste = librosa.feature.spectral_contrast(y=data[ini:(ini + quebra_arquivo)],sr=fs)
                tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(data[ini:(ini + quebra_arquivo)]),sr=fs)
                
                mfccsscaled = np.mean(filtrando.T,axis=0)                                
                zero_rate = np.mean(zero_rate.T,axis=0)
                banda = np.mean(banda.T,axis=0)
                centro = np.mean(centro.T,axis=0)                
                chroma_cq = np.mean(chroma_cq.T,axis=0)   
                mel = np.mean(mel.T,axis=0)   
                contraste = np.mean(contraste.T,axis=0)   
                tonnetz = np.mean(tonnetz.T,axis=0)                   
                
                self.contador += 1
                print("teste" + str(self.contador))                                            

                self.dados_mfccsscaled_teste.append([mfccsscaled, zero_rate, banda, centro, chroma_cq, mel, contraste, tonnetz, str(letras[0][i])])

                #self.salvarPNGeWAV(i,letras[0][i],dados_p_seg,fs)
                #self.salvarPNG(str(letras[0][i]), mfccs)

Dados = Tratando_Dados()
files = os.listdir(r'TREINAMENTO/')
print("processando dados de treino")
for f in files:
    #print(r'TREINAMENTO/' + f)
    Dados.separandoArquivos_treino(r'TREINAMENTO/' + f)

print("processando dados de validacao")
files = os.listdir(r'VALIDACAO/')
for f in files:
    Dados.separandoArquivos_teste(r'VALIDACAO/' + f)

#separando os arquivos
featuresdf = pd.DataFrame(Dados.dados_mfccsscaled, columns=['Audio','Zero', 'Banda', 'Centro', 'Chroma', 'Mel', 'Contraste', 'Torre', 'Letra'])
featuresdf_teste = pd.DataFrame(Dados.dados_mfccsscaled_teste, columns=['Audio','Zero', 'Banda', 'Centro', 'Chroma', 'Mel', 'Contraste', 'Torre', 'Letra'])


y = np.array(featuresdf.Letra.tolist()).reshape(-1, 1)
y_teste = np.array(featuresdf_teste.Letra.tolist()).reshape(-1, 1)
#y = featuresdf.Letra.tolist()

enc = pp.LabelEncoder()
enc.fit(y)
y_t = enc.transform(y)
y_t_teste = enc.transform(y_teste)

#X = featuresdf.Audio.tolist()
X = np.hstack(
    (featuresdf.Audio.tolist(), 
    featuresdf.Zero.tolist(),
    #featuresdf.Banda.tolist(),
    featuresdf.Centro.tolist(),
    featuresdf.Chroma.tolist(),
    featuresdf.Mel.tolist(),
    featuresdf.Contraste.tolist(),
    featuresdf.Torre.tolist()
    ))
X_teste = np.hstack(
    (featuresdf_teste.Audio.tolist(), 
    featuresdf_teste.Zero.tolist(),
    #featuresdf_teste.Banda.tolist(),
    featuresdf_teste.Centro.tolist(),
    featuresdf_teste.Chroma.tolist(),
    featuresdf_teste.Mel.tolist(),
    featuresdf_teste.Contraste.tolist(),
    featuresdf_teste.Torre.tolist()
    ))

'''
de = DecisionTreeClassifier()
des = ExtraTreeClassifier()
ne = KNeighborsClassifier(n_neighbors=9)
ne1 = KNeighborsClassifier(n_neighbors=1)
ne2 = KNeighborsClassifier(n_neighbors=20)
rfr = RandomForestRegressor(n_estimators=10, random_state=0)
clf = GaussianNB()
'''
lr = LogisticRegression()
#dummy = DummyClassifier()
#sv = svm.SVC(kernel = 'rbf')
rfc = RandomForestClassifier()
rfc1 = RandomForestClassifier(n_estimators=10000)
rfc2 = RandomForestClassifier(n_estimators=5000)
#rfc3 = RandomForestClassifier(n_estimators=20000)

print('Calculando as IA')

#ne1.fit(X,y_t)
#ne.fit(X,y_t)
#ne2.fit(X,y_t)

#print("Taxa de acerto do KNeighborsClassifier1: ", np.mean(y_t_teste == ne1.predict(X_teste)))
lr.fit(X,y_t)
rfc.fit(X,y_t)
rfc1.fit(X,y_t)
rfc2.fit(X,y_t)
#rfc3.fit(X,y_t)

#print("Taxa de acerto do RandomForestClassifier: ", np.mean(y_t_teste == rfc.predict(X_teste)))


'''
de.fit(X,y_t)
des.fit(X,y_t)
ne1.fit(X,y_t)
ne2.fit(X,y_t)
rfr.fit(X,y_t)
clf.fit(X,y_t)
lr.fit(X,y_t)
dummy.fit(X,y_t)
print("Taxa de acerto do KNeighborsClassifier1: ", np.mean(y_t_teste == ne1.predict(X_teste)))
print("Taxa de acerto do KNeighborsClassifier9: ", np.mean(y_t_teste == ne.predict(X_teste)))
print("Taxa de acerto do KNeighborsClassifier50: ", np.mean(y_t_teste == ne2.predict(X_teste)))
'''
print("Taxa de acerto do LogisticRegression: ", np.mean(y_t_teste == lr.predict(X_teste)))
print("Taxa de acerto do RandomForestClassifier: ", np.mean(y_t_teste == rfc.predict(X_teste)))
print("Taxa de acerto do RandomForestClassifier5000: ", np.mean(y_t_teste == rfc2.predict(X_teste)))
print("Taxa de acerto do RandomForestClassifier10000: ", np.mean(y_t_teste == rfc1.predict(X_teste)))
#print("Taxa de acerto do RandomForestClassifier30000: ", np.mean(y_t_teste == rfc3.predict(X_teste)))

'''
print("Taxa de acerto do DecisionTreeClassifier: ", np.mean(y_t_teste == de.predict(X_teste)))
print("Taxa de acerto do ExtraTreeClassifier: ", np.mean(y_t_teste == des.predict(X_teste)))
print("Taxa de acerto do KNeighborsClassifier1: ", np.mean(y_t_teste == ne.predict(X_teste)))
print("Taxa de acerto do KNeighborsClassifier5: ", np.mean(y_t_teste == ne1.predict(X_teste)))
print("Taxa de acerto do KNeighborsClassifier7: ", np.mean(y_t_teste == ne2.predict(X_teste)))
print("Taxa de acerto do RandomForestRegressor: ", np.mean(y_t_teste == rfr.predict(X_teste)))
print("Taxa de acerto do GaussianNB: ", np.mean(y_t_teste == clf.predict(X_teste)))
print("Taxa de acerto do LogisticRegression: ", np.mean(y_t_teste == lr.predict(X_teste)))
print("Taxa de acerto do DummyClassifier: ", np.mean(y_t_teste == dummy.predict(X_teste)))
''' 
