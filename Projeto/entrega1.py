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
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing as pp
import pickle
from datetime import datetime
from sklearn import preprocessing


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
                
                #harmonica
                mfcc_delta = librosa.feature.delta(filtrando)
                y_harmonic, y_percussive = librosa.effects.hpss(data[ini:(ini + quebra_arquivo)])
                tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=fs)
                chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=fs)
                beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
                beat_mfcc_delta = librosa.util.sync(np.vstack([filtrando, mfcc_delta]), beat_frames)  
                #D, wp = librosa.sequence.dtw(filtrando ,chromagram)
                              
                mfccsscaled = np.mean(filtrando.T,axis=0)                                
                zero_rate = np.mean(zero_rate.T,axis=0)
                banda = np.mean(banda.T,axis=0)
                centro = np.mean(centro.T,axis=0)                
                chroma_cq = np.mean(chroma_cq.T,axis=0)   
                mel = np.mean(mel.T,axis=0)   
                contraste = np.mean(contraste.T,axis=0)   
                tonnetz = np.mean(tonnetz.T,axis=0)
                beat_mfcc_delta = np.mean(beat_mfcc_delta.T, axis=0)
                beat_chroma = np.mean(beat_chroma.T, axis=0)
                #wp = np.mean(wp.T, axis=0)
                #D = np.mean(D.T, axis=0)

                self.contador += 1
                print("letra de teste processado:" + str(self.contador))                                            

                self.dados_mfccsscaled.append([mfccsscaled, zero_rate, banda, centro, chroma_cq, mel, contraste, tonnetz, beat_mfcc_delta, beat_chroma, str(letras[0][i])])

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
                
                #harmonica
                mfcc_delta = librosa.feature.delta(filtrando)
                y_harmonic, y_percussive = librosa.effects.hpss(data[ini:(ini + quebra_arquivo)])
                tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=fs)
                chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=fs)
                beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
                beat_mfcc_delta = librosa.util.sync(np.vstack([filtrando, mfcc_delta]), beat_frames)                
                #D, wp = librosa.sequence.dtw(filtrando ,chromagram)
                              
                mfccsscaled = np.mean(filtrando.T,axis=0)                                
                zero_rate = np.mean(zero_rate.T,axis=0)
                banda = np.mean(banda.T,axis=0)
                centro = np.mean(centro.T,axis=0)                
                chroma_cq = np.mean(chroma_cq.T,axis=0)   
                mel = np.mean(mel.T,axis=0)   
                contraste = np.mean(contraste.T,axis=0)   
                tonnetz = np.mean(tonnetz.T,axis=0)
                beat_mfcc_delta = np.mean(beat_mfcc_delta.T, axis=0)
                beat_chroma = np.mean(beat_chroma.T, axis=0)
                #wp = np.mean(wp.T, axis=0)
                #D = np.mean(D.T, axis=0)
                
                
                self.contador += 1
                print("letra de teste processado:" + str(self.contador))                                            

                self.dados_mfccsscaled_teste.append([mfccsscaled, zero_rate, banda, centro, chroma_cq, mel, contraste, tonnetz, beat_mfcc_delta, beat_chroma, str(letras[0][i])])

                #self.salvarPNGeWAV(i,letras[0][i],dados_p_seg,fs)
                #self.salvarPNG(str(letras[0][i]), mfccs)

pastaTreinamento = 'TREINAMENTO'#input('Entre com o nomes da pasta onde estão os arquivos de treinamento: (Exatamente igual, e sem a /) ')
pastaTeste = 'VALIDACAO'#input('Entre com o nomes da pasta onde estão os arquivos de teste: (Exatamente igual, e sem a /) ')

t0 = datetime.now()

Dados = Tratando_Dados()
files = os.listdir(str(pastaTreinamento) + '/')
print("processando dados de treino")
for f in files:
    #print(r'TREINAMENTO/' + f)
    Dados.separandoArquivos_treino(str(pastaTreinamento) + '/' + f)

t1 = datetime.now()

print("processando dados de validacao")
files = os.listdir(str(pastaTeste) + '/')
for f in files:
    Dados.separandoArquivos_teste(str(pastaTeste) + '/' + f)

t2 = datetime.now()

#separando os arquivos
featuresdf = pd.DataFrame(Dados.dados_mfccsscaled, columns=['Audio','Zero', 'Banda', 'Centro', 'Chroma', 'Mel', 'Contraste', 'Torre','B_MFCC','B_CHR','D','wp', 'Letra'])
featuresdf_teste = pd.DataFrame(Dados.dados_mfccsscaled_teste, columns=['Audio','Zero', 'Banda', 'Centro', 'Chroma', 'Mel', 'Contraste', 'Torre', 'B_MFCC','B_CHR','D','wp', 'Letra'])

y = np.array(featuresdf.Letra.tolist()).reshape(-1, 1)
y_teste = np.array(featuresdf_teste.Letra.tolist()).reshape(-1, 1)

enc = pp.LabelEncoder()
enc.fit(y)
y_t = enc.transform(y)
y_t_teste = enc.transform(y_teste)

X = np.hstack(
    (featuresdf.Audio.tolist(), 
    featuresdf.Zero.tolist(),
    featuresdf.Banda.tolist(),
    featuresdf.Centro.tolist(),
    featuresdf.Chroma.tolist(),
    featuresdf.Mel.tolist(),
    featuresdf.Contraste.tolist(),
    featuresdf.Torre.tolist(),
    featuresdf.B_MFCC.tolist(),
    featuresdf.B_CHR.tolist(),
    #featuresdf.D.tolist(),
    #featuresdf.wp.tolist()

    ))
X_teste = np.hstack(
    (featuresdf_teste.Audio.tolist(), 
    featuresdf_teste.Zero.tolist(),
    featuresdf_teste.Banda.tolist(),
    featuresdf_teste.Centro.tolist(),
    featuresdf_teste.Chroma.tolist(),
    featuresdf_teste.Mel.tolist(),
    featuresdf_teste.Contraste.tolist(),
    featuresdf_teste.Torre.tolist(),
    featuresdf_teste.B_MFCC.tolist(),
    featuresdf_teste.B_CHR.tolist(),
    #featuresdf_teste.D.tolist(),
    #featuresdf_teste.wp.tolist()
    ))

X =  np.nan_to_num(X)#preprocessing.normalize(np.nan_to_num(X))
X_teste = np.nan_to_num(X_teste)#preprocessing.normalize(np.nan_to_num(X_teste))

t3 = datetime.now()

rfc = RandomForestClassifier(n_estimators=10000)
rfc.fit(X,y_t)

precisao = rfc.predict(X_teste)
precisao =  np.stack((precisao, y_t_teste), axis=-1)
corretude = []
for i in range(0,len(precisao)):
    if (precisao[i][0] == precisao[i][1]):
        corretude.append(1)
    else:
        corretude.append(0)
        
captchar = []
i = 0
while i < len(corretude):
    if (corretude[i]+corretude[i+1]+corretude[i+2]+corretude[i+3]) == 4:
        captchar.append(1)
    else:
        captchar.append(0)
    i += 4

#print("Taxa de acerto do RandomForestClassifier: ", np.mean(y_t_teste == rfc.predict(X_teste)))
print("Taxa de acerto do RandomForestClassifier: ", np.mean(captchar))

t4 = datetime.now()

print( "Tempo do processamento treinando: " + str(t1 - t0) + " ms" )
print( "Tempo do processamento testando: " + str(t2 - t1) + " ms" )
print( "Tempo do ajuste do treino: " + str(t3 - t2) + " ms" )
print( "Treinando e predizendo: " + str(t4 - t3) + " ms" )

