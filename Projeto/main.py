import librosa
import os
import pandas as pd
import matplotlib.pyplot as plt

#para compilar instale as bibliotecas pandas, librosa, matplotlib e atualize o colorama. (pip3 install ...)
#python3 main.py

class Tratando_Dados:

    contador = 0
    dados_agrupado = []

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
        dados_p_seg = {}

        texto = arquivo
        letras = texto.split("/")[1].split(".wav")

        for i,ini in enumerate(range(0, data.shape[0], quebra_arquivo)):
            if(i < 4):
                dados_p_seg[i] = pd.Series(data[ini:(ini + quebra_arquivo)])
                self.salvarPNGeWAV(i,letras[0][i],dados_p_seg,fs)


        self.dados_agrupado.append(dados_p_seg)


Dados = Tratando_Dados()
files = os.listdir(r'TREINAMENTO/')
for f in files:
    print(r'TREINAMENTO/' + f)
    Dados.separandoArquivos(r'TREINAMENTO/' + f)

#print(Dados.dados_agrupado)