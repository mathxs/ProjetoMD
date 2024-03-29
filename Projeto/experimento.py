import librosa
import pandas as pd
import matplotlib.pyplot as plt

#para compilar instale as bibliotecas pandas, librosa, matplotlib e atualize o colorama. (pip3 install ...)
#python3 main.py

def salvarPNGeWAV(dados):
    dados_p_seg[dados].plot().get_figure().savefig('EXPERIMENTO/' + str(dados) + '.png')
    plt.close()
    librosa.output.write_wav('EXPERIMENTO/' + str(dados) + '.wav', dados_p_seg[dados].to_numpy(),fs)

data , fs = librosa.load (r'TREINAMENTO/6a66.wav', None)
duracao_total = data.shape[0] / fs
intervalo = 2.130430839
quebra_arquivo = int(fs*intervalo)
dados_p_seg = {}

for i,ini in enumerate(range(0, data.shape[0], quebra_arquivo)):
    dados_p_seg[i] = pd.Series(data[ini:(ini + quebra_arquivo)])
    salvarPNGeWAV(i)


