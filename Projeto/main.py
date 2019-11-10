import librosa
import pandas as pd

data , fs = librosa.load (r'TREINAMENTO/6a66.wav', None)
duracao_total = data.shape[0] / fs
intervalo = 1
dados_p_seg = {}

for i,ini in enumerate(range(0, data.shape[0], fs*intervalo)):
    dados_p_seg[i] = pd.Series(data[ini:(ini + fs*intervalo)])

dados_p_seg[0].plot()
dados_p_seg[1].plot()

#para compilar instale as bibliotecas pandas, librosa, matplotlib e atualize o colorama. (pip3 install ...)
#python3 main.py