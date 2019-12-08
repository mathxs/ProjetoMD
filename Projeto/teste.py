import librosa
import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
from sklearn.ensemble import RandomForestClassifier


def get_features(audio, fs, classe):
  mfccs = librosa.feature.mfcc(y = audio, sr = fs, n_mfcc = 40)                
  mfcc_filtrado = librosa.decompose.nn_filter(mfccs, aggregate = np.median)
  mel = librosa.feature.melspectrogram(y = audio, sr = fs)
  return [
    np.mean(mel.T, axis = 0),
    np.mean(mfcc_filtrado.T, axis = 0),
    classe
  ]

def normalize_audio(audio):
  return librosa.util.normalize(audio)

def get_features_from_audios(audios):
  features = []
  for (audio_serie, fs), file_name in audios:
    quebra_arquivo = int(fs * 1.9)
    for i, inicio in enumerate(range(0, audio_serie.shape[0], quebra_arquivo)): 
      classe = file_name.split('/')[-1][i]
      audio_normalizado = normalize_audio(audio_serie[inicio:(inicio + quebra_arquivo)])
      features.append(get_features(audio_normalizado, fs, classe))
      if i == 3:
        break
  return features

def get_audios_from_folder(folder):
  files = librosa.util.find_files(folder)
  return [(librosa.load(file, None), file) for file in files]

def main():
  audios_treinamento = get_audios_from_folder('TREINAMENTO')
  features_treinamento = get_features_from_audios(audios_treinamento)
  treinamento = pd.DataFrame(features_treinamento, columns = ['Audio', 'Mel', 'Classe'])

  audios_teste = get_audios_from_folder('VALIDACAO')
  features_teste = get_features_from_audios(audios_teste)
  teste = pd.DataFrame(features_teste, columns = ['Audio', 'Mel', 'Classe'])

  classes_treinamento = np.array(treinamento.T.iloc[-1].tolist()).reshape(-1, 1)
  classes_teste = np.array(teste.T.iloc[-1].tolist()).reshape(-1, 1)
  encoder = pp.LabelEncoder()
  encoder.fit(classes_treinamento)
  classes_t_treinamento = encoder.transform(classes_treinamento)
  classes_t_teste = encoder.transform(classes_teste)

  X = np.hstack(
    (treinamento.Audio.tolist(),
    treinamento.Mel.tolist())
  )
  
  X_teste = np.hstack(
    (teste.Audio.tolist(),
    teste.Mel.tolist())
  )

  rfc = RandomForestClassifier(n_estimators = 500)
  rfc.fit(X, classes_t_treinamento)
  print("Taxa de acerto do RandomForestClassifier da letra: ", np.mean(classes_t_teste == rfc.predict(X_teste)))
  
  
  precisao = rfc.predict(X_teste)
  precisao =  np.stack((precisao, classes_t_teste), axis=-1)
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

  print("Taxa de acerto do RandomForestClassifierdo captchar: ", np.mean(captchar))


main()