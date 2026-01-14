#ref: https://www.kaggle.com/code/msripooja/steps-to-convert-audio-clip-to-spectrogram
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Activation, TimeDistributed

#to play audio
import IPython.display as ipd

# Nesse caso você precisa ter no seu diretório uma pasta com o mesmo nome da variável base_dir
base_dir = "audioset"
# E dentro da pasta audioset, você precisa de 3 pastas: carros, motocicletas e motosserras contendo os áudios
audio_carro_path = base_dir + "/carros/"
audio_moto_path = base_dir + "/motocicletas/"  # caminho onde estão os arquivos de audio
audio_serra_path = base_dir + "/motosserras/"

spec_base_path = "./audioset/espectrogramas_com_log"
spec_carro_path = os.path.join(spec_base_path, "carros")
spec_moto_path = os.path.join(spec_base_path, "motocicletas")
spec_serra_path = os.path.join(spec_base_path, "motosserras")
 
os.makedirs(spec_base_path, exist_ok=True)
os.makedirs(spec_carro_path, exist_ok=True)
os.makedirs(spec_moto_path, exist_ok=True)
os.makedirs(spec_serra_path, exist_ok=True)

#--------------------------------
audio_carro_clips = os.listdir(audio_carro_path)[:250]
print(f"Número de arquivos .wav na pasta {audio_carro_path}= ",len(audio_carro_clips))

audio_moto_clips = os.listdir(audio_moto_path)[:250]
print(f"Número de arquivos .wav na pasta {audio_moto_path}= ",len(audio_moto_clips))

audio_serra_clips = os.listdir(audio_serra_path)[:500]
print(f"Número de arquivos .wav na pasta {audio_serra_path}= ",len(audio_serra_clips))

#-------------------------------
#aqui está selecionando o primeiro arquivo da lista, pode ser feito um for para pegar todos de uma vez
#verificar esse sample rate
x_carro, sr_carro = librosa.load(audio_carro_path+audio_carro_clips[0], sr=44100) 
x_moto, sr_moto = librosa.load(audio_moto_path+audio_moto_clips[4], sr=44100) 
x_serra, sr_serra = librosa.load(audio_serra_path+audio_serra_clips[6], sr=44100) 

print(type(x_carro), type(sr_carro))
print(x_carro.shape, sr_carro)

print(type(x_moto), type(sr_moto))
print(x_moto.shape, sr_moto)

print(type(x_serra), type(sr_serra))
print(x_serra.shape, sr_serra)
# 
# #-------------------------------
# 
# #converter o audio para espectrograma:
# # é nesse amplitude_to_dB que ele está convertendo para pressão sonora. Podemos explorar outras ponderações aqui. Se não tiver implementado, podemos implementar.
def salvar_espectrograma(x, sr, spec_path, log=False):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure(figsize=(1, 1))

    if log:
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    else: librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.margins(0,0)
    
    plt.axis("off")
    plt.savefig(spec_path, dpi=300, bbox_inches = 'tight',
                pad_inches = 0, transparent=False, facecolor='white')
    print(f"{spec_path} salvo com sucesso.")

#| # Salvar imagens de espectrograma
#| - Pule esse passo se as imagens já estiverem na pasta

#-------------------------------
# Salvar espectrogramas de carros
for audio_name in audio_carro_clips:
    x_carro, sr_carro = librosa.load(audio_carro_path + audio_name, sr=44100) 
    audio_name = audio_name.replace(".mp3", ".png")
    spec_path = os.path.join(spec_carro_path, audio_name)
    try:
        salvar_espectrograma(x_carro, sr_carro, spec_path, log=True)
    except:
        print("Erro ao salvar arquivo %s" % spec_path)

# Salvar espectrogramas de motos
for audio_name in audio_moto_clips:
    x_moto, sr_moto = librosa.load(audio_moto_path + audio_name, sr=44100) 
    audio_name = audio_name.replace(".mp3", ".png")
    spec_path = os.path.join(spec_moto_path, audio_name)
    try:
        salvar_espectrograma(x_moto, sr_moto, spec_path, log=True)
    except:
        print("Erro ao salvar arquivo %s" % spec_path)

# Salvar espectrogramas de motosserras
for audio_name in audio_serra_clips:
    x_serra, sr_serra = librosa.load(audio_serra_path + audio_name, sr=44100) 
    audio_name = audio_name.replace(".mp3", ".png")
    spec_path = os.path.join(spec_serra_path, audio_name)
    try:
        salvar_espectrograma(x_serra, sr_serra, spec_path, log=True)
    except:
        print("Erro ao salvar arquivo %s" % spec_path)
 
#-------------------------------
#| # Treinamento e teste de rede neural

# Carregar espectrogramas com uma CNN e classificar entre "motosserras e não motosserras"

espectrogramas_carro = os.listdir(spec_carro_path)
espectrogramas_moto = os.listdir(spec_moto_path)
espectrogramas_serra = os.listdir(spec_serra_path)

lista_imagens = []
labels = []

def adicionar_rotulos_imagens(espectrogramas, espectro_path, label):
    for spec in espectrogramas:
        image_path = os.path.join(espectro_path, spec)     
        img = io.imread(image_path)
        lista_imagens.append(img[:, :, :-1])
        # 0 corresponde aos carros e motos
        labels.append(label)

adicionar_rotulos_imagens(espectrogramas_carro, spec_carro_path, 0)
adicionar_rotulos_imagens(espectrogramas_moto, spec_moto_path, 0)
adicionar_rotulos_imagens(espectrogramas_serra, spec_serra_path, 1)

X_train, X_test, y_train, y_test = train_test_split(
    np.array(lista_imagens), np.array(labels), test_size=0.2, random_state=42
)

#-------------------------------
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=X_train[0].shape))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='RMSProp', metrics='accuracy')
model.fit(X_train, y_train, batch_size=32 , epochs=30)

y_pred = model.predict(X_test)[:, 0]
y_pred = np.round(y_pred)
acc = len(np.where(y_pred == y_test)[0]) / len(y_test)
print("Acurácia:", acc)
