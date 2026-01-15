!git clone https://github.com/mcdermottLab/pycochleagram.git

import os

os.chdir('./pycochleagram')
!python setup.py install


os.chdir('/content/drive/MyDrive/3_Ciclo/GreenBug/')

import sys
sys.path.append('/content/pycochleagram/pycochleagram')
import cochleagram as cgram


# Nesse caso você precisa ter no seu diretório uma pasta com o mesmo nome da variável base_dir
base_dir = "clean_audioset"

tipo_espectrograma = "LM_MFCC"
# E dentro da pasta audioset, você precisa de 3 pastas: carros, motocicletas e motosserras contendo os áudios
audio_carro_path = base_dir + "/carros/"
audio_moto_path = base_dir + "/motocicletas/"  # caminho onde estão os arquivos de audio
audio_serra_path = base_dir + "/motosserras/"

spec_base_path = base_dir + f"/{tipo_espectrograma}"
spec_carro_path = os.path.join(spec_base_path, "carros")
spec_moto_path = os.path.join(spec_base_path, "motocicletas")
spec_serra_path = os.path.join(spec_base_path, "motosserras")

os.makedirs(spec_base_path, exist_ok=True)
os.makedirs(spec_carro_path, exist_ok=True)
os.makedirs(spec_moto_path, exist_ok=True)
os.makedirs(spec_serra_path, exist_ok=True)

FILE_FORMAT = "png"
ALTURA_IMAGEM = 40
LARGURA_IMAGEM = 128


audio_carro_clips = sorted(os.listdir(audio_carro_path))[:50]
print(f"Número de arquivos .wav na pasta {audio_carro_path}= ",len(audio_carro_clips))

audio_moto_clips = sorted(os.listdir(audio_moto_path))[:50]
print(f"Número de arquivos .wav na pasta {audio_moto_path}= ",len(audio_moto_clips))

audio_serra_clips = sorted(os.listdir(audio_serra_path))[:50]
print(f"Número de arquivos .wav na pasta {audio_serra_path}= ",len(audio_serra_clips))


#aqui está selecionando o primeiro arquivo da lista, pode ser feito um for para pegar todos de uma vez
#verificar esse sample rate
x_carro, sr_carro = librosa.load(audio_carro_path+audio_carro_clips[0])
x_moto, sr_moto = librosa.load(audio_moto_path+audio_moto_clips[4])
x_serra, sr_serra = librosa.load(audio_serra_path+audio_serra_clips[6])

print(type(x_carro), type(sr_carro))
print(x_carro.shape, sr_carro)

print(type(x_moto), type(sr_moto))
print(x_moto.shape, sr_moto)

print(type(x_serra), type(sr_serra))
print(x_serra.shape, sr_serra)


import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load(os.path.join(audio_serra_path, audio_serra_clips[20]))

import erbfilter as erb

class cochleagram:
    IMAGE_HEIGHT = 256 # Estava tentando modificar por esse parâmetro. Mas depois usei o opencv
    IMAGE_WIDTH = 576 # Estava tentando modificar por esse parâmetro. Mas depois usei o opencv
    SR=16000
    hi_lim = SR//2 
    low_lim = 10    
    n_filters = int(np.floor(erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)) - 1)  # Calcula o número de filtros
    nonlinearity = 'db'
    ret_mode='envs'
    sample_factor = 1  # Ajuste do fator de amostragem
	
	
human_co = cgram.human_cochleagram(y,sr =cochleagram.SR,
                             n = cochleagram.n_filters,
                             low_lim = cochleagram.low_lim, hi_lim =cochleagram.hi_lim,
                             sample_factor = cochleagram.sample_factor,
                             nonlinearity=cochleagram.nonlinearity)
							 
import cv2
res = cv2.resize(human_co, dsize=(287, 40), interpolation=cv2.INTER_CUBIC)
librosa.display.specshow(res, sr=sr, x_axis="time")
print(res.shape)