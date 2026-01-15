import os

BATCH_SIZE = 32
LEARNING_RATE = 5e-6
EPOCHS = 100

NUM_CLASSES = 10

N_FOLDS = 10

# TIPO_ESPECTROGRAMA = "logmel"

TAM_IMAGENS = {"melspec": 128, "logmel": 128, "l2m": 128, "l3m": 128}

TEMPO_AUDIO_MAXIMO = 2.56 #segundos

CAMINHO_SAIDA_METADADOS = os.path.join("metadata_urban_sounds", "informacoes_audios.csv")

DIR_DATA = 'data_urban_sounds'
DIR_BASE_ESPECTROGRAMAS = os.path.join(DIR_DATA, "spectrograms")