import os

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 5

NUM_CLASSES = 10

N_FOLDS = 10


TIPO_ESPECTROGRAMA = "logmel"

TAM_IMAGENS = {"melspec": 128, "logmel": 128, "l2m": 128, "l3m": 128}

TEMPO_AUDIO_MAXIMO = 2.56 #segundos

CAMINHO_SAIDA_METADADOS = os.path.join("data", "informacoes_audios.csv")

DIR_BASE_ESPECTROGRAMAS = f"./data/spectrograms/{TIPO_ESPECTROGRAMA}"
