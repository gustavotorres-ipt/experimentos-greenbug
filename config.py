import os

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 100

N_FOLDS = 10

# TIPO_ESPECTROGRAMA = "logmel"

TAM_IMAGENS = {"melspec": 128, "logmel": 128, "l2m": 128, "l3m": 128}

TEMPO_AUDIO_MAXIMO = 2.56 #segundos

NUM_CLASSES = 3

CAMINHO_METADADOS = os.path.join("metadata_motosserras", "informacoes_audios.csv")
# CAMINHO_ARQUIVO_ENTRADA = os.path.join("metadata_motosserras", "motosserras_carros_motos.csv")

PASTA_RESULTADOS = "resultados_motosserras"

DIR_DATA = 'data_motosserras'
DIR_BASE_ESPECTROGRAMAS = os.path.join(DIR_DATA, "spectrograms")
