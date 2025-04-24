import os

TIPO_ESPECTROGRAMA = "melspec"

TAM_IMAGENS = {"melspec": 128, "logmel": 128, "l2m": 128, "l3m": 128}

TEMPO_AUDIO_MAXIMO = 2.56 #segundos

CAMINHO_SAIDA_METADADOS = os.path.join("data", "informacoes_audios.csv")

DIR_BASE_ESPECTROGRAMAS = f"./data/spectrograms/{TIPO_ESPECTROGRAMA}"
