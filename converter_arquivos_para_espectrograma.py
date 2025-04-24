import os
import pandas as pd
from funcoes_espectrogramas import salvar_espectrogramas
from config import CAMINHO_SAIDA_METADADOS, TEMPO_AUDIO_MAXIMO, DIR_BASE_ESPECTROGRAMAS

#-------------------------------------------------
def filtrar_audios_curtos(audios_names, valid_audios):
    """ Elimina os áudios muito curtos de acordo com o nome
        no df metadata.
    """
    long_audios = [audio for audio in audios_names if audio in valid_audios]

    return long_audios

# Adicionar os folds dos áudios que foram salvos como espectrogramas
def adicionar_folds(folds_audios, audios_names):
    folds = []

    for audio in audios_names:
        # Procura o arquivo nos folds até encontrar
        for n_fold in range(1, 11):
            # Se encontrar o fold correspondente, posso sair do loop
            if audio in folds_audios[f"fold{n_fold}"]:
                folds.append(n_fold)
                break

    return folds

#--------------------------------------------------
# Cria a pasta para salvar os espectrogramas
os.makedirs(DIR_BASE_ESPECTROGRAMAS, exist_ok=True)

#-------------------------------------------------
metadata = pd.read_csv('./data/UrbanSound8K.csv')

metadata["duration"] = metadata["end"].values.astype(float) - metadata["start"].values.astype(float)

metadata = metadata.loc[metadata["duration"].values.astype(float) >= TEMPO_AUDIO_MAXIMO]

#----------------------------------------------------
# Dicionário contendo o fold correspondente aos áudios
folds_audios = {}

for i in range(1, 11):
    audios_names = os.listdir(f"./data/fold{i}")
    audios_names = filtrar_audios_curtos(audios_names, metadata["slice_file_name"].values)

    folds_audios[f"fold{i}"] = audios_names

    # Gerar  espectrogramas para o fold, salvar em uma pasta data/spectrogramas/tipo_spec/foldx
    dir_spectrograms = f"{DIR_BASE_ESPECTROGRAMAS}/fold{i}"
    os.makedirs(dir_spectrograms, exist_ok=True)

    audio_path = f"./data/fold{i}"

    salvar_espectrogramas(audios_names, audio_path, dir_spectrograms)

#----------------------------------------------------
print("Adicionando os folds...")

metadata["fold"] = adicionar_folds(folds_audios, metadata["slice_file_name"].values)

# Muda a extensão de arquivos de som para png
metadata["spectrogram_name"] = [f"{audio[:-4]}.png" for audio in metadata["slice_file_name"].values]

# Salva os metadados atualizados
metadata.to_csv(f"{CAMINHO_SAIDA_METADADOS}", index=False)
print(f"{CAMINHO_SAIDA_METADADOS} salvo com sucesso.")
