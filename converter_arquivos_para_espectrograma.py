import os
import sys
import argparse
import pandas as pd
from funcoes_espectrogramas import salvar_espectrogramas
from config import CAMINHO_METADADOS, TEMPO_AUDIO_MAXIMO, DIR_BASE_ESPECTROGRAMAS, \
    N_FOLDS, DIR_DATA, TAM_IMAGENS


COL_ARQUIVOS_SONS = "slice_file_name"
COL_ESPECTROGRAMAS = "spectrogram_name"


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
        for n_fold in range(1, N_FOLDS + 1):
            # Se encontrar o fold correspondente, posso sair do loop
            if audio in folds_audios[f"fold{n_fold}"]:
                folds.append(n_fold)
                break

    return folds

def main():
    #--------------------------------------------------
    # Cria a pasta para salvar os espectrogramas
    dir_espectrogramas = os.path.join(DIR_BASE_ESPECTROGRAMAS, args.espectrograma)
    os.makedirs(dir_espectrogramas, exist_ok=True)

    #-------------------------------------------------
    metadata = pd.read_csv(CAMINHO_METADADOS)
    
    metadata["duration"] = metadata["end"].values.astype(float) - metadata["start"].values.astype(float)

    metadata = metadata.loc[metadata["duration"].values.astype(float) >= TEMPO_AUDIO_MAXIMO]

    #----------------------------------------------------
    # Dicionário contendo o fold correspondente aos áudios
    folds_audios = {}

    for i in range(1, N_FOLDS + 1):
        audio_path = os.path.join(DIR_DATA, f"fold{i}")

        audios_names = os.listdir(audio_path)
        audios_names = filtrar_audios_curtos(
            audios_names, metadata[COL_ARQUIVOS_SONS].values)

        folds_audios[f"fold{i}"] = audios_names

        # Gerar  espectrogramas para o fold, salvar em uma pasta dir_data/spectrogramas/tipo_spec/foldx
        dir_espectrogramas_fold = f"{dir_espectrogramas}/fold{i}"
        os.makedirs(dir_espectrogramas_fold, exist_ok=True)

        salvar_espectrogramas(audios_names, audio_path,
                              dir_espectrogramas_fold, args.espectrograma)

    #----------------------------------------------------
    print("Adicionando os folds...")

    metadata["fold"] = adicionar_folds(
        folds_audios, metadata[COL_ARQUIVOS_SONS].values)

    # Muda a extensão de arquivos de som para png
    metadata["spectrogram_name"] = [
        f"{audio[:-4]}.png" for audio in metadata[COL_ARQUIVOS_SONS].values]

    # Salva os metadados atualizados
    metadata.to_csv(f"{CAMINHO_METADADOS}", index=False)
    print(f"{CAMINHO_METADADOS} salvo com sucesso.")


if __name__ == "__main__":
    possiveis_espectrogramas = list(TAM_IMAGENS.keys())

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e", "--espectrograma", type=str, required=True,
        help = f"Espectrograma usado: Opções {possiveis_espectrogramas}")
    # Read arguments from command line
    args = parser.parse_args()
    if args.espectrograma not in possiveis_espectrogramas:
        print("Erro: tipo de espectrograma inválido.",
              f"Válidos {possiveis_espectrogramas}")
        sys.exit(1)

    main()
