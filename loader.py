import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from model import ConvNet
from numpy.typing import NDArray
from PIL import Image
from config import N_FOLDS, DIR_BASE_ESPECTROGRAMAS
from tqdm import tqdm


def carregar_imagens(caminhos_arquivos: list[str], rotulos_metadados: NDArray
                     ) -> tuple[NDArray, NDArray]:
    # Carrega os espectrogramas e os rótulos correspondetes
    espectrogramas = []
    labels = []

    caminho_pasta = "/".join(caminhos_arquivos[0].split("/")[:-1])
    arquivos_fold = os.listdir(caminho_pasta)

    for caminho_arquivo, lbl in zip(caminhos_arquivos, rotulos_metadados):
        nome_arquivo_sem_formato = caminho_arquivo.split("/")[-1][:-4]

        # Filtra todos os subarquivos com o prefixo desejado
        # e coloca eles em uma lista.
        # Isso é para os casos de imagens longas que foram cortadas.
        # pat = re.compile(r'%s_[0-9]+.png' % nome_arquivo_sem_formato)
        # subarquivos = [arquivo for arquivo in arquivos_fold if pat.match(arquivo)]

        # for subarquivo in subarquivos:
        # image_espec = Image.open(f"{caminho_pasta}/{subarquivo}").convert('RGB')
        image_espec = Image.open(caminho_arquivo).convert('RGB')
        #image_espec.show()

        espectrog = np.asarray(image_espec)
        espectrog = np.moveaxis(espectrog, 2, 0)

        # espetrog = espetrog.reshape((1, espetrog.shape[0] , espetrog.shape[1]))

        espectrogramas.append(espectrog)
        labels.append(lbl)

    return np.array(espectrogramas).astype(np.float32), \
           np.array(labels).astype(np.int64)



def carregar_dados_treino(
    df_metadata: pd.DataFrame, folds_treino: list[int], lbl_encoder: LabelEncoder,
) -> tuple[NDArray, NDArray]:
    """ Função que lê todas as imagens de treino, seus respectivos rótulos e
    devolve ambos no formato NumPy.

    Args:
        df_metadata  Dataframe contendo nomes de imagens, classes
            e outros metadados.
        folds_treino: lista de folds de treino (de 1 a 10).
        lbl_encoder: transformar palavras em números.

    Returns:
        Tupla contendo os inputs (X_train) e os rótulos (y_train)
            para treinamento.
    """
    caminhos_audios_treino = []
    X_train, y_train = [], []

    for fold_train in folds_treino:

        # Dataframe contendo apenas o nome da imagem e o label do respectivo fold.
        df_fold = df_metadata.loc[
            df_metadata['fold'] == fold_train, ['spectrogram_name', 'class']
        ]
        # Embaralha os dados
        shuffled_order = np.random.permutation(len(df_fold))
        df_fold = df_fold.iloc[shuffled_order]

        caminhos_audios_treino = [
            f"{DIR_BASE_ESPECTROGRAMAS}/fold{fold_train}/{arquivo}"
            for arquivo in df_fold["spectrogram_name"]
        ]

        labels = df_fold["class"].values
        labels = lbl_encoder.transform(labels)

        X_fold, y_fold = carregar_imagens( caminhos_audios_treino, labels )

        X_train.append(X_fold)
        y_train.append(y_fold)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    # Normaliza os dados
    X_train = X_train / 255
    return X_train, y_train

def carregar_dados_teste(
    df_metadata: pd.DataFrame, fold_test: int, lbl_encoder: LabelEncoder
) -> tuple[NDArray, NDArray]:
    """ Função que lê todas as imagens de teste, seus respectivos rótulos e
    devolve ambos no formato NumPy.

    Args:
        df_metadata: Dataframe contendo nomes de imagens, classes
            e outros metadados.
        fold_test: fold de teste.
        lbl_encoder: transformar palavras em números.

    Returns:
        X_test, y_test: Tupla contendo os inputs (X) e os rótulos (y)
            para treinamento.
    """
    df_teste = df_metadata.loc[
        df_metadata['fold'] == fold_test,
        # Filtra apenas as amostras do fold e teste.
        ['spectrogram_name', 'class']
        # Filtra apenas o nome do arquivo e a classe.
    ]

    # Embaralha o dataset
    shuffled_order = np.random.permutation(len(df_teste))
    df_teste = df_teste.iloc[shuffled_order]

    caminhos_audios_teste = [
        f"{DIR_BASE_ESPECTROGRAMAS}/fold{fold_test}/{arquivo}"
        for arquivo in df_teste["spectrogram_name"]]

    labels = df_teste["class"].values
    labels = lbl_encoder.transform(labels)

    X_test, y_test = carregar_imagens( caminhos_audios_teste, labels )

    # Normalizar os dados
    X_test = X_test / 255
    return X_test, y_test
