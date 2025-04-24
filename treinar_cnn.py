import os
import re
import json
import tensorflow as tf
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#import skimage.io as io
import librosa.display
from config import CAMINHO_SAIDA_METADADOS, DIR_BASE_ESPECTROGRAMAS, TAM_IMAGENS, TIPO_ESPECTROGRAMA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Conv2D, MaxPooling2D, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Activation, TimeDistributed
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

#----------------------------------------
# TODO gerar o resto dos espectrogramas

N_EPOCHS = 200
BATCH_SIZE = 32
N_FOLDS = 10

#----------------------------------------
def treinar_modelo(X_train, y_train, X_val, y_val, num_classes):
    if len(tf.config.list_physical_devices(device_type='GPU')) > 0:
        device = "/GPU:0"
    else:
        device = "/CPU:0"

    input_shape = X_train[0].shape

    with tf.device(device):
        model = Sequential()

        model.add(Conv2D(96, kernel_size=(11,11), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal'))

        model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                               padding= 'valid', data_format= None))

        model.add(Conv2D(256, kernel_size=(5,5), strides= 1,
                         padding= 'same', activation= 'relu',
                         kernel_initializer= 'he_normal'))

        model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                               padding= 'valid', data_format= None))

        model.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                         padding= 'same', activation= 'relu',
                         kernel_initializer= 'he_normal'))

        model.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                         padding= 'same', activation= 'relu',
                         kernel_initializer= 'he_normal'))

        model.add(Conv2D(256, kernel_size=(3,3), strides= 1,
                         padding= 'same', activation= 'relu',
                         kernel_initializer= 'he_normal'))

        model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None))

        model.add(Flatten())
        model.add(Dense(4096, activation= 'relu'))
        model.add(Dense(4096, activation= 'relu'))
        model.add(Dense(1000, activation= 'relu'))
        model.add(Dense(num_classes, activation= 'softmax'))

        # Compile the model using categorical crossentropy for multi-class classification
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(learning_rate=1e-4),
                      metrics=['accuracy'])

        # Print the model summary
        model.summary()
        model.fit(X_train, y_train, epochs=N_EPOCHS, validation_data=(X_val, y_val),
                  batch_size=BATCH_SIZE, shuffle=True)
    return model


#----------------------------------------
def validar_modelo(modelo_treinado, X_val, y_val) -> dict:

    y_score = modelo_treinado.predict(X_val)
    y_pred = np.copy(y_score)

    idx_max = np.argmax(y_score, axis=1)
    y_pred[:] = 0
    y_pred[range(len(y_pred)), idx_max] = 1

    roc_auc = roc_auc_score( y_val, y_score, multi_class="ovr", average="micro")

    resultados_classificacao = classification_report(y_val, y_pred, output_dict=True)
    accuracy_value = accuracy_score(y_val, y_pred)

    dict_resultados  = {
        "accuracy": accuracy_value,
        "roc_auc": roc_auc,
        "y_pred": np.argmax(y_pred, axis=1).tolist(),
        "y_val": np.argmax(y_val, axis=1).tolist(),
        "classification_report": resultados_classificacao
        #"f1-score": resultados_classificacao["weighted avg"]["f1-score"],
        #"precision": resultados_classificacao["weighted avg"]["precision"],
        #"recall": resultados_classificacao["weighted avg"]["recall"],
    }

    print("Accuracy:", accuracy_value)
    print("ROC AUC:", roc_auc)
    return dict_resultados


#----------------------------------------
def salvar_modelo_e_resultados(modelo_treinado, dict_resultados, fold) -> None:
    """Salva os parâmetros do modelo e os resultados de
    classificação para o fold."""
    pasta_resultados = f"resultados_{TIPO_ESPECTROGRAMA}"

    os.makedirs(pasta_resultados, exist_ok=True)

    nome_base = f"fold_{fold}_validacao"

    caminho_arquivo_modelo = f"{pasta_resultados}/{nome_base}.h5"

    modelo_treinado.save(caminho_arquivo_modelo)
    print(f"{caminho_arquivo_modelo} salvo com sucesso.")

    caminho_arquivo_json = f"{pasta_resultados}/{nome_base}.json"

    with open(caminho_arquivo_json, 'w') as f:

        json.dump(dict_resultados, f, indent=4)

    print(f"{caminho_arquivo_json} savo com sucesso.")


#----------------------------------------
def carregar_espectrogramas(caminhos_arquivos, rotulos_metadados):
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
        pat = re.compile(r'%s_[0-9]+.png' % nome_arquivo_sem_formato)
        subarquivos = [arquivo for arquivo in arquivos_fold if pat.match(arquivo)]

        for subarquivo in subarquivos:
            espect = np.asarray(Image.open(f"{caminho_pasta}/{subarquivo}"))

            espect = espect.reshape((espect.shape[0] , espect.shape[1], 1))

            espectrogramas.append(espect)
            labels.append(lbl)

    return np.array(espectrogramas), np.array(labels)

#---------------------------------
metadata = pd.read_csv(CAMINHO_SAIDA_METADADOS)

# Folds 1 to 10
for fold_val in range(1, N_FOLDS+1):

    # Separa quais são os folds de treino
    folds_train = list(range(1, N_FOLDS+1))
    folds_train.remove(fold_val)

    # df_train = metadata.loc[metadata['fold'].isin(folds_train), ['spectrogram_name', 'class']]

    df_val = metadata.loc[metadata['fold'] == fold_val, ['spectrogram_name', 'class']]
    df_val = df_val.iloc[np.random.permutation(len(df_val))]

    caminhos_audios_val = [f"{DIR_BASE_ESPECTROGRAMAS}/fold{fold_val}/{arquivo}"
                             for arquivo in df_val["spectrogram_name"]]
    caminhos_audios_fold = []
    X_train, y_train = [], []

    for fold_train in folds_train:
        df_fold = metadata.loc[metadata['fold'] == fold_train, ['spectrogram_name', 'class']]
        df_fold = df_fold.iloc[np.random.permutation(len(df_fold))]

        caminhos_audios_fold = [f"{DIR_BASE_ESPECTROGRAMAS}/fold{fold_train}/{arquivo}"
                                for arquivo in df_fold["spectrogram_name"]]

        X_fold, y_fold= carregar_espectrogramas(caminhos_audios_fold, df_fold["class"].values)

        X_train.append(X_fold)
        y_train.append(y_fold)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    X_val, y_val = carregar_espectrogramas(caminhos_audios_val, df_val["class"].values)

    # Normalizar os dados
    X_train = X_train / 255
    X_val = X_val / 255

    y = np.hstack((y_train, y_val))
    n_classes = np.unique(y).shape[0]

    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(y.reshape(-1, 1))

    y_train = one_hot_encoder.transform(np.array(y_train).reshape(-1, 1)).todense()
    y_train = np.asarray(y_train.astype(np.int32))

    y_val = one_hot_encoder.transform(np.array(y_val).reshape(-1, 1)).todense()
    y_val = np.asarray(y_val.astype(np.int32))

    modelo_treinado = treinar_modelo(X_train, y_train, X_val, y_val,  n_classes)

    dict_resultados = validar_modelo(modelo_treinado, X_val, y_val)
    salvar_modelo_e_resultados(modelo_treinado, dict_resultados, fold_val)
