import os
import sys
import json
import torch
import argparse
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder
from torch import nn
from model import ConvNet, ResNet101, EarlyStopping
from config import LEARNING_RATE, N_FOLDS, BATCH_SIZE
from config import CAMINHO_SAIDA_METADADOS, NUM_CLASSES, EPOCHS
from loader import carregar_dados_treino, carregar_dados_teste
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


device = "cuda" if torch.cuda.is_available() else "cpu"


def calc_metricas( y_real: NDArray, y_prob: NDArray, epoca: int) -> dict[str, float]:
    """ Calcula métricas de acurácia, precisão, recall e F1.

    Args:
        total_correct_epoch: Number of correctly classified samples.
        num_samples: total of samples.

    Returns:
        dict[str, float]: Dicionário contendo as métrica e seus
            respectivos valores para esse fold de teste.
    """
    y_pred = y_prob.argmax(1)

    accuracy = accuracy_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred, average='macro', zero_division=0.0)
    recall = recall_score(y_real, y_pred, average='macro', zero_division=0.0)
    f1 = f1_score(y_real, y_pred, average='macro', zero_division=0.0)

    return {
        'epoca': epoca,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def treinar_modelo(
    X_train: NDArray, X_test: NDArray,
    y_train: NDArray, y_test: NDArray,
    fold_teste: int) -> None:
    """ Treina o modelo utilizando dados de treinamento e verifica
    o progresso utilizando os dados de validação.

    Args:
        X_train: Batch com imagens para treinamento.
            size (num_images, Channels, Height, Width)
        X_test: Batch com imagens de teste.
            size (num_images, Channels, Height, Width)
        y_train: rótulos de treinamento.
        y_test: rótulos de test.
        fold_teste

    """
    input_shape = (BATCH_SIZE, *X_train[0].shape)

    # X_train, y_train, X_val, y_val = split_train_val(X_train, y_train)

    X_val, y_val = X_test, y_test

    if args.model == "resnet101":
        dl_model = ResNet101(NUM_CLASSES)

    elif args.model == "convnet":
        dl_model = ConvNet(input_shape, NUM_CLASSES)

    else:
        print("Erro: modelo inválido.")
        sys.exit(1)

    dl_model.to(device)

    progresso_metricas = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(dl_model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping()

    print("Starting training...")
    for epoch in range(EPOCHS):

        print(f"Epoch {epoch}...")

        avg_loss_train, y_prob_train = train_one_epoch(
            X_train, y_train, dl_model, criterion, optimizer)
        avg_loss_val, y_prob_val = val_one_epoch(
            X_val, y_val, dl_model, criterion)

        dict_metricas_train = calc_metricas(y_train, y_prob_train, epoch)
        dict_metricas_val = calc_metricas(y_val, y_prob_val, epoch)

        accuracy_train = dict_metricas_train['accuracy']
        accuracy_val = dict_metricas_val['accuracy']

        print(f"Training loss: {avg_loss_train} / Validation loss: {avg_loss_val}")
        print(f"Training accuracy: {accuracy_train} / Validation accuracy: {accuracy_val}")

        early_stopping(-accuracy_val, dl_model, epoch)
        if early_stopping.early_stop:
            print(f"Early stop na época {epoch}. Melhor época: {early_stopping.best_epoch}")
            break

        progresso_metricas.append(
            {"treino": dict_metricas_train, "validacao": dict_metricas_val,}
        )

    early_stopping.load_best_model(dl_model)

    avg_loss_val, y_prob_val = val_one_epoch(
        X_val, y_val, dl_model, criterion)
    melhores_metricas_val = calc_metricas(
        y_val, y_prob_val, early_stopping.best_epoch)

    # accuracy_val = dict_metricas_val['accuracy']
    print(melhores_metricas_val)
    salvar_modelo(dl_model, fold_teste)
    salvar_metricas(melhores_metricas_val, progresso_metricas, fold_teste)

    # _, avg_accuracy_test = val_one_epoch(
    #     X_test, y_test, dl_model, criterion)
    # print(f"Test accuracy: {avg_accuracy_test}")

def split_train_val(X_train: NDArray, y_train: NDArray
                    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    shuffled_idx = np.random.permutation(len(X_train))

    X_train = X_train[shuffled_idx]
    y_train = y_train[shuffled_idx]

    n_samples_train = int(0.78 * len(X_train))

    X_val, y_val = X_train[n_samples_train:], y_train[n_samples_train:]
    X_train_new, y_train_new = X_train[0:n_samples_train], y_train[0:n_samples_train]
    return X_train_new, y_train_new, X_val, y_val

def train_one_epoch(X_train, y_train, dl_model, criterion, optimizer):
    dl_model.train()

    # total_correct_epoch = 0
    total_loss_epoch = 0
    num_samples_train = X_train.shape[0]

    y_prob_classes = []

    for start_batch in range(0, num_samples_train, BATCH_SIZE):

        end_batch = start_batch + BATCH_SIZE

        X_batch = X_train[start_batch:end_batch]
        y_batch = y_train[start_batch:end_batch]

        X_batch = torch.asarray(X_batch).to(device)
        y_batch = torch.asarray(y_batch).to(device)

        y_prob_class = dl_model(X_batch)
        # y_pred = y_prob_class.argmax(1)

        loss = criterion(y_prob_class, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch = loss.item()

        total_loss_epoch += loss_batch

        y_prob_classes.append(y_prob_class.detach().cpu())
        # total_correct_epoch += (y_pred == y_batch).sum().item()

        # print("Loss batch:", loss_batch)
    avg_loss_epoch = total_loss_epoch / num_samples_train
    y_prob_classes = np.vstack(y_prob_classes)

    # Calcular os valores das métricas e coloca tudo em um dicionário
    # avg_accuracy_epoch = total_correct_epoch / num_samples_train

    return avg_loss_epoch, y_prob_classes


def val_one_epoch(X_val, y_val, dl_model, criterion):

    dl_model.eval()

    total_correct_epoch = 0
    total_loss_epoch = 0
    num_samples_train = X_val.shape[0]
    y_prob_classes = []

    with torch.no_grad():
        for start_batch in range(0, num_samples_train, BATCH_SIZE):

            end_batch = start_batch + BATCH_SIZE

            X_batch = X_val[start_batch:end_batch]
            y_batch = y_val[start_batch:end_batch]

            X_batch = torch.asarray(X_batch).to(device)
            y_batch = torch.asarray(y_batch).to(device)

            y_prob_class = dl_model(X_batch)
            y_pred = y_prob_class.argmax(1)

            loss = criterion(y_prob_class, y_batch)

            loss_batch = loss.item()

            total_loss_epoch += loss_batch
            total_correct_epoch += (y_pred == y_batch).sum().item()

            y_prob_classes.append(y_prob_class.detach().cpu())

            # print("Loss batch:", loss_batch)
        avg_loss_epoch = total_loss_epoch / num_samples_train
        # avg_accuracy_epoch = total_correct_epoch / num_samples_train

    y_prob_classes = np.vstack(y_prob_classes)
    return avg_loss_epoch, y_prob_classes


def salvar_modelo( modelo_treinado: nn.Module, fold: int) -> None:
    """ Salva os parâmetros do modelo e os resultados de
    classificação para o fold.
    """
    pasta_resultados = f"resultados_{args.espectrograma}"

    os.makedirs(pasta_resultados, exist_ok=True)

    nome_base = f"fold_{fold}_validacao"

    caminho_arquivo_modelo = f"{pasta_resultados}/{nome_base}.pth"

    torch.save(modelo_treinado.state_dict(), caminho_arquivo_modelo)
    print(f"{caminho_arquivo_modelo} salvo com sucesso.")


def salvar_metricas(
        melhores_metricas: dict[str, float],
        progresso_metricas: list[dict],
        fold: int):
    pasta_resultados = f"resultados_{args.espectrograma}"

    os.makedirs(pasta_resultados, exist_ok=True)

    nome_base = f"fold_{fold}_validacao"

    caminho_json_final = f"{pasta_resultados}/melhor_{nome_base}.json"

    with open(caminho_json_final, 'w') as f:

        json.dump(melhores_metricas, f, indent=4)

    print(f"{caminho_json_final} salvo com sucesso.")

    ############
    caminho_json_evolucao = f"{pasta_resultados}/evolucao_{nome_base}.json"

    with open(caminho_json_evolucao, 'w') as f:

        json.dump(progresso_metricas, f, indent=4)

    print(f"{caminho_json_evolucao} salvo com sucesso.")


def main():
    metadata = pd.read_csv(CAMINHO_SAIDA_METADADOS)

    classes_possiveis = metadata.loc[:, 'class'].values
    lbl_encoder = LabelEncoder()
    lbl_encoder.fit_transform(classes_possiveis)

    # Folds 1 to 10
    for fold_teste in range(1, N_FOLDS+1):

        # Separa quais são os folds de treino
        folds_treino = list(range(1, N_FOLDS+1))
        folds_treino.remove(fold_teste)

        print("Carregando dados...")
        X_train, y_train = carregar_dados_treino(
            metadata, folds_treino, lbl_encoder, args.espectrograma)

        X_val, y_val = carregar_dados_teste(
            metadata, fold_teste, lbl_encoder, args.espectrograma)

        treinar_modelo(X_train, X_val, y_train, y_val, fold_teste)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, required=True,
                        help = "Modelo de Deep learning usado")
    parser.add_argument("-e", "--espectrograma", type=str, required=True,
                        help = "Espectrograma usado")

    # Read arguments from command line
    args = parser.parse_args()

    main()
