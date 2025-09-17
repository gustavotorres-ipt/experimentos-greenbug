import os
import re
import torch
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder
from torch import nn
from model import ConvNet, ResNet101, EarlyStopping
from config import LEARNING_RATE, N_FOLDS, BATCH_SIZE, CAMINHO_SAIDA_METADADOS, NUM_CLASSES, EPOCHS
from loader import carregar_dados_treino, carregar_dados_teste


device = "cuda" if torch.cuda.is_available() else "cpu"


def treinar_modelo(
    X_train: NDArray, X_test: NDArray, y_train: NDArray, y_test: NDArray
) -> None:
    """ Treina o modelo utilizando dados de treinamento e verifica
    o progresso utilizando os dados de validação.

    Args:
        X_train: Batch com imagens para treinamento.
            size (num_images, Channels, Height, Width)
        X_test: Batch com imagens de teste.
            size (num_images, Channels, Height, Width)
        y_train: rótulos de treinamento.
        y_test: rótulos de test.

    """
    input_shape = (BATCH_SIZE, *X_train[0].shape)

    # X_train, y_train, X_val, y_val = split_train_val(X_train, y_train)

    X_val, y_val = X_test, y_test

    dl_model = ConvNet(input_shape, NUM_CLASSES)
    # dl_model = ResNet101(NUM_CLASSES)
    dl_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dl_model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping()

    print("Starting training...")
    for epoch in range(EPOCHS):

        print(f"Epoch {epoch}...")
        avg_loss_train, avg_accuracy_train = train_one_epoch(
            X_train, y_train, dl_model, criterion, optimizer)
        avg_loss_val, avg_accuracy_val = val_one_epoch(
            X_val, y_val, dl_model, criterion)

        print(f"Training loss: {avg_loss_train} / Validation loss: {avg_loss_val}")
        print(f"Training accuracy: {avg_accuracy_train} / Validation accuracy: {avg_accuracy_val}")

        early_stopping(avg_loss_val, dl_model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    _, avg_accuracy_test = val_one_epoch(
        X_test, y_test, dl_model, criterion)
    print(f"Test accuracy: {avg_accuracy_test}")

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

    total_correct_epoch = 0
    total_loss_epoch = 0
    num_samples_train = X_train.shape[0]

    for end_batch in range(BATCH_SIZE, num_samples_train, BATCH_SIZE):

        start_batch = end_batch - BATCH_SIZE

        X_batch = X_train[start_batch:end_batch]
        y_batch = y_train[start_batch:end_batch]

        X_batch = torch.asarray(X_batch).to(device)
        y_batch = torch.asarray(y_batch).to(device)

        y_prob_class = dl_model(X_batch)
        y_pred = y_prob_class.argmax(1)

        loss = criterion(y_prob_class, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch = loss.item()

        total_loss_epoch += loss_batch
        total_correct_epoch += (y_pred == y_batch).sum().item()

        # print("Loss batch:", loss_batch)
    avg_loss_epoch = total_loss_epoch / num_samples_train
    avg_accuracy_epoch = total_correct_epoch / num_samples_train

    return avg_loss_epoch, avg_accuracy_epoch


def val_one_epoch(X_val, y_val, dl_model, criterion):

    dl_model.eval()

    total_correct_epoch = 0
    total_loss_epoch = 0
    num_samples_train = X_val.shape[0]

    with torch.no_grad():
        for end_batch in range(BATCH_SIZE, num_samples_train, BATCH_SIZE):

            start_batch = end_batch - BATCH_SIZE

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

            # print("Loss batch:", loss_batch)
        avg_loss_epoch = total_loss_epoch / num_samples_train
        avg_accuracy_epoch = total_correct_epoch / num_samples_train
    return avg_loss_epoch, avg_accuracy_epoch

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
        X_train, y_train = carregar_dados_treino(metadata, folds_treino, lbl_encoder)

        X_val, y_val = carregar_dados_teste(metadata, fold_teste, lbl_encoder)

        treinar_modelo(X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    main()
