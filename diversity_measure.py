import torch
import numpy as np
import pandas as pd
import os
from loader import carregar_modelo
from config import BATCH_SIZE, CAMINHO_METADADOS, N_FOLDS, PASTA_RESULTADOS, TAM_IMAGENS
from augmentation import augment_batch
from loader import carregar_dados_teste
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def disagreement_measure(performance_1, performance_2):
    n11 = np.sum(performance_1 & performance_2) # Both correct
    n00 = np.sum(~performance_1 & ~performance_2) # Both incorrect
    n10 = np.sum(performance_1 & ~performance_2)  # 1 correct, 2 incorrect
    n01 = np.sum(~performance_1 & performance_2)  # 1 incorrect, 2 correct

    # 2. Apply Kuncheva's Disagreement Measure formula
    disagreement_measure = (n01 + n10) / (n11 + n00 + n01 + n10)
    return disagreement_measure


def carregar_chekpoint(fold, espectrograma, model):

    pasta_resultados_espec = os.path.join(
        PASTA_RESULTADOS, f"{espectrograma}_{model}")

    os.makedirs(pasta_resultados_espec, exist_ok=True)

    nome_base = f"fold_{fold}_validacao"

    caminho_arquivo_modelo = f"{pasta_resultados_espec}/{nome_base}.pth"

    state_dict = torch.load(caminho_arquivo_modelo)
    return state_dict


def predict_and_compare(X_val, y_val, dl_model):

    dl_model.eval()

    num_samples_train = X_val.shape[0]
    y_pred = []

    with torch.no_grad():
        for start_batch in range(0, num_samples_train, BATCH_SIZE):

            end_batch = start_batch + BATCH_SIZE

            X_batch = X_val[start_batch:end_batch]

            X_batch = torch.asarray(X_batch).to(device)
            X_batch = augment_batch(X_batch)
            # visualizar_imagem(X_batch)

            y_prob_class = dl_model(X_batch)
            y_pred_batch = y_prob_class.argmax(1)

            y_pred += y_pred_batch.tolist()

        # avg_accuracy_epoch = total_correct_epoch / num_samples_train

    y_pred = np.array(y_pred)
    mask_correct = (y_val == y_pred)
    return mask_correct


def classify_samples_and_get_corrects(
    espectrograma, nome_modelo, dl_model, metadata, lbl_encoder
):
    mask_corrects = []
    # Folds 1 to 10
    print(f"Classifying folds for {espectrograma}")
    for fold_teste in tqdm(range(1, N_FOLDS+1)):
        state_dict = carregar_chekpoint(fold_teste, espectrograma, nome_modelo)
        dl_model.load_state_dict(state_dict)

        # Separa quais são os folds de treino
        folds_treino = list(range(1, N_FOLDS+1))
        folds_treino.remove(fold_teste)

        X_val, y_val = carregar_dados_teste(
            metadata, fold_teste, lbl_encoder, espectrograma)

        mask_fold = predict_and_compare(X_val, y_val, dl_model)
        mask_corrects += mask_fold.tolist()

    mask_corrects = np.array(mask_corrects)
    return mask_corrects


def main():
    espectrogramas = list(TAM_IMAGENS)
    espectrogramas.remove('logmel')

    model = 'convnet'

    print(f'====================== Model {model} =======================')
    metadata = pd.read_csv(CAMINHO_METADADOS)

    classes_possiveis = metadata.loc[:, 'class'].values

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit_transform(classes_possiveis)

    X_val, _ = carregar_dados_teste(
        metadata, 1, lbl_encoder, espectrogramas[0])
    input_shape = (BATCH_SIZE, * X_val[0].shape)
    dl_model = carregar_modelo(model, input_shape)
    dl_model.to(device)

    spec1 = 'logmel'
    # Get a boolean array of size N (number of samples validation) indicating
    # which samples were classifiers correctly.
    mask_corrects_1 = classify_samples_and_get_corrects(
        spec1, model, dl_model, metadata, lbl_encoder)

    for spec2 in espectrogramas:
        # Get a boolean array of size N (number of samples validation) indicating
        # which samples were classifiers correctly.
        mask_corrects_2 = classify_samples_and_get_corrects(
            spec2, model, dl_model, metadata, lbl_encoder)

        print(f"Comparison between {spec1} and {spec2}:")
        print(sum(mask_corrects_1))
        print(sum(mask_corrects_2))
        # Calc diversity score
        print('Disagreement metric:',
              disagreement_measure(mask_corrects_1, mask_corrects_2))

if __name__ == "__main__":
    main()

