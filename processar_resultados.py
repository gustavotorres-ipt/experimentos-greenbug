import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config import N_FOLDS, PASTA_RESULTADOS


CLF_METRICAS = ['Accuracy', 'Precision', 'Recall', 'F1-score']


def gerar_curvas_aprendizado(caminhos_experimentos: list[str]):
    # 'curva_evolucao_fold'

    for caminho in caminhos_experimentos:
        for f in range(1, N_FOLDS+1):
            caminho_resultado_fold = os.path.join(
                caminho, f'evolucao_fold_{f}_validacao.json'
            )

            f = open(caminho_resultado_fold)
            results_fold = json.load(f)

            n_epocas = len(results_fold)

            valores_val = [ results_fold[i]['validacao']['accuracy']
                           for i in range(n_epocas) ]

            valores_treino = [ results_fold[i]['treino']['accuracy']
                              for i in range(n_epocas) ]

            plt.plot(valores_treino, alpha=0.4)
            plt.plot(valores_val, alpha=0.4)
        plt.show()
        breakpoint()

    # plt.xlabel("Epoch")
    # plt.ylabel("Validation Loss")
    # plt.title("Learning Curves Across Folds")


def calcular_metricas_medias(caminho_pasta: str) -> tuple[dict, dict]:
    resultados_acuracia = []
    resultados_precisao = []
    resultados_recall = []
    resultados_f1 = []

    for f in range(1, N_FOLDS+1):
        caminho_melhor = os.path.join(
            caminho_pasta, f'melhor_fold_{f}_validacao.json')

        with open(caminho_melhor) as f:
            results_fold = json.load(f)
            resultados_acuracia.append(results_fold['accuracy'])
            resultados_precisao.append(results_fold['precision'])
            resultados_recall.append(results_fold['recall'])
            resultados_f1.append(results_fold['f1'])

    medias_resultados = {'accuracy': np.mean(resultados_acuracia),
                         'precision': np.mean(resultados_precisao),
                         'recall': np.mean(resultados_recall),
                         'f1': np.mean(resultados_f1), }

    desvios_resultados = {'accuracy': np.std(resultados_acuracia),
                          'precision': np.std(resultados_precisao),
                          'recall': np.std(resultados_recall),
                          'f1': np.std(resultados_f1), }
    return medias_resultados, desvios_resultados


def plotar_melhores_resultados(caminhos_experimentos: list[str]):
    espectrogramas = []
    modelos_classificacao = []
    valores_metricas = []
    metricas = []

    for caminho in caminhos_experimentos:
        medias, desvios = calcular_metricas_medias(caminho)
        pasta = os.path.split(caminho)[-1]
        espectrograma, rede_neural = tuple(pasta.split("_"))

        valores_metricas += [medias['accuracy'], medias['precision'],
                             medias['recall'], medias['f1']]
        metricas += CLF_METRICAS
        espectrogramas += [espectrograma] * 4
        modelos_classificacao += [rede_neural] * 4
    dict_values = {
        'Value': valores_metricas, 'Metric name': metricas,
        'Spectrogram': espectrogramas, 'Model': modelos_classificacao,
    }

    sns.set_style("darkgrid")
    sns.scatterplot(
        data=dict_values,
        x="Metric name", y="Value",
        hue="Spectrogram", style="Model", s=120
    )

    caminho_salvar = os.path.join(
        PASTA_RESULTADOS, f'comparacao_melhores.png')
    plt.savefig(caminho_salvar)
    print(caminho_salvar, "salvo.")
    plt.clf()


def main():
    conteudo_pasta = [os.path.join(PASTA_RESULTADOS, pasta)
                      for pasta in os.listdir(PASTA_RESULTADOS)]
    pastas_experimentos = [pasta for pasta in conteudo_pasta
                           if os.path.isdir(pasta)]
    gerar_curvas_aprendizado(pastas_experimentos)
    # gerar_matrizes_confusao
    plotar_melhores_resultados(pastas_experimentos)


if __name__ == "__main__":
    main()
