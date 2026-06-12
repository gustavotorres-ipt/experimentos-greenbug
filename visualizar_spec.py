import os
import librosa
import argparse
import numpy as np
import cv2
import random
import pycochleagram.cochleagram as cgram
import pycochleagram.erbfilter as erb
import matplotlib.pyplot as plt
from PIL import Image
from funcoes_espectrogramas import FUNCOES_GERACAO_SPEC
from config import TAM_IMAGENS


def display_spec(spec, sr):
    _, ax = plt.subplots(figsize=(5, 5))
    librosa.display.specshow(spec, sr=sr, ax=ax, x_axis='time')

    plt.tight_layout()
    plt.show()
    plt.clf()


# def calc_cochleagram(y, sr):
# 
#     hi_lim = sr//2
#     low_lim = 1
#     n_filters = int(np.floor(
#         erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)
#     ) - 1)  # Calcula o número de filtros
# 
#     human_co = cgram.human_cochleagram(
#         y,
#         sr=sr,
#         n = n_filters,
#         low_lim = low_lim,
#         hi_lim = hi_lim,
#         sample_factor = 1,
#         nonlinearity='db'
#     )
# 
#     return 
    # spec_norm = 255 * (spec.max() - spec) / (spec.max() - spec.min())
    # spec_pil = Image.fromarray(spec_norm.astype(np.uint8))
    # spec_pil.show()


def main(args):
    dir_audios = './data_motosserras/chainsaw/'
    audios = os.listdir(dir_audios)
    audios = [os.path.join(dir_audios, audio) for audio in audios]

    random.seed(42)
    selected_file = random.choice(audios)
    print(selected_file)

    y, sr = librosa.load(selected_file)

    func_spec = FUNCOES_GERACAO_SPEC[args.espectrograma]
    spec = func_spec(y, sr)
    spec = cv2.resize(spec, dsize=(128, 128),
                      interpolation=cv2.INTER_CUBIC)
    display_spec(spec, sr)

# librosa.display.specshow(res, sr=sr, x_axis="time")
# breakpoint()
if __name__ == "__main__":

    possiveis_espectrogramas = list(TAM_IMAGENS.keys())

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e", "--espectrograma", type=str, required=True,
        help = f"Espectrograma usado: Opções {possiveis_espectrogramas}")
    args = parser.parse_args()
    main(args)
