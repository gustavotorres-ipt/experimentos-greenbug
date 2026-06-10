import os
import librosa
import numpy as np
import cv2
import random
import pycochleagram.cochleagram as cgram
import pycochleagram.erbfilter as erb
from PIL import Image


def calc_cochleagram(y, sr):

    hi_lim = sr//2
    low_lim = 1
    n_filters = int(np.floor(
        erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)
    ) - 1)  # Calcula o número de filtros

    human_co = cgram.human_cochleagram(
        y,
        sr=sr,
        n = n_filters,
        low_lim = low_lim,
        hi_lim = hi_lim,
        sample_factor = 1,
        nonlinearity='db'
    )
    spec = cv2.resize(human_co, dsize=(128, 128),
                      interpolation=cv2.INTER_CUBIC)

    spec_norm = 255 * (spec.max() - spec) / (spec.max() - spec.min())

    spec_pil = Image.fromarray(spec_norm.astype(np.uint8))
    spec_pil.show()


def main():
    dir_audios = './data_motosserras/chainsaw/'
    audios = os.listdir(dir_audios)
    audios = [os.path.join(dir_audios, audio) for audio in audios]

    selected_file = random.choice(audios)
    print(selected_file)

    y, sr = librosa.load(selected_file)
    calc_cochleagram(y, sr)

# librosa.display.specshow(res, sr=sr, x_axis="time")
# breakpoint()
if __name__ == "__main__":
    main()
