import os
import librosa
import numpy as np
from config import TAM_IMAGENS
from PIL import Image

#--------------------------------------
def gerar_mel_spec(y, sr):
    return librosa.feature.melspectrogram(y=y, sr=sr, n_fft=4096)

def gerar_log_mel_spec(y, sr):
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=4096)
    S_dB = (librosa.power_to_db(spect, ref=np.max) + 60) / 10
    return S_dB

def gerar_l2_mel_spec(y, sr):
    S_dB = gerar_log_mel_spec(y, sr)
    S_linha_dB =  (librosa.power_to_db(S_dB, ref=np.max) + 10) / 40
    return S_linha_dB

def gerar_l3_mel_spec(y, sr):
    S_linha_dB =  gerar_l2_mel_spec(y, sr)
    S_2linhas_dB =  (librosa.power_to_db(S_linha_dB, ref=np.max) + 60) / 10
    return S_2linhas_dB


FUNCOES_GERACAO_SPEC = {"melspec": gerar_mel_spec,
                        "logmel": gerar_log_mel_spec,
                        "l2m": gerar_l2_mel_spec,
                        "l3m": gerar_l3_mel_spec,
                        }

#--------------------------------------
def cortar_espectrograma(spectrogram, largura_janela, altura_janela):
    altura_spec = spectrogram.shape[0] + 1
    largura_spec = spectrogram.shape[1]
    spectrogram_slices = []

    print("Tamanho espectrograma:", spectrogram.shape )

    for h in range(altura_janela, altura_spec, altura_janela):
        for w in range(largura_janela, largura_spec, largura_janela):
            mini_slice = spectrogram[h-altura_janela : h,   # largura
                                w-largura_janela : w,  # altura
                                np.newaxis]
            spectrogram_slices.append(mini_slice)

    return spectrogram_slices


def salvar_espectrogramas(audio_clips, audio_path, spectrogram_path, espec_tipo):
    gerar_espectrograma = FUNCOES_GERACAO_SPEC[espec_tipo]

    for i, audio_name in enumerate(audio_clips):
        y, sr = librosa.load( os.path.join(audio_path, audio_name) )
        audio_name = audio_name.replace(".mp3", "").replace(".wav", "")

        fullpath = os.path.join(spectrogram_path, audio_name)
        try:
            mel_spec = gerar_espectrograma(y, sr)
            mel_spec = 255 * (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
            mel_spec = np.flip(mel_spec, axis=0) # put low frequencies at the bottom in image

            im = Image.fromarray(mel_spec).convert("L")

            dim = TAM_IMAGENS[espec_tipo]
            resized_img = im.resize((dim, dim))
            full_filename = fullpath + ".png"
            resized_img.save(full_filename)

            # mel_spec_slices = cortar_espectrograma(
            #     mel_spec, TAM_IMAGENS[espec_tipo], TAM_IMAGENS[espec_tipo]
            # )
            # # Salva o arquivo e começa o próximo
            # for j in range(len(mel_spec_slices)):
            #     full_filename = fullpath + f"_{j+1}.png"
            #     spec_save = mel_spec_slices[j][:, :, 0]

            #     im = Image.fromarray(spec_save).convert("L")
            #     im.save(full_filename)

            print("Arquivo %s salvo com sucesso." % full_filename)

        except Exception as e:
            print( "Erro ao salvar %s: %s." % (fullpath, e) )
