# /home/gustavo/OneDrive_gustavotorres@ipt.br/Ciclo 3/Greenbug/Codigos/audioset/motocicletas
# wav, mpga
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io.wavfile import read
from pydub.silence import split_on_silence
from pydub import AudioSegment, effects

caminho_audios = "./audioset"
caminho_audios_limpos = "./clean_audioset"
caminho_waveforms = "./waveforms_audioset"
tipos_audios = ["carros", "motocicletas", "motosserras"]

N_MAX_ARQUIVOS = 100
SAMPLE_RATE = 22050


for tipo_audio in tipos_audios:
    os.makedirs(os.path.join(caminho_audios_limpos, tipo_audio), exist_ok=True)
for tipo_audio in tipos_audios:
    os.makedirs(os.path.join(caminho_waveforms, tipo_audio), exist_ok=True)


def limpar_audio(audio, rate):
    # anything under silence_thresh dBFS is considered silence
    audio_segs = split_on_silence(audio, min_silence_len=500, silence_thresh=-50)

    output = audio_segs[0]
    for seg in audio_segs[1:]:
        output += seg

    return output

def salvar_wave_audio(audio_bruto, audio_limpo, arquivo_saida, mostrar=False):
    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(audio_bruto);
    ax[0].set_title('Áudio Original');
    ax[0].set_xlabel('Tempo');
    ax[0].set_ylabel('Amplitude');

    ax[1].plot(audio_limpo);
    ax[1].set_title('Áudio Limpo');
    ax[1].set_xlabel('Tempo');
    ax[1].set_ylabel('Amplitude');

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(arquivo_saida)
    print(f"{arquivo_saida} salvo com sucesso.")
    if mostrar:
        plt.show()
    plt.clf()

def salvar_audio_limpo(audio_limpo, arquivo_saida):
    audio_limpo.export(arquivo_saida, format="mp3")
    print(f"{arquivo_saida} salvo com sucesso.")


for tipo in tipos_audios:
    print(f"======= Processando áudios de {tipo} =======")
    arquivos_audio = os.listdir(os.path.join(caminho_audios, tipo))[:N_MAX_ARQUIVOS]

    for arquivo in tqdm(arquivos_audio):
        caminho_arquivo = os.path.join(caminho_audios, tipo, arquivo)
        try:
            audio = AudioSegment.from_file(caminho_arquivo)

            audio_limpo = limpar_audio(audio, SAMPLE_RATE)

            audio_limpo_np = np.array(audio_limpo.get_array_of_samples())
            audio_bruto_np = np.array(audio.get_array_of_samples())

            arquivo_saida = os.path.join(caminho_audios_limpos, tipo, f"{arquivo[:-4]}.mp3")
            salvar_audio_limpo(audio_limpo, arquivo_saida)

            arquivo_saida = os.path.join(caminho_waveforms, tipo, f"{arquivo[:-4]}.png")
            salvar_wave_audio(audio_bruto_np, audio_limpo_np, arquivo_saida)

        except Exception as e:
            print("Erro:", e)
