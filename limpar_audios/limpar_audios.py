# /home/gustavo/OneDrive_gustavotorres@ipt.br/Ciclo 3/Greenbug/Codigos/audioset/motocicletas
# wav, mpga
import os
import re
import random
import shutil
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import unicodedata
from tqdm import tqdm
# from scipy.io.wavfile import read
# from pydub.silence import split_on_silence
# from pydub import AudioSegment

CAMINHO_AUDIOS = "data_audioset"
CAMINHO_AUDIOS_LIMPOS = "data_clean_audioset"
CAMINHO_WAVEFORMS = "waveforms_audioset"
TIPOS_AUDIOS = ["carros", "motocicletas", "motosserras"]

N_MAX_ARQUIVOS = 641
SAMPLE_RATE = 22050


def clean_filename( name: str, max_len: int = 120, replace_with: str = "_"
                   ) -> str:
    # Normalize unicode (é → e, ： → :)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    # Split extension
    base, ext = os.path.splitext(name)

    # Replace illegal characters
    base = re.sub(r'[<>:"/\\|?*\#\(\)\[\]\{\}]', replace_with, base)

    # Replace any remaining non-safe chars
    base = re.sub(r'[^A-Za-z0-9._ -]', replace_with, base)

    # Collapse repeated separators
    base = re.sub(rf'{replace_with}+', replace_with, base)

    # Trim spaces / dots
    base = base.strip(" ._-")

    # Truncate safely
    base = base[:max_len]

    # Fallback name
    if not base:
        base = "file"

    base = re.sub(rf'\s+', '_', base)
    return base + ext


def split_on_silence(y, min_silence_len=500, silence_thresh=50):
    intervals = librosa.effects.split(
        y,
        top_db=silence_thresh,          # similar to silence_thresh=-50 dB
        frame_length=2048,
        hop_length=512
    )

    # Extract segments
    audio_segs = [y[start:end] for start, end in intervals]
    return audio_segs


def limpar_audio(audio_y, rate):
    # anything under silence_thresh dBFS is considered silence
    audio_segs = split_on_silence(audio_y)

    output = [audio_segs[0]]
    for seg in audio_segs[1:]:
        output.append(seg)
    
    output = np.hstack(output)
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

def salvar_audio_limpo(y, sr, arquivo_saida):
    sf.write(arquivo_saida, y, sr)
    # audio_limpo.export(arquivo_saida, format="mp3")
    print(f"{arquivo_saida} salvo com sucesso.")


def main():

    shutil.rmtree(CAMINHO_AUDIOS_LIMPOS, ignore_errors=True)
    for tipo_audio in TIPOS_AUDIOS:
        os.makedirs(os.path.join(CAMINHO_AUDIOS_LIMPOS, tipo_audio), exist_ok=True)
    #for tipo_audio in tipos_audios:
    #    os.makedirs(os.path.join(caminho_waveforms, tipo_audio), exist_ok=True)

    for tipo in TIPOS_AUDIOS:
        print(f"======= Processando áudios de {tipo} =======")
        arquivos_audio = os.listdir(os.path.join(CAMINHO_AUDIOS, tipo))
        random.shuffle(arquivos_audio)
                                    
        arquivos_audio = [
            arquivo for arquivo in arquivos_audio
            if ".~1~" not in arquivo and ".webm" not in arquivo
        ][:N_MAX_ARQUIVOS]

        for arquivo in tqdm(arquivos_audio):
            caminho_arquivo = os.path.join(CAMINHO_AUDIOS, tipo, arquivo)
            #try:
            y, sr = librosa.load(caminho_arquivo)
            # audio = AudioSegment.from_file(caminho_arquivo)

            y_limpo = limpar_audio(y, SAMPLE_RATE)
            # audio_limpo_np = np.array(audio_limpo.get_array_of_samples())
            # audio_bruto_np = np.array(audio.get_array_of_samples())

            # arquivo_saida = os.path.join(caminho_audios_limpos, tipo, f"{arquivo[:-4]}.wav")
            arquivo_saida = os.path.join(
                CAMINHO_AUDIOS_LIMPOS, tipo, clean_filename(f"{arquivo[:-4]}.wav"))
            salvar_audio_limpo(y_limpo, sr, arquivo_saida)

            # arquivo_saida = os.path.join(caminho_waveforms, tipo, f"{arquivo[:-4]}.png")
            # salvar_wave_audio(audio_bruto_np, audio_limpo_np, arquivo_saida)

            #except Exception as e:
            #    print("Erro:", e)

if __name__ == "__main__":
    main()

