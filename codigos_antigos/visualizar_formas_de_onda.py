# /home/gustavo/OneDrive_gustavotorres@ipt.br/Ciclo 3/Greenbug/Codigos/audioset/motocicletas
import os
import librosa
import matplotlib.pyplot as plt


waveforms_path = "./waveforms"
os.makedirs(os.path.join(waveforms_path, "carros"), exist_ok=True)
os.makedirs(os.path.join(waveforms_path, "motocicletas"), exist_ok=True)
os.makedirs(os.path.join(waveforms_path, "motosserras"), exist_ok=True)


def salvar_waveform(arquivo_leitura, arquivo_saida, sample_rate=22050):
    signal, sample_rate = librosa.load(arquivo_leitura, sr=sample_rate)
    # librosa.display.waveshow(signal)
    arquivo_leitura = arquivo_leitura.replace(".mp3", ".png")
    plt.plot(signal);
    plt.title('Signal');
    plt.xlabel('Time (samples)');
    plt.ylabel('Amplitude');
    plt.savefig(arquivo_saida)
    print(f"{arquivo_saida} salvo com sucesso.")
    plt.clf()


diretorios = ["carros", "motocicletas", "motos"]

for diretorio in diretorios:
    arquivos_audio = os.listdir(os.path.join("audioset", diretorio))
    for arquivo in arquivos_audio:
        try:
            caminho_mp3 = os.path.join("audioset", diretorio, arquivo)
            caminho_destino = os.path.join(waveforms_path, diretorio, arquivo.replace(".mp3", ".png"))
            salvar_waveform(caminho_mp3, caminho_destino)
        except Exception as e:
            print("Erro:", e)
