import os
import librosa
import random
import shutil
import numpy as np
import pandas as pd
from config import CAMINHO_METADADOS, N_FOLDS, DIR_DATA

# N_SAMPLES_CLASS = 300


def remove_invalid_files(paths_files):
    # Remove all files ending with ".~1~"
    return [f for f in paths_files if f[-4:]  != ".~1~"]


def copy_files_folds(paths_files: list[str], classname: str):

    filenames_folds = []

    for fold in range(1, N_FOLDS+1):

        # Utilizado para calcular o número de amostras corretas no fold.
        # Serve para evitar que classes com menos amostras fiquem com
        # menos folds.
        n_files_fold = len(paths_files) // N_FOLDS

        files_fold = paths_files[(fold-1) * n_files_fold : n_files_fold * fold]

        fold_path = os.path.join(DIR_DATA, f'fold{fold}')
        os.makedirs(fold_path, exist_ok=True)

        for file_path in files_fold:

            file_ext = os.path.split(file_path)[-1].split('.')[-1]

            time_audio = hash(file_path)#datetime.now().microsecond
            dst_path = os.path.join(fold_path, f'audio{time_audio}_{classname}.{file_ext}')

            shutil.copy(file_path, dst_path)

            filenames_folds.append((dst_path, fold))
            print(dst_path, 'copied.')
    return filenames_folds


def get_info_files_class(filenames_folds: list[str], classname: str
                         ) -> list[str]:
    info_files = []

    for fpath, fold in filenames_folds:
        filename = os.path.split(fpath)[-1]
        start = 0.0
        end = librosa.get_duration(filename=fpath)
        duration = end - start

        info_files.append([filename, start, end, fold, duration, classname])

    return info_files


def main():
    # Remove folds folders
    for fold in range(1, N_FOLDS+1):
        fold_path = os.path.join(DIR_DATA, f'fold{fold}')
        shutil.rmtree(fold_path, ignore_errors=True)

    classes = [path_class for path_class in os.listdir(DIR_DATA)
               if os.path.isdir(os.path.join(DIR_DATA, path_class))]
    class_folders = [os.path.join(DIR_DATA, path) for path in classes]

    info_audios = []
    # Copy files to fold
    for c, class_folder in enumerate(class_folders):
        paths_files = [os.path.join(class_folder, f)
                       for f in os.listdir(class_folder)]

        paths_valid_files = remove_invalid_files(paths_files)
        random.shuffle(paths_valid_files)

        print("Copying files for", classes[c])
        filenames_folds = copy_files_folds(paths_valid_files, classes[c])

        # Extrai informações dos arquivos dessa classe
        info_audios += get_info_files_class(filenames_folds, classes[c])
        # get_info_files_class(filenames_folds, classes[c])

    columns_names = ['slice_file_name', 'start', 'end', 'fold', 'duration', 'class']
    info_audios = np.array(info_audios)
    df_info_audios = pd.DataFrame(data=info_audios, columns=columns_names)
    df_info_audios.to_csv(CAMINHO_METADADOS, index=False)

    print(CAMINHO_METADADOS, "salvo.")

if __name__ == "__main__":
    main()
