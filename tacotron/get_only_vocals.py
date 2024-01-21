import os
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import demucs.separate

def get_only_vocals(input_folder, output_folder):
    file_count = len(os.listdir(input_folder))

    for i in range(file_count):
        options = [str(input_folder / f"segment_{i}.wav"),  # str()を使ってPathを文字列に変換
                   "-n", "htdemucs",
                   "--two-stems", "vocals",
                   "-o", str(output_folder)  # 同様に、output_folderもstr()で文字列に変換
                   ]
        demucs.separate.main(options)


if __name__ == "__main__":
    input_folder = Path("./wav/minutes_30")
    output_folder = Path("./wav/only_vocals")
    input_files = []
    output_folders = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_num = len(os.listdir(input_folder))

    for i in range(file_num):
        new_output_folder = output_folder / f"original_{i}"
        if not os.path.exists(new_output_folder):
            os.makedirs(new_output_folder)

        input_files.append(input_folder / f"original_{i}")
        output_folders.append(new_output_folder)

    with ProcessPoolExecutor(4) as executor:
        futures = [
            executor.submit(
                get_only_vocals,
                input_file,
                output_folder,
            )
            for input_file, output_folder in zip(input_files, output_folders)
        ]
        for future in tqdm(futures):
            future.result()