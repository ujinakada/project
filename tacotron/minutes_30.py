import os
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm

def minutes_30(input_path, output_folder, segment_duration=30 * 60 * 1000): #wavファイルを読み込み、30分ごとに分割する。get_only_vocalsのメモリが足りないため
    audio = AudioSegment.from_wav(input_path)

    for i in range(0, len(audio), segment_duration):
        segment = audio[i:i + segment_duration]

        # 分割されたファイルを保存
        output_path = f"{output_folder}/segment_{i // segment_duration }.wav"
        segment.export(output_path, format="wav")




if __name__ == "__main__":

    input_folder = "./wav/original"
    output_folder = "./wav/minutes_30"
    input_files = []
    output_folders = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_num = len(os.listdir(input_folder))

    for i in range(file_num):
        if not os.path.exists(output_folder + "/original_" + str(i)):
            os.makedirs(output_folder + "/original_" + str(i) )

        input_files.append(Path(input_folder + "/original_" + str(i) + ".wav"))
        output_folders.append(Path(output_folder + "/original_" + str(i)))

    with ProcessPoolExecutor(4) as executor:
        futures = [
            executor.submit(
                minutes_30,
                input_file,
                output_folder,
            )
            for input_file, output_folder in zip(input_files, output_folders)
        ]
        for future in tqdm(futures):
            future.result()