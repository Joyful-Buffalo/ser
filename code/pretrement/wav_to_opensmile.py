import csv
import os


def to_cmd(extract_file_path, config_path, audio_path, output):
    cmd = extract_file_path + " -C " + config_path + ' -I ' + audio_path + " -csvoutput " + output
    return cmd


def csv_to_list(csv_path):
    with open(csv_path) as f:
        f_csv = csv.reader(f)
        header = next(f_csv)
        for row in f_csv:
            data = row[0].replace(';', ',').split(',')[1:]
    txt_path = csv_path.split('.csv')[0] + '.txt'
    print(txt_path)
    with open(txt_path, 'w+') as f:
        for d in data:
            f.write(d)
            f.write(',')
    os.remove(csv_path)
    return data


def extract_feature():
    path_extract_file = r"D:\opensmile-3.0-win-x64\bin\SMILExtract.exe"
    path_config = r"D:\opensmile-3.0-win-x64\config\is09-13\IS09_emotion.conf"

    print(len(os.listdir(wav_dir)))
    count = 0
    for i, audio in enumerate(os.listdir(wav_dir), 1):
        audio_path = os.path.join(wav_dir, audio)
        csv_path = output_dir + audio.split('.')[0] + '.csv'

        command = to_cmd(
            extract_file_path=path_extract_file,
            config_path=path_config,
            audio_path=audio_path,
            output=csv_path
        )
        os.system(command)
        csv_to_list(csv_path)
        count += 1
    print(count)


if __name__ == '__main__':
    wav_dir = r'D:\iemo\wav'
    output_dir = r"D:\python3.8Project\HHpaper\dataset\IEMOCAP\opensmile\\"
    extract_feature()
