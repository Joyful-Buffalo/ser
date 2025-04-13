import os
import wave

import numpy as np


def read_lis(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    star_lis = []
    end_lis = []
    emo_lis = []
    emo_need = ['sad', 'hap', 'neu', 'ang', 'exc']
    # emo_need = ['exc', 'hap']
    for line in lines:
        if line[0] == '[':
            emo = line.split('[')[-2][-4:-1]
            if emo in emo_need:
                time_ = line.split(']')[0].strip('[')
                star_lis.append(eval(time_.split('-')[0]))
                end_lis.append(eval(time_.split('-')[1]))
                emo_lis.append(emo)
    return star_lis, end_lis, emo_lis


def get_sav_path(sav_dir, emo, num_lis):
    if emo == 'neu':
        sav_path = sav_dir + "\\" + emo + str(num_lis[0]) + '.wav'
        num_lis[0] += 1
    elif emo == 'hap':
        sav_path = sav_dir + "\\" + emo + str(num_lis[1]) + '.wav'
        num_lis[1] += 1
    elif emo == 'ang':
        sav_path = sav_dir + '\\' + emo + str(num_lis[2]) + '.wav'
        num_lis[2] += 1
    elif emo == 'exc':
        sav_path = sav_dir + '\\' + emo + str(num_lis[3]) + '.wav'
        num_lis[3] += 1
    else:
        sav_path = sav_dir + '\\' + emo + str(num_lis[4]) + '.wav'
        num_lis[4] += 1
    return sav_path


def split_wav(num_lis, wav_path, txt_path, sav_dir):
    wave_file = wave.open(wav_path, 'rb')

    sample_rate = wave_file.getframerate()  # 采样率
    channel_num = wave_file.getnchannels()  # 通道数
    frame_num = wave_file.getnframes()  # 帧数

    # 将wav文件读取到numpy数组中
    wav_data = wave_file.readframes(frame_num)  # 按照通道数据从前往后写的一维数组
    wav_arr = np.frombuffer(wav_data, dtype=np.int16)

    # 将numpy数组重新组成一个二维数组，每行表示一个声道
    wav_arr = np.reshape(wav_arr, (frame_num, channel_num))

    star_lis, end_lis, emo_lis = read_lis(txt_path=txt_path)

    for star, end, emo in zip(star_lis, end_lis, emo_lis):
        star_pos = int(star * sample_rate)
        end_pos = int(end * sample_rate)
        temp_wav_arr = wav_arr[star_pos:end_pos]

        # save
        sav_path = get_sav_path(sav_dir, emo, num_lis)
        temp_wav_data = temp_wav_arr.tobytes()
        temp_wav_file = wave.open(sav_path, 'wb')
        temp_wav_file.setparams(wave_file.getparams())
        temp_wav_file.writeframes(temp_wav_data)
        temp_wav_file.close()


if __name__ == '__main__':
    NUM_lis = [1, 1, 1, 1, 1]
    # savDir = 'D:\\iemo\\wav'
    savDir = "D:\\SerPorjects\\HHpaper\\dataset\\IEMOCAP\\wav"
    ses_wav_dir = ['D:\\iemo\\session' + str(i) + '\\wav' for i in range(1, 6)]
    ses_txt_dir = ['D:\\iemo\\session' + str(i) + '\\EmoEvaluation' for i in range(1, 6)]

    for wav_dir, txt_dir in zip(ses_wav_dir, ses_txt_dir):
        txt_name = [f for f in os.listdir(txt_dir) if f.split('.')[-1] == 'txt']
        wav_name = [f for f in os.listdir(wav_dir) if f.split('.')[-1] == 'wav']
        for wavFile, txtFile in zip(wav_name, txt_name):
            wavPath = os.path.join(wav_dir, wavFile)
            txtPath = os.path.join(txt_dir, txtFile)
            split_wav(num_lis=NUM_lis, wav_path=wavPath, txt_path=txtPath, sav_dir=savDir)
