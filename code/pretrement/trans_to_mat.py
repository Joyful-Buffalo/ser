import json
import os.path

import numpy
from scipy import io
from scipy.fftpack import dct
from scipy.io import wavfile


def trans_to_mat(wav_path):
    sample_rate, signal = wavfile.read(wav_path)
    # signal = signal[:len(signal)]

    # 预强调
    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # 分帧处理
    frame = 0.064
    stride = 0.016
    frame_length, frame_step = frame * sample_rate, stride * sample_rate  # 秒数转为频率
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))  # 四舍五入  # 一帧包含频率数
    frame_step = int(round(frame_step))  # 一帧前进频率数
    num_frame = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # 确保帧数大于1

    pad_signal_length = num_frame * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z)
    indices = numpy.tile(numpy.arange(0, frame_length), (num_frame, 1)) + numpy.tile(
        numpy.arange(0, num_frame * frame_step, frame_step), (frame_length, 1)).T  # 获取每一帧的索引, 每一行为改帧的全部索引
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]  # 每一行为一帧

    # 加窗
    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

    # DFT
    NFFT = 512  # 每一帧计算nfft的采样点 一般使用512或者256
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # 能量谱

    # 梅尔滤波器
    num_filter = 40  # 滤波器数量, 默认40
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, num_filter + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)
    fbank = numpy.zeros((num_filter, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, num_filter + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB
    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    # print(filter_banks)

    # # 梅尔倒谱系数
    num_ceps = 40
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]
    nframes, ncoeff = mfcc.shape
    n = numpy.arange(ncoeff)

    cep_lifter = 22  # 默认是22
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift  # *

    # print(sav_path)
    return mfcc


def get_path(json_path):
    # json_path = '/home/pwy/lhc/HHpaper/dataset/SCRIPT_IEMOCAP/labels.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    # print(data['neu'], '\n', type(data['neu']))
    print(data.keys())
    return data


def IEMOCAP_main():
    savdir = '/home/pwy/lhc/HHpaper/dataset/SCRIPT_IEMOCAP/mfcc/'
    from_path = '/home/pwy/lhc/HHpaper/dataset/SCRIPT_IEMOCAP/labels.json'
    if not os.path.exists(savdir):
        os.mkdir(savdir)
    count = 0
    json_data = get_path(from_path)

    for key in json_data.keys():
        num = 0
        for wav_path in json_data[key]:
            feature = trans_to_mat(wav_path)
            sav_path = savdir + 'ang' + str(num) if key == 'anger' else savdir + str(key) + str(num)
            sav_path += '.mat'
            io.savemat(sav_path, {'vector': feature})
            count += 1
            num += 1
            print(sav_path)
    print(count)


def EMODB_main():
    savdir = '/home/pwy/lhc/HHpaper/dataset/EMODB/mfcc/'
    from_path = '/home/pwy/lhc/HHpaper/dataset/EMODB/wav/'

    if True:
        for i in os.listdir(savdir):
            path = savdir + i
            os.remove(path)
            print(path)

    if not os.path.exists(savdir):
        os.mkdir(savdir)
    count = 0

    for file_name in os.listdir(from_path):
        num = 0
        wav_path = from_path + file_name
        feature = trans_to_mat(wav_path)
        # sav_path = savdir + 'ang' + str(num) if key == 'anger' else savdir + str(key) + str(num)
        sav_path = savdir + file_name.split('.')[-2] + '.mat'
        io.savemat(sav_path, {'vector': feature})
        count += 1
        num += 1
        print(sav_path)
    print(count)


if __name__ == '__main__':
    # IEMOCAP_main()
    EMODB_main()
