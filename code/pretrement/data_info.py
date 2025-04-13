import os

import torchaudio
from matplotlib import pyplot as plt


def get_wav_len(wav_path):
    audio, sr = torchaudio.load(wav_path)
    return audio.shape[1] / sr


def get_iemocap_len(dir_path):
    wav_paths = [dir_path + i for i in os.listdir(dir_path)]
    ang_len, sad_len, joy_len, neu_len = [], [], [], []
    count = 0
    for wav_path in wav_paths:
        if 'ang' in wav_path:
            ang_len.append(get_wav_len(wav_path))
        elif 'sad' in wav_path:
            sad_len.append(get_wav_len(wav_path))
        elif 'joy' in wav_path:
            # joy_len.append(get_wav_len(wav_path))
            count +=1
        else:
            neu_len.append(get_wav_len(wav_path))

    # print('ang', sum(ang_len) / len(ang_len), len(ang_len), ang_len[int(len(ang_len) // 2)])
    # print('sad', sum(sad_len) / len(sad_len), len(sad_len), sad_len[int(len(sad_len) // 2)])
    # print('joy', sum(joy_len) / len(joy_len), len(joy_len), joy_len[int(len(joy_len) // 2)])
    # print('neu', sum(neu_len) / len(neu_len), len(neu_len), neu_len[int(len(neu_len) // 2)])
    # print('total', (sum(ang_len) + sum(sad_len) + sum(joy_len) + sum(neu_len)) / (
    #         len(ang_len) + len(sad_len) + len(joy_len) + len(neu_len)))
    # show_len(ang_len, 'ang')
    # show_len(sad_len, 'sad')
    # show_len(joy_len, 'joy')
    # show_len(neu_len, 'neu')
    print(count)


def show_len(len_, comment):
    len_ = sorted(len_)
    print(comment, sum(len_) / len(len_), len(len_), len_[int(len(len_) // 2)])
    plt.plot([i for i in range(len(len_))], len_)
    plt.show()


if __name__ == '__main__':
    print('IMPRO')
    path = '/home/pwy/lhc/HHpaper/dataset/IMPRO_IEMOCAP/wav/'
    get_iemocap_len(path)
    print('total')
    path = '/home/pwy/lhc/HHpaper/dataset/IEMOCAP/wav/'
    get_iemocap_len(path)
