import math
import os

import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Pretreatment:
    def __init__(self, txt_path='', each_data="EMODB", fit_frame=400, shuffle=False, train_rate=0.8):
        self.each_data = each_data
        is_online = True if os.path.abspath(__file__)[0] == '/' else False
        if is_online and os.path.abspath(__file__).split('/')[1] == 'home':
            pre_path = '/home/pwy/lhc/'
        else:
            pre_path = '/hy-tmp/'
        self.dataset_path = (pre_path if is_online else '../../') + 'HHpaper/dataset/' + each_data + '/' + 'wav'
        self.fit_frame = fit_frame
        if shuffle:
            self.split_data(txt_path=txt_path, dataset_path=self.dataset_path, train_rate=train_rate)
        self.files = self.get_files(txt_path)

    def wav_to_mfcc(self):
        data = []
        for f in self.files:
            mat_path = f.replace(os.path.dirname(f).split('/')[-1], 'wav')
            wav_path = mat_path.replace('.txt', '.wav').replace('.mat', '.wav')
            j = wav_path.split('.')[-2].split('/')[-1].split('_')[-1]
            if j == '144':
                continue
            y, sr = librosa.load(wav_path, sr=None)
            y = y / np.max(np.abs(y))
            hop_length = int(0.016 * sr)
            window = 'hamming'
            n_mfcc = 40
            n_fft = 1024
            fmin = 40  # 40 Hz
            fmax = 7600  # 7600 Hz
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                         hop_length=hop_length, n_fft=n_fft,
                                         window=window, center=True,
                                         fmin=fmin, fmax=fmax)
            mfccs = mfccs[:40]
            mfccs = mfccs.T
            data.append(mfccs)
        return data

    def get_emo_labels(self):
        labels = [self.get_emo_label(f.split('.')[-2].split('/')[-1].split('_')[0]) for f in self.files]
        return labels

    @staticmethod
    def get_emo_label(label_idx):
        # if label_idx == 'W':  # angry
        #     label = [1, 0, 0, 0, 0, 0, 0]
        # elif label_idx == 'L':  # boredom
        #     label = [0, 1, 0, 0, 0, 0, 0]
        # elif label_idx == 'E':  # disgust
        #     label = [0, 0, 1, 0, 0, 0, 0]
        # elif label_idx == 'A':  # fear
        #     label = [0, 0, 0, 1, 0, 0, 0]
        # elif label_idx == 'F':  # happy
        #     label = [0, 0, 0, 0, 1, 0, 0]
        # elif label_idx == 'T':  # sad
        #     label = [0, 0, 0, 0, 0, 1, 0]
        # else:  # neutral
        #     label = [0, 0, 0, 0, 0, 0, 1]
        # return label
        if label_idx == '1':  # ang
            label = [1, 0]
        elif label_idx == '2':  # neu
            label = [0, 1]
        # elif label_idx == 'h' or label_idx == 'j':  # hap
        #     label = [0, 0, 1, 0]
        # else:  # sad
        #     label = [0, 0, 0, 1]
        return label

    def make_matrix(self, mfcc_datas):
        new_data = []
        for data in mfcc_datas:
            if data.shape[0] == self.fit_frame:
                matrix = data
            elif data.shape[0] > self.fit_frame:
                star_idx = int((data.shape[0] - self.fit_frame) / 2)
                matrix = data[star_idx:star_idx + self.fit_frame]
            else:
                times = math.ceil(self.fit_frame / data.shape[0])
                t_matrix = np.tile(data, (times, 1))
                matrix = t_matrix[:self.fit_frame]
            new_data.append(matrix.ravel())
        return new_data

    def get_fit_frame(self):
        return self.fit_frame

    # @staticmethod
    def get_files(self, path):
        f = open(path, 'r')
        files = f.readlines()
        f.close()
        files = [s.strip() for s in files]
        return files

    @staticmethod
    def split_data(txt_path, dataset_path, train_rate=0.8):
        files_name = os.listdir(dataset_path)
        file_path = [dataset_path + "/" + file_name for file_name in files_name]
        file_num = len(file_path)
        index = [i for i in range(file_num)]
        # np.random.shuffle(index)
        split_num = int(file_num * train_rate)
        with open(txt_path, 'w+') as f:
            for idx in index[:split_num]:
                f.write(file_path[idx])
                f.write('\n')
        test_path = txt_path.replace('Training', 'test')
        with open(test_path, 'w+') as f:  # test
            for idx in index:
                f.write(file_path[idx])
                f.write('\n')


class MySet(Dataset):
    def __init__(self, is_train_set=True, each_data='EMODB', each_feature='total', fit_frame=400, shuffle=False,
                 train_rate=0.8, output_size=7):
        self.dataset_path = '../../HHpaper/dataset/' + each_data + '/mfcc'
        self.txt_filename = self.get_txt_file_name(each_data=each_data, is_train_set=is_train_set)
        self.Pre_class = Pretreatment(
            txt_path=self.txt_filename,
            fit_frame=fit_frame,
            shuffle=shuffle,
            train_rate=train_rate,
            each_data=each_data,
        )
        self.data, self.mfcc_col = self.get_data(each_feature=each_feature)
        self.labels = self.get_labels(each_data=each_data)
        self.label_list = [[0] * output_size for _ in range(output_size)]
        self.files = self.Pre_class.get_files(path=self.txt_filename)
        for i in range(output_size):
            self.label_list[i][i] = 1
        self.label_dict = self.make_dict()

    def __getitem__(self, idx):
        item = self.data[idx], self.label_dict[tuple(self.labels[idx])]
        return item

    def __len__(self):
        return len(self.data)

    def make_dict(self):
        label_dict = dict()
        for idx, label in enumerate(self.label_list, 0):
            label_dict[tuple(label)] = idx
        return label_dict

    def get_mfcc_column(self):
        return self.mfcc_col

    @staticmethod
    def get_txt_file_name(each_data, is_train_set):
        if each_data == 'EVALUATE':
            txt_filename = '../eval_train_path.txt' if is_train_set else '../eval_test_path.txt'
        else:
            raise NotImplementedError()
        if not os.path.exists(txt_filename):
            open(txt_filename, 'w+').close()
        return txt_filename

    def get_data(self, each_feature):
        if each_feature == 'mfcc':
            data = self.Pre_class.wav_to_mfcc()
            mfcc_col = data[1].shape[1]
            data = self.Pre_class.make_matrix(data)
        else:
            raise NotImplementedError()
        return data, mfcc_col

    def get_labels(self, each_data):
        if each_data == 'EVALUATE':
            labels = self.Pre_class.get_emo_labels()
        else:
            raise NotImplementedError()
        return labels

    def get_files(self):
        return self.files


def get_loader(batch_size=128, each_data='EMODB', each_feature="total", fit_frame=400, output_size=7):
    evaluate_set = MySet(
        is_train_set=False,
        each_feature=each_feature,
        fit_frame=fit_frame,
        each_data=each_data,
        output_size=output_size,
        shuffle=True
    )
    mfcc_column = MySet.get_mfcc_column(evaluate_set)
    files = MySet.get_files(evaluate_set)
    eval_loader = DataLoader(evaluate_set, batch_size=batch_size, shuffle=False)
    return eval_loader, mfcc_column, files
