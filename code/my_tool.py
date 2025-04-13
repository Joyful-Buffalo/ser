import datetime
import os
import re

import librosa
import math
import numpy as np
import torch
import torchaudio
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB
from tqdm import tqdm


# from pretrement.trans_logmelspectrogram import LogMelSpectrogram


class Pretreatment:
    def __init__(self, txt_path='', each_data="EMODB", fit_frame=400, shuffle=False, train_rate=0.8, test_speaker='03'):
        self.each_data = each_data
        is_online = True if os.path.abspath(__file__)[0] == '/' else False
        if is_online and os.path.abspath(__file__).split('/')[1] == 'home':
            pre_path = '/home/pwy/lhc/'
        else:
            pre_path = '/hy-tmp/'
        self.dataset_path = (pre_path if is_online else '../../') + 'HHpaper/dataset/' + each_data + '/' + 'wav'
        self.fit_frame = fit_frame
        if shuffle:
            self.split_data(train_path=txt_path, dataset_path=self.dataset_path, test_speaker=test_speaker)
        self.files = self.get_files(txt_path)

    def get_opensmile(self):
        opensmile_data = np.asarray(())
        for f in self.files:
            # get opensmile_data
            txt_path = f.replace(os.path.dirname(f).split('/')[-1], 'opensmile')
            txt_path = txt_path.replace('.mat', '.txt')
            f = open(txt_path)
            last_line = f.readlines()[-1]
            f.close()
            feature = np.array(last_line.split(','))
            feature = np.transpose(feature[1:-1])  # 去掉第一个和最后一个元素 并转置
            if opensmile_data.size == 0:
                opensmile_data = feature
            else:
                opensmile_data = np.vstack((opensmile_data, feature))  # 将列数相同的矩阵堆起来
        return opensmile_data  # 返回原始竖向堆起来的，没有padding的数据

    def get_mfcc(self):
        datas = []
        for f in self.files:
            mat_path = f.replace(os.path.dirname(f).split('/')[-1], 'mfcc')
            mat_path = mat_path.replace('.txt', '.mat')
            m = loadmat(mat_path)
            data = np.array(m['vector'])
            datas.append(data)  # MFCC特征矩阵

        return datas  # 返回原始数据[n_sample, frame, ]

    def wav_to_mfcc(self):
        data = []
        for f in self.files:
            mat_path = f.replace(os.path.dirname(f).split('/')[-1], 'wav')
            wav_path = mat_path.replace('.txt', '.wav').replace('.mat', '.wav')
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

    def get_log_mel_spectrogram(self):
        data = []
        transform = LogMelSpectrogram(sample_rate=16000, audio_length=3, n_fft=2048, hop_length=512)
        for f in self.files:
            wav_path = f.replace(os.path.dirname(f).split('/')[-1], 'wav')
            jpg_path = wav_path.replace('.txt', '.wav').replace('.mat', '.wav')
            audio, _ = torchaudio.load(jpg_path)
            audio = torch.mean(audio, 0)
            log_mel = transform(audio)
            data.append(np.array(log_mel))
        return data

    def get_emo_labels(self):
        labels = [self.get_emo_label(f.split('.')[-2][-2]) for f in self.files]
        return labels

    def get_iem_labels(self):
        labels = [self.get_iem_label(f.split('.')[-2][-1]) for f in self.files]
        return labels

    @staticmethod
    def get_emo_label(label_idx):
        if label_idx == 'W':  # angry
            label = [1, 0, 0, 0, 0, 0, 0]
        elif label_idx == 'L':  # boredom
            label = [0, 1, 0, 0, 0, 0, 0]
        elif label_idx == 'E':  # disgust
            label = [0, 0, 1, 0, 0, 0, 0]
        elif label_idx == 'A':  # fear
            label = [0, 0, 0, 1, 0, 0, 0]
        elif label_idx == 'F':  # happy
            label = [0, 0, 0, 0, 1, 0, 0]
        elif label_idx == 'T':  # sad
            label = [0, 0, 0, 0, 0, 1, 0]
        else:  # neutral
            label = [0, 0, 0, 0, 0, 0, 1]
        return label

    @staticmethod
    def get_iem_label(label_idx):  # ['anger', 'sad', 'neu', 'joy']
        if label_idx == 'g':
            label = [1, 0, 0, 0]
        elif label_idx == 'd':
            label = [0, 1, 0, 0]
        elif label_idx == 'u':
            label = [0, 0, 1, 0]
        else:
            label = [0, 0, 0, 1]
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

    def read_data(self):  # path :'train_path.txt'
        mfcc_data = self.get_mfcc()
        mfcc_column = mfcc_data[1].shape[1]
        mfcc_matrix = self.make_matrix(mfcc_data)

        opensmile_data = self.get_opensmile()
        min_max_scaler = preprocessing.MaxAbsScaler()
        opensmile_data = min_max_scaler.fit_transform(opensmile_data)  # openSmile数据进行缩放，使得数据的最大绝对值为1

        combine_data = np.hstack((mfcc_matrix, opensmile_data))  # 将两个数组沿水平方向拼接
        return combine_data, mfcc_column

    def get_fit_frame(self):
        return self.fit_frame

    @staticmethod
    def get_files(path):
        f = open(path, 'r')
        files = f.readlines()
        f.close()
        files = [s.strip() for s in files]
        return files

    # def split_data(self,train_path, dataset_path, train_rate=0.8):
    def split_data(self, train_path, dataset_path, test_speaker='03'):
        # file_path = [dataset_path + "/" + file_name for file_name in os.listdir(dataset_path)]
        train_data, test_data = [], []
        for i in os.listdir(dataset_path):
            if self.each_data == 'EMODB':
                speaker = i[:2]
            elif "IEMOCAP" in self.each_data:
                speaker = i[3:5] + i[-11]
            else:
                raise NotImplementedError
            if speaker == test_speaker:
                test_data.append(os.path.join(dataset_path, i))
            else:
                train_data.append(os.path.join(dataset_path, i))
        # file_num = len(train_data)
        # train_index = [i for i in range(len(train_data))]
        # test_index = [i for i in range(len(test_data))]
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        with open(train_path, 'w+') as f:
            for p in train_data:
                f.write(p)
                f.write('\n')
            # pass
        test_path = train_path.replace('Training', 'test')
        with open(test_path, 'w+') as f:
            for p in test_data:
                f.write(p)
                f.write('\n')


class MySet(Dataset):
    def __init__(self, is_train_set=True, each_data='EMODB', each_feature='total', fit_frame=400, shuffle=False,
                 train_rate=0.8, output_size=7, data_augmentation=True, noise_rate=None, new_data_num=1,
                 augmentation_func=None, test_speaker='03'):
        self.dataset_path = '../../HHpaper/dataset/' + each_data + '/mfcc'
        self.txt_filename = self.get_txt_file_name(each_data=each_data, is_train_set=is_train_set)
        self.Pre_class = Pretreatment(
            txt_path=self.txt_filename,
            fit_frame=fit_frame,
            shuffle=shuffle,
            train_rate=train_rate,
            each_data=each_data,
            test_speaker=test_speaker,
        )
        self.data, self.mfcc_col = self.get_data(each_feature=each_feature)
        self.labels = self.get_labels(each_data=each_data)

        if data_augmentation:
            func_list = []
            if isinstance(augmentation_func, list):
                for i in augmentation_func:
                    func_list.append(self.get_augment_func(i))
            else:
                func_list.append(self.get_augment_func(augmentation_func))
            if not isinstance(noise_rate, list):
                noise_rate = [noise_rate]
            self.data = np.asarray(self.data)
            data = None
            for func, rate in zip(func_list, noise_rate):
                noise = np.random.normal(size=self.data.shape)
                new_data = func(self.data, noise, rate)
                data = np.vstack([self.data, new_data]) if data is None else np.vstack([data, new_data])
            self.data = data

            self.labels = self.labels[:] * (len(func_list) + 1)
        self.label_list = [[0] * output_size for _ in range(output_size)]
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
        if each_data == "EMODB":
            txt_filename = '../train_path.txt' if is_train_set else '../test_path.txt'
        elif each_data == "IEMOCAP":
            txt_filename = '../iem_train_path.txt' if is_train_set else '../iem_test_path.txt'
        elif each_data == 'IMPRO_IEMOCAP':
            txt_filename = '../impro_iemo_train_path.txt' if is_train_set else '../impro_iemo_test_path.txt'
        elif each_data == "SCRIPT_IEMOCAP":
            txt_filename = '../script_iemo_train_path.txt' if is_train_set else '../script_iemo_test_path.txt'
        else:
            raise NotImplementedError()
        if not os.path.exists(txt_filename):
            open(txt_filename, 'w+').close()
        return txt_filename

    def get_data(self, each_feature):
        if each_feature == 'mfcc':
            data = self.Pre_class.wav_to_mfcc()
            # data = self.Pre_class.get_mfcc()
            mfcc_col = data[1].shape[1]
            data = self.Pre_class.make_matrix(data)
        elif each_feature == 'opensmile':
            data = self.Pre_class.get_opensmile()
            data = data.astype(float)
            mfcc_col = None
        elif each_feature == 'log-mel':
            data = self.Pre_class.get_log_mel_spectrogram()
            mfcc_col = data[1].shape[1]
            data = self.Pre_class.make_matrix(data)
        elif each_feature == 'total':
            data, mfcc_col = self.Pre_class.read_data()
            data = data.astype(float)
        else:
            raise NotImplementedError()
        return data, mfcc_col

    def get_labels(self, each_data):
        if each_data == "EMODB":
            labels = self.Pre_class.get_emo_labels()
        elif each_data == "IEMOCAP" or each_data == "SCRIPT_IEMOCAP" or each_data == "IMPRO_IEMOCAP":
            labels = self.Pre_class.get_iem_labels()
        else:
            raise NotImplementedError()
        return labels

    def get_augment_func(self, i):
        if i == 'xy+x':
            # func = lambda x, y: x + (y * noise_rate * x)
            func = self.xy_plus_x
        elif i == 'x+y':
            # func = lambda x, y: x + (y * noise_rate)
            func = self.x_plus_y
        elif i == 'sqrt(x)*y+x':
            # func = lambda x, y: x + (np.sign(x) * np.sqrt(x) * y * noise_rate)
            func = self.sqrt_xy_plus_x
        elif i == 'random':
            func = self.random_plus_noise
        else:
            raise NotImplementedError()
        return func

    @staticmethod
    def xy_plus_x(x, noise, noise_rate):
        return x + noise * noise_rate * x

    @staticmethod
    def x_plus_y(x, noise, noise_rate):
        return x + noise * noise_rate

    @staticmethod
    def sqrt_xy_plus_x(x, noise, noise_rate):
        return np.sqrt(x) * np.sign(x) * noise_rate * noise + x

    @staticmethod
    def random_plus_noise(data, noise, noise_rate=0.005):
        total = len(data)
        idx = np.random.choice(total, total // 2, replace=False)
        data[idx] = data[idx] * noise[idx] * noise_rate + data[idx]
        return data

    @staticmethod
    def get_noise_in_data(data, labels, noise_rate=0.005, new_data_num=1):
        new_data = []
        for _ in range(new_data_num):
            for i in data:
                noise = np.random.normal(size=i.shape)
                i = i * noise * noise_rate + i
                new_data.append(i)
        data += new_data
        new_label = []
        for _ in range(new_data_num):
            new_label.extend(labels[:])
        labels += new_label
        return data, labels


def get_loader(batch_size=128, each_data='EMODB', each_feature="total", shuffle=False, rate=0.8, fit_frame=400,
               output_size=7, tr_set_shuffle=True, data_augmentation=False, noise_rate=0.0005, augment_func=None,
               seed=None, test_speaker='03'):
    if seed is not None:
        pass
        # np.random.seed(seed=seed)

    tr_set = MySet(
        is_train_set=True,
        each_feature=each_feature,
        shuffle=shuffle,
        train_rate=rate,
        fit_frame=fit_frame,
        each_data=each_data,
        output_size=output_size,
        data_augmentation=data_augmentation,
        noise_rate=noise_rate,
        augmentation_func=augment_func,
        test_speaker=test_speaker,
    )
    te_set = MySet(
        is_train_set=False,
        each_feature=each_feature,
        fit_frame=fit_frame,
        each_data=each_data,
        output_size=output_size,
        data_augmentation=data_augmentation,
        noise_rate=noise_rate,
        augmentation_func=augment_func,
        test_speaker=test_speaker,
    )
    mfcc_column = MySet.get_mfcc_column(tr_set)
    train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=tr_set_shuffle)
    test_loader = DataLoader(te_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(tr_set), len(te_set), mfcc_column


def train(model, loader, device, optimizer, criterion, pbar):
    correct = 0
    pred_list = []
    label_list = []
    train_loss = 0
    total = len(loader.dataset)
    for i, (combines, labels) in enumerate(loader, 1):
        inputs, target = make_tensors(combines, labels, device=device)
        output = model(inputs)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            _, pred = torch.max(output.data, dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            pred_list.append(pred)
            label_list.append(target)

        train_loss += loss.item()
        pbar.update(len(combines))
    return correct / total, train_loss, pred_list, label_list


def test(model, loader, device, criterion, pbar):
    correct = 0
    total = len(loader.dataset)
    test_loss = 0.0
    label_list, pred_list = [], []
    with torch.no_grad():
        for i, (combines, labels) in enumerate(loader, 1):
            inputs, target = make_tensors(combines, labels, device=device)  # 区分labels和target， labels make_tensor 后不容易拉直
            output = model(inputs)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, pred = torch.max(output.data, dim=1)

            correct += pred.eq(target.view_as(pred)).sum().item()
            pred_list.append(pred)
            label_list.append(target)
            pbar.update(len(combines))

    return correct / total, test_loss, pred_list, label_list


def main(test_loader, train_loader, tr_num, te_num, scheduler, model,
         epochs=50, stop=False, device=None, optimizer=None, criterion=None,
         train_func=None, test_func=None, fitlog=None, comment=None, args=None):
    train_func = train if train_func is None else train_func
    test_func = test if test_func is None else test_func
    te_acc_lis = []
    tr_acc_lis = []
    te_lo_lis = []
    tr_lo_lis = []
    tr_acc = 0
    te_label_lis = []
    te_pred_lis = []
    tr_label_lis = []
    tr_pred_lis = []
    best_f1, best_uar, best_wa = 0, 0, 0
    with tqdm(total=epochs * (tr_num + te_num)) as pbar:
        for epoch in range(1, epochs + 1):
            tr_acc, tr_lo, tr_labels, tr_prediction = train_func(model=model,
                                                                 loader=train_loader,
                                                                 device=device,
                                                                 pbar=pbar,
                                                                 criterion=criterion,
                                                                 optimizer=optimizer)
            te_acc, te_lo, te_labels, te_prediction = test_func(model=model,
                                                                loader=test_loader,
                                                                device=device,
                                                                criterion=criterion,
                                                                pbar=pbar)
            te_acc_lis.append(te_acc)
            tr_acc_lis.append(tr_acc)
            tr_lo_lis.append(tr_lo / tr_num)
            te_lo_lis.append(te_lo / te_num)
            te_label_lis.append(te_labels)
            tr_label_lis.append(tr_labels)
            te_pred_lis.append(te_prediction)
            tr_pred_lis.append(tr_prediction)
            scheduler.step()
            if epoch % 5 == 0:
                print(f'\rTraining loss{tr_lo / tr_num}')
                print(f'\rtest  loss{te_lo / te_num}')
                print(f'accuracy on test:{te_acc:.5g}%')
                print(f'\r[EPOCH {epoch}]', f'accuracy in train_set:{tr_acc}%')
            if epoch > 0 and te_acc == max(te_acc_lis):
                max_labels_lis = te_labels
                max_te_pred_lis = te_prediction
                # sav_pt
                formatted_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                model_name = type(model).__name__
                os.makedirs('trained_model/', exist_ok=True)
                save_pt_path = '../trained_model/' + model_name + args.each_data + formatted_time + f'{int(te_acc)}' + '.pt'
                # torch.save(model, save_pt_path)

            if te_lo / te_num > 5 * tr_lo / tr_num and stop:
                break

            # _, te_recall, te_fscore, _ = precision_recall_fscore_support(
            #     y_true=torch.hstack(te_labels).cpu().numpy(),
            #     y_pred=torch.hstack(te_prediction).cpu().numpy(), zero_division=1)
            # _, tr_recall, tr_fscore, _ = precision_recall_fscore_support(
            #     y_true=torch.hstack(tr_labels).cpu().numpy(),
            #     y_pred=torch.hstack(tr_prediction).cpu().numpy(), zero_division=1)
            # tmp_uar = sum(te_recall) / len(te_recall)
            # tmp_f1 = sum(te_fscore) / len(te_fscore)
            y_pred = torch.hstack(te_prediction).cpu().numpy()
            y_true = torch.hstack(te_labels).cpu().numpy()
            # report = classification_report(y_true, y_pred, target_names=Plot.get_type_lis(args.each_data), digits=6,
            #                                zero_division=0)
            report = classification_report(y_true, y_pred,
                                           labels=[i for i in range(7 if args.each_data == "EMODB" else 4)],
                                           digits=6,
                                           zero_division=0)
            tmp_wa = eval(re.findall(r'weighted avg\s+([\d.]+)', report)[0])
            values = re.findall(r'macro avg\s+[\d.]+\s+([\d.]+)\s+([\d.]+)', report)
            tmp_uar, tmp_f1 = eval(values[0][0]), eval(values[0][1])
            if fitlog is not None:
                fitlog.add_loss(tr_lo / tr_num, name=f'{comment}train_loss', step=epoch)
                fitlog.add_loss(te_lo / te_num, name=f'{comment}test_loss', step=epoch)
                fitlog.add_metric({f'{comment}Training': {'acc': tr_acc}}, step=epoch)
                # fitlog.add_metric({f"{comment}Training": {'UAR': sum(tr_recall) / len(tr_recall)}}, step=epoch)
                # fitlog.add_metric({f"{comment}Training": {'Macro_F1': sum(tr_fscore) / len(tr_fscore)}}, step=epoch)
                fitlog.add_metric({f'{comment}test': {'acc': te_acc}}, step=epoch)
                fitlog.add_metric({f"{comment}test": {'UAR': tmp_uar}}, step=epoch)
                fitlog.add_metric({f"{comment}test": {'Macro_F1': tmp_f1}}, step=epoch)
                fitlog.add_metric({f"{comment}test": {'wa': tmp_wa}}, step=epoch)
                # fitlog.add_metric({f'{comment}test': {'Wa': wa}}, step=epoch)
                if te_acc == max(te_acc_lis):
                    fitlog.add_best_metric(max(te_acc_lis), comment)
                if tmp_uar > best_uar:
                    best_uar = tmp_uar
                    fitlog.add_best_metric(best_uar, comment + 'UAR')
                if tmp_f1 > best_f1:
                    best_f1 = tmp_f1
                    fitlog.add_best_metric(best_f1, comment + 'f1')
                if tmp_wa > best_wa:
                    best_wa = tmp_wa
                    fitlog.add_best_metric(best_wa, comment + 'wa')
    return (te_acc_lis, tr_acc_lis, te_lo_lis, tr_lo_lis), \
        (max_labels_lis, max_te_pred_lis, te_label_lis, te_pred_lis, tr_label_lis, tr_pred_lis), \
        (te_lo / te_num, tr_lo / tr_num), tr_acc, (best_wa, best_f1, best_uar)


def else_work(tr_acc, dropout, batch_size, lo_l2, lr, optim_w, epoch, model, each_data,
              te_acc_lis, tr_acc_lis, tr_lo_lis, te_lo_lis, max_labels_lis, max_pred_lis,
              last_tr_lo, last_te_lo, te_label_lis, te_pred_lis, tr_label_lis, tr_pred_lis, args,
              fit_frame=300, plot_mean_matrix=False, trans='',
              step_size=10, gamma=1, mean_len=20, plot_acc=True, plot_lo=True, save_pt=True,
              data_augmentation=True, noise_rate=0.0005, total_sample=None, augment_func=None, seed=None):
    print("dropout = ", dropout, 'l2 = ', lo_l2, 'lr = ', lr, 'gamma=', gamma, 'step_size=', step_size)
    print("\nmax WA ：", max(te_acc_lis))

    # calculate max mean
    mean_acc, star = max_sum_avg(te_acc_lis, mean_len=mean_len)
    print('max_mean accuracy : ', mean_acc, 'star =', star)

    # get path
    model_name = type(model).__name__
    formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = get_path(model_name=model_name, each_data=each_data, formatted_time=formatted_time)
    save_pt_path, logger_path, acc_path, mat_path, lo_path, param_path, report_path, detail_path = path

    # plot loss、 accuracy、 matrix
    if plot_acc:
        Plot.plot_accuracy(acc_list=te_acc_lis, acc_path=acc_path, tr_acc_lis=tr_acc_lis)
    if plot_lo:
        Plot.plot_loss(tr_lo_lis=tr_lo_lis, te_lo_lis=te_lo_lis, lo_path=lo_path)

    type_list = Plot.get_type_lis(each_data=each_data)
    num_class = len(type_list)
    if plot_mean_matrix:
        conf_matrix = get_conf_matrix(labels=te_label_lis, predictions=te_pred_lis, star=star, mean_len=mean_len,
                                      plot_mean_matrix=plot_mean_matrix, num_class=num_class)
    else:
        conf_matrix = get_conf_matrix(max_labels_lis, max_pred_lis, plot_mean_matrix, num_class=num_class)
    uar = Plot.plot_matrix(conf_matrix=conf_matrix, sav_path=mat_path, type_list=type_list)

    y_pred = []
    y_true = []
    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         y_true.extend([i + 1] * conf_matrix[i][j])
    #         y_pred.extend([j + 1] * conf_matrix[i][j])
    report = classification_report(y_true, y_pred, target_names=type_list, digits=4)
    print(report)

    #  write param
    total_param = write_param(param_path=param_path, model=model)

    # logger
    with open(logger_path, 'a+') as f:
        if not trans == '':
            print('\n', trans, file=f, end='')
        f.write('\n')
        f.write(formatted_time)
        print(f'dataset:{each_data}', f'total_param_num={total_param}', file=f)
        print(f'max mean acc:{mean_acc:.3f}% star:{star} uar {uar:.3f} te_max_acc{max(te_acc_lis)} tr_acc:{tr_acc:.3f}',
              file=f)
        print(f'last_tr_lo: {last_tr_lo:.4g}  te_lo:{last_te_lo:.4g}', file=f)
        print("dropout=", dropout, ' batch_size=', batch_size, ' l2=', lo_l2, ' lr=', lr, 'step_size=', step_size,
              'gamma=', gamma, ' optim_w=', 'fit_frame=', fit_frame,
              optim_w, ' epoch=', epoch, file=f)
        if data_augmentation:
            print('noise_rate', noise_rate, 'total_sample', total_sample, 'augmentation_func', augment_func, file=f)
        print('seed', seed, file=f)

    with open(report_path, 'a+') as f:
        print(formatted_time, '\n', report, file=f)

    te_label_lis = trans_to_lis(te_label_lis)
    te_pred_lis = trans_to_lis(te_pred_lis)
    tr_label_lis = trans_to_lis(tr_label_lis)
    tr_pred_lis = trans_to_lis(tr_pred_lis)
    with open(detail_path, 'a+') as f:
        print('te_acc_lis,', te_acc_lis, file=f)
        print("tr_acc_lis,", tr_acc_lis, file=f)
        print("te_lo_lis,", te_lo_lis, file=f)
        print('tr_lo_lis,', tr_lo_lis, file=f)
        print('te_pred_lis,', te_pred_lis, file=f)
        print('te_label_lis,', te_label_lis, file=f)
        print('tr_pred_lis,', tr_pred_lis, file=f)
        print('tr_label_lis,', tr_label_lis, file=f)
    if save_pt:
        torch.save(model, save_pt_path)


def get_path(model_name, each_data, formatted_time):
    save_pt_path = '../trained_model/' + model_name + each_data + '.pt'
    logger_path = '../logger/' + model_name + each_data + 'logger.txt'

    picture_path = f'../ran_picture/{str(formatted_time)}/'
    os.mkdir(picture_path)
    acc_path = picture_path + model_name + each_data + ' accuracy'
    mat_path = picture_path + model_name + each_data + 'matrix'
    lo_path = picture_path + model_name + each_data + ' loss'

    detail_data_path = f'../each_epoch_data/{str(formatted_time)}.txt'
    report_path = '../report/' + model_name + each_data + '.txt'
    param_path = '../param/' + model_name + str(formatted_time) + 'param.txt'
    return save_pt_path, logger_path, acc_path, mat_path, lo_path, param_path, report_path, detail_data_path


def trans_to_lis(inputs):
    numpy_array = [torch.hstack(i).cpu().numpy() for i in inputs]
    out = []
    for array in numpy_array:
        out.append([i for i in array])
    return out


def write_param(param_path, model):
    total_param = sum([param.nelement() for param in model.parameters()])
    mod = 'w+' if os.path.exists(param_path) else 'a+'
    with open(param_path, mod) as f:
        print(f'total_param {total_param}\n', file=f)
        for name, param in model.named_parameters():
            print(name, param.numel(), param.shape, file=f)
    print(f'total_param {total_param}')
    return total_param


def get_conf_matrix(labels, predictions, plot_mean_matrix=False, star=0, mean_len=0, num_class=7):
    if plot_mean_matrix:
        labels = [torch.hstack(i).cpu().numpy() for i in labels]
        predictions = [torch.hstack(i).cpu().numpy() for i in predictions]
    else:
        labels = torch.hstack(labels).cpu().numpy()
        predictions = torch.hstack(predictions).cpu().numpy()
    if plot_mean_matrix:
        conf_matrix = []
        for label, pred in zip(labels[star:star + mean_len], predictions[star:star + mean_len]):
            conf_matrix_ = np.bincount(label * num_class + pred, minlength=num_class ** 2).reshape(
                (num_class, num_class))
            conf_matrix.append(conf_matrix_)
        conf_matrix = np.sum(conf_matrix, axis=0)
    else:
        conf_matrix = np.bincount(labels * num_class + predictions, minlength=num_class ** 2).reshape(
            (num_class, num_class))
    return conf_matrix


if __name__ == '__main__':
    pass
    # TRAIN_LOADER, TEST_LOADER, tr_num, te_num, mfcc_column = get_loader(batch_size=256,
    #                                                                     fit_frame=400,
    #                                                                     each_data='EMODB',
    #                                                                     shuffle=True,
    #                                                                     each_feature="log-mel",
    #                                                                     output_size=7,
    #                                                                     tr_set_shuffle=True, )


class Plot:
    @staticmethod
    def plot_accuracy(acc_list, acc_path, tr_acc_lis=None):

        plt.figure()
        plt.plot(range(1, len(acc_list) + 1), acc_list)

        if tr_acc_lis:
            if len(tr_acc_lis) == len(acc_list):
                tr_acc_lis = tr_acc_lis[::5]
            step = len(acc_list) // len(tr_acc_lis)
            plt.scatter(range(step, len(acc_list) + 1, step), tr_acc_lis)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend(['test_acc', 'train_acc'] if tr_acc_lis else ['test_acc'])

        max_acc = max(acc_list)
        max_idx = acc_list.index(max_acc) + 1

        # 标出最大值点
        # plt.plot(max_idx, max_acc, 'ro')  # 用红色圆圈标出最大值点
        plt.axhline(y=max_acc, color='gray', linestyle='--')  # 用虚线标出最大值水平线
        plt.axvline(x=max_idx, color='gray', linestyle='--')  # 用虚线标出最大值垂直线
        plt.text(max_idx + 0.1, max_acc + 0.02, f'({max_idx},{max_acc:.3f})', fontsize=10)  # 在最大值点标出坐标

        plt.savefig(acc_path)
        plt.show()

    @staticmethod
    def plot_loss(tr_lo_lis, te_lo_lis, lo_path):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(range(1, len(tr_lo_lis) + 1), tr_lo_lis)
        plt.plot(range(1, len(te_lo_lis) + 1), te_lo_lis)
        plt.legend(['train_loss', 'test_loss'])
        plt.savefig(lo_path)
        plt.show()

    #  绘制混淆矩阵
    @staticmethod
    def plot_matrix(conf_matrix, type_list, sav_path=''):

        num_class = len(type_list)
        conf_matrix = conf_matrix.astype(float)
        recall_list = []
        for i in range(num_class):
            total = np.sum(conf_matrix[i, :])
            if not total == 0:
                recall_list.append(conf_matrix[i, i] / total * 100)
                conf_matrix[i] = conf_matrix[i, :] / total * 100.0
            else:
                recall_list.append(0)

        fig, ax = plt.subplots()
        im = ax.imshow(conf_matrix, cmap='Blues')

        for i in range(num_class):
            for j in range(num_class):
                text = '{:.2f}'.format(conf_matrix[i, j])
                ax.text(j, i, text, ha='center', va='center')

        uar = sum(recall_list) / len(recall_list)
        # 设置坐标轴标签和标题
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        # 设置坐标轴标签的位置和文本
        ax.set_xticks(np.arange(conf_matrix.shape[1]))
        ax.set_yticks(np.arange(conf_matrix.shape[0]))
        ax.set_xticklabels(type_list)
        ax.set_yticklabels(type_list)
        fig.colorbar(im)  # 添加颜色条
        plt.tight_layout()  # 自动调整子图布局
        plt.xlabel(f'uar:{uar:.3f}')
        plt.savefig(sav_path)
        plt.show()
        print('UA:', uar)
        return uar

    @staticmethod
    def get_type_lis(each_data='EMODB'):
        if each_data == "EMODB":
            type_list = ['angry', 'boredom', 'disgust', 'fear', 'happy', 'sad', 'neutral']
        elif "IEMOCAP" in each_data:
            type_list = ['angry', 'neutral', 'happy', 'sad']
        else:
            raise NotImplementedError()
        return type_list


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        attended_output, _ = self.multihead_attention(inputs, inputs, inputs)
        output = self.layer_norm(inputs + attended_output)
        return output


# additive attention
class ATTLayer(nn.Module):
    def __init__(self, n_hidden, att_size):
        super(ATTLayer, self).__init__()
        self.attention_w = nn.Parameter(torch.randn(n_hidden, att_size) * torch.tensor(0.1),
                                        requires_grad=True)  # [n_hidden, att_size][64, 128]
        self.attention_u = nn.Parameter(torch.randn(att_size, 1) * torch.tensor(0.1), requires_grad=True)  # [128, 1]
        self.attention_b = nn.Parameter(torch.ones(att_size) * torch.tensor(0.1), requires_grad=True)  # [128]

    def forward(self, inputs):  # [batch, seq_len, n_hidden] [256, 19, 64]
        u_list = []
        seq_size = inputs.shape[1]
        # n_hidden = inputs.shape[2]
        for t in range(seq_size):
            u_t = torch.tanh(torch.matmul(inputs[:, t, :], self.attention_w + self.attention_b))
            u = torch.matmul(u_t, self.attention_u)  # [256, 1]
            u_list.append(u)
        logit = torch.cat(u_list, dim=1)  # [256, 19]
        weights = torch.softmax(logit, dim=1)
        att_out = torch.sum(inputs * weights.reshape(-1, seq_size, 1), dim=1)
        return att_out


def att_layer(inputs, att_size, n_hidden, seq_size, device):
    attention_w = nn.Parameter(torch.randn(n_hidden, att_size) * torch.tensor(0.1), requires_grad=True).to(device)
    attention_u = nn.Parameter(torch.randn(att_size, 1) * torch.tensor(0.1), requires_grad=True).to(device)
    attention_b = nn.Parameter(torch.ones(att_size) * torch.tensor(0.1), requires_grad=True).to(device)
    u_list = []
    # seq_size = inputs.size(1)
    # n_hidden = inputs.size(2)
    for t in range(seq_size):
        u_t = torch.tanh(torch.matmul(inputs[:, t, :], attention_w + attention_b))
        u = torch.matmul(u_t, attention_u)
        u_list.append(u)
    logit = torch.cat(u_list, dim=1)
    weights = torch.softmax(logit, dim=1)
    att_out = torch.sum(inputs * weights.reshape(-1, seq_size, 1), dim=1)
    return att_out


def const2device(parameters, device=None):
    par_list = []
    for par in parameters:
        par_list.append(torch.tensor(par).to(device).detach())
    return par_list


def to_device(tensor, use_gpu=False, device=None):
    if use_gpu:
        tensor = tensor.to(device)
    return tensor


def make_tensors(data, labels, use_gpu=True, device=None):
    data = data.float()
    labels = labels.long()
    if device.type == 'cuda':
        use_gpu = True
    return to_device(data, use_gpu, device), to_device(labels, use_gpu, device)


class L2Regularization(nn.Module):
    def __init__(self, weight_decay):
        super().__init__()
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        regularization_loss = 0
        for param in self.criterion.parameters():
            regularization_loss += torch.norm(param, 2)
        regularization_loss *= self.weight_decay
        return regularization_loss + loss


def max_sum_avg(lis, mean_len=20):
    if len(lis) < mean_len:
        return sum(lis) / len(lis), 0

    window_sum = sum(lis[:mean_len])
    max_sum = window_sum
    star = 0
    for i in range(mean_len, len(lis)):
        window_sum += lis[i] - lis[i - mean_len]
        if window_sum > max_sum:
            max_sum = window_sum
            star = i - mean_len
    return max_sum / mean_len, star


class SEBlock(nn.Module):
    def __init__(self, in_channel=512, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel // reduction, kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel // reduction, out_channels=in_channel, kernel_size=(1, 1), bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x : [8, 512, 19, 1]
        h = x
        h = self.avg_pool(h)  # h : [8, 512, 1, 1]
        h = self.excitation(h)
        return x * h


def fitlog_decorator(func):
    import fitlog

    def wrapper(*args, **kwargs):
        fitlog.set_log_dir('logs/')
        fitlog.add_hyper_in_file(__file__)
        result = func(*args, **kwargs)
        fitlog.finish()
        return result

    return wrapper()


class LogMelSpectrogram(object):
    """Returns the log-mel spectrogram of the resampled audio clip.

    Args:
        sample_rate (int, optional): Sample rate to which the audio clip
            will be resampled. Default: 16000
        audio_length (int, optional): Length in seconds of the audio
            clip, if the clip is longer it will be cut to this
            value, if it is shorter it will be padded with zeros.
            Default: 8
        n_fft (int): Size of FFT for the spectrogram. Default: 2048
        hop_length (int, optional): Length of hop between STFT windows.
            Default: 512
    """

    def __init__(self, sample_rate, audio_length, n_fft, hop_length):
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, waveform):
        waveform = Resample(new_freq=self.sample_rate)(waveform)

        # cut or pad audio clip
        waveform.resize_(self.sample_rate * self.audio_length)

        mel_spec = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )(waveform)

        # since we pad the audio clip the mel-spectrogram might have
        # nan values, compromising the loss computation
        mel_spec = torch.nan_to_num(mel_spec, 1e-5)

        log_mel_spec = AmplitudeToDB()(mel_spec)

        return log_mel_spec
