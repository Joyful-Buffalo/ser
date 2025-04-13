import json
import os
import re
import shutil


def copy():  # 复制音频
    original_path = '/home/pwy/lhc/cnn-lstm/data/emodb/wav/'
    sav_dir = '/home/pwy/lhc/HHpaper/dataset/EMODB/wav/'
    for i in os.listdir(sav_dir):
        os.remove(sav_dir + i)

    print(len(os.listdir(sav_dir)))

    for i in os.listdir(original_path):
        i = original_path + i
        shutil.copy(i, sav_dir)
        print(i)

    print(len(os.listdir(sav_dir)))


def new_copy():  # 复制音频，并重命名文件
    sav_dir = '/home/pwy/lhc/HHpaper/dataset/SCRIPT_IEMOCAP/wav/'
    for i in os.listdir(sav_dir):
        os.remove(sav_dir + i)
    with open('/home/pwy/lhc/HHpaper/dataset/SCRIPT_IEMOCAP/labels.json', 'r') as f:
        data = json.load(f)

    for i in data.keys():
        count = 1
        for p in data[i]:
            p = p.replace('/wav/', '/ori_wav/')
            shutil.copy(p, sav_dir)
            new_p = sav_dir + p.split('/')[-1]
            os.rename(new_p, sav_dir + i + str(count) + '.wav')
            count += 1
            print(sav_dir + i + str(count))
    print(os.listdir(sav_dir))


def copy_metric_loss_best_data(from_p, to_p):
    with open(from_p, 'r') as f:
        data = f.readlines()
    with open(to_p, 'a+') as f:
        for i in data:
            print(i, end='', file=f)


def get_frame(dir_path):
    other_path = dir_path + 'other.log'
    with open(other_path, 'r') as f:
        data = f.readlines()
    pattern = r'"frame"\s*:\s*"(\d+)"'
    for data in data:
        match = re.search(pattern, data)
        if match:
            return match.group(1)


def judge(from_dir, to_dir):
    if get_frame(from_dir) == get_frame(to_dir):
        return True
    return False


def copy_data(from_dir=None, to_dir=None):
    from_dir = '/home/pwy/lhc/EMODB/code/logs/log_20230521_162305/'
    to_dir = '/home/pwy/lhc/EMODB/code/logs/log_20230521_153800/'
    if not judge(from_dir, to_dir):
        raise "frame not same"
    for path in os.listdir(from_dir):
        if 'loss' in path or 'best_metric' in path or 'metric' in path:
            from_p = from_dir + path
            to_p = to_dir + path
            copy_metric_loss_best_data(from_p, to_p)


copy_data()

# copy_metric_data(from_path, to_path)
# sav_dir = '/home/pwy/lhc/HHpaper/dataset/EMODB/wav/'
# print(os.listdir(sav_dir))
# new_copy()
# copy()
