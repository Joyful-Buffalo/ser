import json
import os
import shutil


# shutil.rmtree('/home/pwy/lhc/HHpaper/dataset/IEMOCAP/wav')
def iemo_pre():
    with open('/home/pwy/lhc/HHpaper/dataset/IEMOCAP/labels.json') as f:
        data = json.load(f)
    mp = '/home/pwy/lhc/HHpaper/dataset/IEMOCAP/ori_wav'
    emo = ['anger', 'sad', 'neu', 'joy']
    for k in emo:
        for p in data[k]:
            op = p.replace('/wav', '/ori_wav')
            new_p = p.split('.')[0] + k[:3] + '.wav'
            shutil.copy(op, new_p)


def iemo_impro_pre():
    mp = '/home/pwy/lhc/HHpaper/dataset/IMPRO_IEMOCAP/wav'
    if os.path.exists(mp):
        shutil.rmtree(mp)
    os.makedirs(mp, exist_ok=True)
    with open('/home/pwy/lhc/HHpaper/dataset/IMPRO_IEMOCAP/labels.json') as f:
        data = json.load(f)

    emo = ['anger', 'sad', 'neu', 'joy']
    for k in emo:
        for p in data[k]:
            op = p.replace('/wav', '/ori_wav')
            new_p = p.split('.')[0] + k[:3] + '.wav'
            shutil.copy(op, new_p)


if __name__ == '__main__':
    iemo_impro_pre()
