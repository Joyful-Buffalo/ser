import os.path
import shutil

with open('emoclass.txt', 'r') as f:
    data = f.readlines()

emo_list = ['angry', 'boredom', 'disgust', 'fear', 'happy', 'sad', 'neutral']

class_list = [[], [], [], [], [], [], []]
for i in data:
    path, emo = i.split('MP3')
    path = path[:-1]
    emo = emo.strip()
    if emo == emo_list[0]:
        class_list[0].append(path)
    elif emo == emo_list[1]:
        class_list[1].append(path)
    elif emo == emo_list[2]:
        class_list[2].append(path)
    elif emo == emo_list[3]:
        class_list[3].append(path)
    elif emo == emo_list[4]:
        class_list[4].append(path)
    elif emo == emo_list[5]:
        class_list[5].append(path)
    elif emo == emo_list[6]:
        class_list[6].append(path)

print(class_list)

master_dir = 'D:\\SerPorjects\\EMODB\\evaled_data\\'
for idx, i in enumerate(emo_list):
    emo_dir = master_dir + i
    if not os.path.exists(emo_dir):
        os.mkdir(emo_dir)

    for emo_wav in class_list[idx]:
        shutil.copy(emo_wav+'.MP3', emo_dir)
