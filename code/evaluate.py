import os
import os.path
import shutil

import torch

from EMODB.code.my_tool import make_tensors, L2Regularization
from evaluate_tool import get_loader


def test(model, loader, device, criterion):
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

    return correct / total, test_loss, pred_list, label_list


def to_emo(label_list):
    label_dict = dict()

    for idx, emo in enumerate(emo_list, 0):
        label_dict[idx] = emo
    pred_list = [label_dict[i] for i in label_list]
    return pred_list


def sav_wav():
    with open('emoclass.txt', 'r') as f:
        data = f.readlines()

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
        # elif emo == emo_list[4]:
        #     class_list[4].append(path)
        # elif emo == emo_list[5]:
        #     class_list[5].append(path)
        # elif emo == emo_list[6]:
        #     class_list[6].append(path)
        else:
            raise NotImplementedError()

    print(class_list)

    master_dir = 'D:\\SerPorjects\\EMODB\\evaled_data\\'
    if not os.path.exists(master_dir):
        os.mkdir(master_dir)

    for idx, i in enumerate(emo_list):
        emo_dir = master_dir + i
        if not os.path.exists(emo_dir):
            os.mkdir(emo_dir)
        for emo_wav in class_list[idx]:
            shutil.copy(emo_wav + '.MP3', emo_dir)


def main():
    device = torch.device('cuda')
    criterion = L2Regularization(weight_decay=0.00001)
    pt_dir = '../trained_model/'
    pt_path = pt_dir + os.listdir(pt_dir)[-1]
    model = torch.load(pt_path, map_location=device)
    test_loader, mfcc_column, files = get_loader(batch_size=128,
                                                 each_data='EVALUATE',
                                                 each_feature='mfcc',
                                                 output_size=len(emo_list),
                                                 fit_frame=433)
    acc, _, pred, label = test(model, test_loader, device=device, criterion=criterion)
    tmp = torch.tensor([], device='cuda')
    for i in pred:
        tmp = torch.cat((tmp, i), dim=0)
    pred_list = to_emo(tmp.tolist())
    print(pred_list)
    print(pred_list.count('1') / len(pred_list))
    # f = open('emoclass.txt', 'w')
    # for emo, file in zip(pred_list, files):
    #     print(file, emo, file=f)
    #     print(file, emo)
    # f.close()
    # sav_wav()


if __name__ == '__main__':
    # emo_list = ['angry', 'boredom', 'disgust', 'fear', 'happy', 'sad', 'neutral']
    emo_list = ['1', '2', '3']
    main()
