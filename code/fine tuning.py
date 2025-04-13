import os.path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_dnn import MyDnn
from my_tool import EmodbSet, plot_loss_acc, plot_matrix, max_sum_avg, make_tensors, L2Regularization


def train(model, optimizer, criterion, pbar, loader, mfcc_column=13):
    total_loss = 0
    for i, (combines, labels) in enumerate(loader, 1):
        inputs, target = make_tensors(combines, labels, use_gpu=USE_GPU, device=device)
        output = model(inputs, frame=Fit_frame, column=mfcc_column)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.update(len(combines))
    return total_loss


def test(model, criterion, pbar, loader, fit_frame=300, mfcc_column=13, tr_num=107):
    total_loss = 0
    correct = 0
    total = len(loader)
    prediction = []
    labels = []
    for i, (combines, labels) in enumerate(loader, 1):
        inputs, target = make_tensors(combines, labels, use_gpu=USE_GPU, device=device)
        output = model(inputs, frame=fit_frame, column=mfcc_column)
        loss = criterion(output, target)
        total_loss += loss.item()
        _, predicted = torch.max(output.data, dim=1)  # _:每一行的最大值   predicted :每一行最大值所在列
        correct += predicted.eq(target.view_as(predicted)).sum().item()
        labels.append(target)
        prediction.append(predicted)
        if total == tr_num:
            pbar.update(len(combines))
    return 100 * correct / total, total_loss, labels, prediction


def main(path, epoch=50, each_data='total'):
    train_set = EmodbSet(is_train_set=True, each_data=each_data, fit_frame=Fit_frame, shuffle=False, train_rate=0.8)
    test_set = EmodbSet(is_train_set=False, each_data=each_data, fit_frame=Fit_frame)
    mfcc_column = EmodbSet.get_mfcc_column(train_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = MyDnn()
    model_weight = torch.load(path, map_location=torch.device('cpu')).to(device)
    model.to(device)
    model.load_state_dict(model_weight['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    for param in model.brf_svm.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=optim_w)
    criterion = L2Regularization(weight_decay=lo_l2)
    tr_lo_lis = []
    te_lo_lis = []
    acc_lis = []
    tr_num, te_num = len(train_set), len(test_set)
    with tqdm(total=epoch * (tr_num + te_num)) as pbar:
        for epo in range(1, epoch + 1):
            tr_lo = train(model, optimizer, criterion, pbar, loader=train_loader)
            acc, te_lo, labels_lis, pred_lis = test(model, criterion=criterion, pbar=pbar, loader=test_loader)
            acc_lis.append(acc)
            tr_lo_lis.append(tr_lo / tr_num)
            te_lo_lis.append(te_lo / te_num)
            if epo % 10 == 0:
                print(f'[epoch {epo}] tr_lo: {tr_lo / tr_num} te_lo:{te_num}  acc: {acc}')
            if epo % 20 == 0:
                acc, t_lo, tr_labels_list, tr_prediction_list = test(model=model, pbar=pbar, loader=train_loader,
                                                                     criterion=criterion, mfcc_column=mfcc_column)
                print('Accuracy on Train set: %.2f %%' % acc)
            if epoch > 0 and acc == max(acc_lis):
                max_label = labels_lis
                max_prediction = pred_lis

    model_name = type(model).__name__

    # calculate max mean
    mean_len = 20
    mean, star = max_sum_avg(acc_lis, mean_len=mean_len)
    print(f'len {mean_len} mean accuracy :{mean}, star :{star}')

    # empty cuda memery
    torch.cuda.empty_cache()

    # save weight
    torch.save(model, 'trained_model/' + model_name + 'fine tuning')

    # plot loss and accuracy
    acc_path = os.path.join('fine tuning picture', model_name + 'acc')
    loss_path = os.path.join('fine tuning picture', model_name + 'loss')
    plot_loss_acc(acc_list=acc_lis, tr_lo_lis=tr_lo_lis, te_lo_lis=te_lo_lis, acc_path=acc_path,
                  lo_path=loss_path)

    # plot matrix
    labels_list = torch.hstack(max_label).cpu().numpy()
    prediction_list = torch.hstack(max_prediction).cpu().numpy()
    sav_path = os.path.join('ran_picture', model_name + 'matrix')
    plot_matrix(labels=labels_list, prediction=prediction_list, sav_path=sav_path)


if __name__ == '__main__':
    pt_path = 'trained_model/MyDnn.pt'
    USE_GPU = True
    if USE_GPU:
        device = torch.device('cuda')
    train_epoch = 50
    Fit_frame = 300
    BATCH_SIZE = 128
    optim_w = 0.0
    lo_l2 = 0
    lr = 0.0005

    main(pt_path, epoch=train_epoch)
