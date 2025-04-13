import argparse
import concurrent
import datetime
import multiprocessing
import os
import os.path
import time

import fitlog
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR

from inception.full19 import FullInception19
from inception.full191 import FullInception191
from inception.full20 import Full20
from inception.full201 import Full201
from inception.fullinception5 import FullInception5
from inception.fullinception5_1 import FullInception5_1
from my_tool import (get_loader,
                     L2Regularization,
                     main)
from vovnet.icassp import Part2
from zhong import Zhong


def batch_main(train_loader=None, test_loader=None, batch_lis=None, fit_frame=None, args=None,
               mfcc_column=43, comment=None, dropout=None, tr_num=None, te_num=None, test_speaker=None,
               device=None, criterion=None, ):
    # fitlog.debug()
    # log_dir = f'logs/{test_speaker}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    # fitlog.set_log_dir(log_dir)
    fitlog.set_log_dir(f'logs/')
    fitlog.add_hyper_in_file(__file__)
    fitlog.add_hyper(args)
    fitlog.add_other(str(mfcc_column), 'mfcc')
    fitlog.add_other(str(args.fit_frame), 'frame')
    fitlog.add_other(str(args.batch_size), 'batch_size')
    fitlog.add_other(str(args.lr), 'lr')
    fitlog.add_other(str(args.lo_l2), 'l2')
    fitlog.add_other(args.each_data, 'dataset')
    fitlog.add_other(test_speaker, 'speaker')
    if batch_lis is None:
        batch_lis = [
            'fullinception5',
            'fullinception51',
            'full19',
            'full191',
            'full20',
            'full201',
            # 'zhong'
        ]
    # train_loader = TRAIN_LOADER if train_loader is None else train_loader
    # test_loader = TEST_LOADER if test_loader is None else test_loader
    fit_frame = args.fit_frame if fit_frame is None else fit_frame
    dropout = args.dropout if dropout is None else dropout
    tr_acc_lis = []
    te_acc_lis = []
    tr_loss_lis = []
    te_loss_lis = []
    best_lis = []
    for cnn_name in batch_lis:
        comment = cnn_name
        if cnn_name == 'fullinception5':
            batch_model = FullInception5(frame=fit_frame, column=mfcc_column, output_size=args.output_size,
                                         dropout=dropout)
        elif cnn_name == 'fullinception51':
            batch_model = FullInception5_1(fit_frame, mfcc_column, args.output_size, dropout=dropout)
        elif cnn_name == 'full19':
            batch_model = FullInception19(frame=fit_frame, column=mfcc_column, output_size=args.output_size,
                                          dropout=dropout)
        elif cnn_name == 'full191':
            batch_model = FullInception191(frame=fit_frame, column=mfcc_column, output_size=args.output_size,
                                           dropout=dropout)
        elif cnn_name == 'full20':
            batch_model = Full20(frame=fit_frame, column=mfcc_column, output_size=args.output_size, dropout=dropout)
        elif cnn_name == 'full201':
            batch_model = Full201(frame=fit_frame, column=mfcc_column, output_size=args.output_size, dropout=dropout)
        elif cnn_name == 'icassp':
            batch_model = Part2(output=args.output_size, frame=fit_frame, column=mfcc_column)
        elif cnn_name == "zhong":
            batch_model = Zhong(frame=fit_frame, column=mfcc_column, output_size=args.output_size)
        else:
            raise NotImplementedError
        optimizer = torch.optim.Adam(batch_model.parameters(), lr=args.lr, weight_decay=args.optim_w)
        scheduler = StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
        batch_model.to(device)
        lis, max_lis, lo, tr_acc, best = main(scheduler=scheduler, optimizer=optimizer,
                                              test_loader=test_loader, train_loader=train_loader, device=device,
                                              tr_num=tr_num, te_num=te_num, epochs=args.epochs, model=batch_model,
                                              criterion=criterion, comment=comment, fitlog=fitlog, args=args)
        # torch.cuda.empty_cache()
        te_acc, tr_acc, te_lo, tr_lo = lis
        tr_acc_lis.append(tr_acc)
        te_acc_lis.append(te_acc)
        tr_loss_lis.append(tr_lo)
        te_loss_lis.append(te_lo)
        best_lis.append(best)
        # formatted_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # model_name = type(batch_model).__name__
        # save_pt_path = '../trained_model/' + model_name + args.each_data + formatted_time + '.pt'
        # torch.save(batch_model, save_pt_path)
    formatted_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = '../ran_picture/' + formatted_time + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    plot(legend=batch_lis, te_acc_lis=te_acc_lis, tr_acc_lis=tr_acc_lis, path=path + 'acc', args=args)
    plot(legend=batch_lis, te_acc_lis=te_loss_lis, tr_acc_lis=tr_loss_lis, path=path + 'loss', args=args)

    fitlog.finish()
    return best_lis


def plot(legend, te_acc_lis, tr_acc_lis, path, args=None):
    plt.figure()
    for i in range(len(legend)):
        plt.plot([i for i in range(args.epochs)], te_acc_lis[i])
    plt.legend([str(i) for i in legend])
    plt.title('test' + args.each_data)
    plt.savefig(path + 'test')
    plt.show()
    plt.figure()
    for i in range(len(legend)):
        plt.plot([i for i in range(args.epochs)], tr_acc_lis[i])
    plt.legend([str(i) for i in legend])
    plt.title('Training')
    plt.savefig(path + 'Training' + args.each_data)
    plt.show()


def get_device():
    # pynvml.nvmlInit()
    # available_devices = [i for i in range(torch.cuda.device_count())]
    # if not available_devices:
    #     return None  # No CUDA devices available
    # useful = 0
    # print('waiting device  --------------------------------------------------')
    # while True:
    #     for idx in available_devices:
    #         handle = pynvml.nvmlDeviceGetHandleByIndex(idx)  # 指定GPU的id
    #         meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #         if meminfo.free / (1024 ** 3) > useful:
    #             useful = meminfo.free / (1024 ** 3)
    #     if useful > 5:
    #         break
    #     time.sleep(17)
    #     # Choose the device with the most available memory
    # best_device = max(available_devices,
    #                   key=lambda device: pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device)).free)
    #
    # if best_device is not None:
    #     return torch.device(f'cuda:{best_device}')
    # else:
    #     return torch.device('cpu')
    return torch.device('cuda:0')


def speaker_main(args, test_speaker, device, criterion):
    TRAIN_LOADER, TEST_LOADER, tr_num, te_num, mfcc_column = get_loader(batch_size=args.batch_size,
                                                                        fit_frame=args.fit_frame,
                                                                        each_data=args.each_data,
                                                                        shuffle=True,
                                                                        each_feature="mfcc",
                                                                        output_size=args.output_size,
                                                                        tr_set_shuffle=True,
                                                                        data_augmentation=args.data_augmentation,
                                                                        noise_rate=args.noise_rate,
                                                                        # augment_func=augmentation_func,
                                                                        # seed=seed,
                                                                        test_speaker=test_speaker
                                                                        )
    device = get_device()
    print('Training sample', tr_num, 'test sample', te_num, 'test_speaker', test_speaker)

    best = batch_main(test_loader=TEST_LOADER, train_loader=TRAIN_LOADER, mfcc_column=mfcc_column,
                      te_num=te_num, tr_num=tr_num, args=args, test_speaker=test_speaker,
                      device=device, criterion=criterion)
    torch.cuda.empty_cache()
    # best_wa, best_f1, best_uar = best
    # best_list[0].append(best_wa)
    # best_list[1].append(best_f1)
    # best_list[2].append(best_uar)
    # print('best_wa: ', best_wa, 'best_f1 : ', best_f1, 'best_uar: ', best_uar)
    # sys.exit()
    # return best


if __name__ == '__main__':
    each_data = 'EMODB'
    # each_data = 'IEMOCAP'
    # each_data = 'IMPRO_IEMOCAP'
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit_frame', type=int, default=185)  # 185 433
    # parser.add_argument('--fit_frame', type=int, default=433)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--att_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--gamma', type=float, default=0.86)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lo_l2', type=float, default=0.00001)
    parser.add_argument('--optim_w', type=float, default=0.)
    parser.add_argument('--data_augmentation', type=bool, default=False)
    parser.add_argument('--noise_rate', type=float, default=0.005)
    parser.add_argument('--mfcc_column', type=int, default=40)
    parser.add_argument('--each_data', type=str, default=each_data)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_size', type=int, default=4 if "IEMOCAP" in each_data else 7)
    # parser.add_argument('--frame_len', type=float, default=0.)
    args = parser.parse_args()

    seed = None
    # NOISE_RATE = [0.005, 1]
    # augmentation_func = ['xy+x', 'x+y']

    augmentation_func = 'xy+x'
    # augmentation_func = 'x+y'
    # augmentation_func = 'sqrt(x)*y+x'
    # augmentation_func = 'random'

    trans = ''

    OUTPUT_SIZE = 4 if "IEMOCAP" in args.each_data else 7
    DEVICE = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    CRITERION = torch.nn.CrossEntropyLoss() if args.lo_l2 == 0 else L2Regularization(weight_decay=args.lo_l2)
    if args.each_data == 'EMODB':
        speakers = [
            '03', '08', '09', '10', '11', '12',
            '13', '14', '15', '16',
            # '03', '08', '09', '10', '11', '12',
            # '13', '14', '15', '16','03', '08', '09', '10', '11', '12',
            # '13', '14', '15', '16',
            # '08', '09', '12', '14'
        ]
    else:
        speakers = [
            '01F', '01M', '02F', '02M', '03F', '03M',
            '04F', '04M', '05F', '05M'
        ]
    # speakers = ['08']
    best_list = [[], [], []]
    processes = []
    multiprocessing.set_start_method('spawn')

    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = []
        for test_speaker in speakers:
            futures.append(executor.submit(speaker_main, args, test_speaker, DEVICE, CRITERION))
            time.sleep(720 if 'IEMOCAP' in each_data else 10)
        concurrent.futures.wait(futures)

    # speaker_main(args, '03', DEVICE, CRITERION)

    # results = [future.result() for future in futures]
    # num = len(results)
    # best_lis = []
    # for i, result in enumerate(results):
    #     best_lis.append(best_list)
    #     print(f"进程 {i} 返回值: {result}")
    # print(best_lis)
