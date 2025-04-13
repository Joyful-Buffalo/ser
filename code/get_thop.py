import torch
from ptflops import get_model_complexity_info
from thop import profile
from torchstat import stat
from torchsummary import summary

from inception.full19 import FullInception19
from inception.full20 import Full20
from inception.full_inception import FullInception
from inception.fullinception11 import FullInception11
from inception.fullinception13 import FullInception13
from inception.fullinception14 import FullInception14
from inception.fullinception17 import FullInception17
from inception.fullinception2 import FullInception2
from inception.fullinception3 import FullInception3
from inception.fullinception4 import FullInception4
from inception.fullinception5 import FullInception5
from inception.fullinception5_1 import FullInception5_1
from inception.fullinception8 import FullInception8
from inception.fullinception9 import FullInception9
from my_channel import MyCnn
from vovnet.big_net import BigNet
from vovnet.icassp import Part2
from vovnet.vovnet import VovNet


def profile_func(net):
    net = net.to(torch.device('cuda'))
    # x = torch.randn((1, frame * column)).to(torch.device('cuda'))

    x = torch.randn((1, 1, frame, column)).to(torch.device('cuda'))
    flops, params = profile(net, inputs=(x,))
    print(type(net).__name__, f'MFLOPs{flops / 1e6}M', "params:{}M".format(params / 1e6), end=' ')
    print("PMU:{:.3f}".format(torch.cuda.max_memory_allocated(torch.device('cuda')) / 1024), 'kb', end=' ')


def ptf_func(net):
    flops, params = get_model_complexity_info(net, (1, frame * column), as_strings=True, print_per_layer_stat=True)
    print("%s %s" % (flops, params))


def summary_func(net):  # 不支持LSTM计算
    net.to(torch.device('cuda'))
    input_size = (1, frame * column)
    summary(net, input_size)


def stat_func(net):  # len(input_size)必须为3且不支持lstm计算
    net.to(torch.device('cuda'))
    shape = torch.Tensor((1, 1, frame * column)).to(torch.device('cuda'))
    stat(net, (1, 1, frame * column,))


def get_module(cnn_name):
    if cnn_name == 'vov' or cnn_name == 'hh' or cnn_name == 'inception1':
        return MyCnn(frame=frame, column=column, attention_size=128, n_hidden=64, dropout=0.5, output_size=4,
                     cnn_block=cnn_name)
    elif cnn_name == 'only_vov':
        return VovNet(dropout=0.5)
    elif cnn_name == 'fullinception':
        return FullInception(frame=frame, column=column, output_size=4)
    elif cnn_name == 'fullinception2':
        return FullInception2(frame=frame, column=column, output_size=4)
    elif cnn_name == 'fullinception3':
        return FullInception3(frame=frame, column=column, output_size=4)
    elif cnn_name == 'fullinception4':
        return FullInception4(frame=frame, column=column, output_size=4)
    elif cnn_name == 'fullinception5':
        return FullInception5(frame=frame, column=column, output_size=4)
    elif cnn_name == 'fullinception51':
        return FullInception5_1(frame=frame, column=column, output_size=4)
    elif cnn_name == 'fullinception8':
        return FullInception8(frame=frame, column=column, output_size=4)
    elif cnn_name == 'fullinception9':
        return FullInception9(frame=frame, column=column, output_size=4)
    elif cnn_name == 'fullinception11':
        return FullInception11(frame=frame, column=column, output_size=4)
    elif cnn_name == 'fullinception13':
        return FullInception13(frame=frame, column=column, output_size=4)
    elif cnn_name == 'fullinception14':
        return FullInception14(frame=frame, column=column, output_size=4)
    elif cnn_name == 'fullinception17':
        return FullInception17(frame=frame, column=column, output_size=4)
    elif cnn_name == 'full19':
        return FullInception19(frame=frame, column=column, output_size=4)
    elif cnn_name == 'full20':
        return Full20(frame=frame, column=column, output_size=4)
    elif cnn_name == 'big':
        return BigNet()
    elif cnn_name == 'icassp':
        return Part2(output=4)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    frame = 433
    column = 40
    batch = 32
    cnn = ['full19', 'fullinception5', 'full20', ]
    # cnn = ['big']
    for cnn_name in cnn:
        my_Cnn = get_module(cnn_name)
        print('\n', cnn_name)
        profile_func(my_Cnn)
        torch.cuda.empty_cache()
    # cnn_name = 'fullinception51'
    # cnn_name = 'fullinception5'
    # cnn_name = 'icassp'
    # cnn_name = 'full20'
    # my_Cnn = get_module(cnn_name)
    # print('\n', cnn_name)
    # profile_func(my_Cnn)
    # summary_func(my_Cnn)
    # ptf_func(my_Cnn)

    # cnn_name = 'full20'
    # cnn_ = get_module(cnn_name)
    # summary_func(cnn_)
    # print('\n\n\n\n')
    # ptf_func(cnn_)

    torch.cuda.empty_cache()
    # stat_func(my_Cnn)
    # if frame == 700:
    #     att = 726528 / 1e6
    # elif frame == 300:
    #     att = 313248 / 1e6
    # print('att_layer Mflops', att)
