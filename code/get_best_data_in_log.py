import re
from matplotlib import pyplot as plt

with open('logs/log_20230517_222342/metric.log', 'r') as f:
    logdata = f.read()


def get_best_data(compile):
    f1_list = re.findall(compile, logdata)
    f1_list = [eval(i) for i in f1_list]
    all_comment = re.findall(r'": {"(.*?)test"', logdata)
    comment = []
    for i in all_comment:
        if i not in comment:
            comment.append(i)
    one_exp_len = len(f1_list) // len(comment)
    # for i in range(len(comment)):
    #     plt.plot([i for i in range(one_exp_len)], f1_list[i * one_exp_len: (i + 1) * one_exp_len])
    # plt.legend(comment)
    # plt.title('test')
    # plt.show()
    return f1_list, one_exp_len, comment


compile_lis = [
    re.compile(r'test": {"Macro_F1": (\d+\.\d+)}'),
    re.compile(r'test": {"UAR": (\d+\.\d+)}'),
    re.compile(r'test": {"acc": (\d+\.\d+)}'),
]
feature_lis = [
    'f1',
    'UAR',
    ''
]
for compile_, f in zip(compile_lis, feature_lis):
    f1_lis, len_, comment = get_best_data(compile_)
    for i in range(len(comment)):
        # print(comment[i], f, max(f1_lis[i * len_: (i + 1) * len_]))
        print('{'+'"metric"'+': {"'+comment[i]+f+'":'+str(max(f1_lis[i * len_: (i + 1) * len_]))+"}}")


