import json
import os

mp = '/home/pwy/lhc/EMODB/code/logs'
# each_data = 'IEMOCAP'
# each_data = "EMODB"
each_data = 'IMPRO_IEMOCAP'
frame = 185
speakers = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16'] if each_data == 'EMODB' else [
    '01F', '01M', '02F', '02M', '03F', '03M',
    '04F', '04M', '05F', '05M'
]
spk_dic = {i: {} for i in speakers}
for i in os.listdir(mp):
    mini_p = os.path.join(mp, i)
    if os.path.isdir(mini_p):
        for j in os.listdir(mini_p):
            p = os.path.join(mini_p, j)
            if j == 'other.log':
                dataset = None
                fit_frame = None
                with open(p, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        speaker = data.get('other', {}).get('speaker')
                        ds = data.get('other', {}).get('dataset')
                        ft = data.get('other', {}).get('frame')
                        if ds:
                            dataset = ds
                        if ft:
                            fit_frame = int(ft)
                        if speaker and dataset == each_data and frame == fit_frame:
                            with open(p.replace('other.log', 'best_metric.log'), 'r') as file:
                                for l in file:
                                    data = json.loads(l)
                                    second_bracket_key = list(data.get('metric', {}).keys())[
                                        0] if 'metric' in data else None
                                    value = data['metric'][second_bracket_key]
                                    if second_bracket_key in spk_dic[speaker]:
                                        if spk_dic[speaker][second_bracket_key] < value:
                                            spk_dic[speaker][second_bracket_key] = value
                                    else:
                                        spk_dic[speaker][second_bracket_key] = value
model = ['full19', 'fullin', 'full20', ]
print(f'         acc\t       UAR\t       f1\t       wa\t       acc\t       UAR\t       f1\t       wa\t')
for mod in model:
    dic = {}
    for spk in speakers:
        for k in spk_dic[spk].keys():
            if mod in k:
                if k not in dic:
                    dic[k] = [spk_dic[spk][k]]
                else:
                    dic[k].append(spk_dic[spk][k])
    print(mod, '\t', end='')
    for k in dic.keys():
        print(f'{sum(dic[k]) / len(dic[k]):.6f}\t', end='')
    print()
    # value = spk_dic[spk]
for spk in speakers:
    print(f'spk:{spk}   acc\t       UAR\t       f1\t       wa\t       acc\t       UAR\t       f1\t       wa\t')
    for mod in model:
        print(mod, '\t', end='')
        for k in spk_dic[spk].keys():
            if mod in k:
                print(f'{spk_dic[spk][k]:.6f}\t', end='')
        print()

# print(spk_dic)
