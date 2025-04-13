'Macro_F1-fullinception3test'
batch_lis = [
    'fullinception', 'fullinception3',
    'fullinception4', 'fullinception5','fullinception51',
    'fullinception6', 'fullinception7',
    'fullinception8', 'fullinception9',
    'fullinception10', 'fullinception12',
    'fullinception13', 'fullinception14',
    'fullinception15', 'fullinception16',
    'fullinception18', 'full19'
]
# f1_lis, uar_lis = [], []
for i in batch_lis:
    print('Macro_F1-' + i + 'test,', "UAR-" + i + 'test,', "wa-" + i + 'test,', end='')

    # f1_lis.append('Macro_F1-'+i+'test')
    # uar_lis.append("UAR-"+i+'test')
