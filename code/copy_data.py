# path = '/home/pwy/lhc/EMODB/code/logs/log_20230518_141802/metric.log'
# path1 = '/home/pwy/lhc/EMODB/code/logs/log_20230518_141802_1/metric.log'
# path2 = '/home/pwy/lhc/EMODB/code/logs/log_20230518_141802_2/metric.log'
# path3 = '/home/pwy/lhc/EMODB/code/logs/log_20230518_141802_3/metric.log'
# path4 = '/home/pwy/lhc/EMODB/code/logs/log_20230518_141802_4/metric.log'
# with open(path, 'r') as f:
#     data = f.readlines()
#
# each = int(len(data) / 4)
# if '/n' in data[0]:
#     print(data[0])
# with open(path1, 'w+') as f:
#     for i in data[:each]:
#         f.write(i)
# with open(path2, 'w+') as f:
#     for i in data[each:each*2]:
#         f.write(i)
#         # f.write('\n')
# with open(path3, 'w+') as f:
#     for i in data[each*2:each*3]:
#         f.write(i)
# with open(path4, 'w+') as f:
#     for i in data[each * 3:each * 4]:
#         f.write(i)


