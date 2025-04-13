batch_lis = [
    #     'fullinception',
    'fullinception3',
    #     'fullinception4',
    'fullinception5', 'fullinception51',
    # 'fullinception6',
    # 'fullinception7',
    # 'fullinception8',
    # 'fullinception9',
    # 'fullinception10',
    # 'fullinception12',
    # 'fullinception13',
    'fullinception14',
    'fullinception15', 'fullinception16', 'fullinception18', 'full19'
]
# "metric": {"hh": "EndOfOrder", "inception1": "EndOfOrder", "vov": "EndOfOrder", "big": "EndOfOrder",
#            "fullinception": "EndOfOrder", "fullinception3": "EndOfOrder", "fullinception5": "EndOfOrder",
#            "fullinception6": "EndOfOrder", "fullinception7": "EndOfOrder", "icassp": "EndOfOrder",
#            "fullinception2": "EndOfOrder", "fullconvinception": "EndOfOrder", "fullinception4": "EndOfOrder",
#            "inception1f1": "EndOfOrder", "fullinceptionf1": "EndOfOrder", "fullinception3f1": "EndOfOrder",
#            "fullinception4f1": "EndOfOrder", "fullinception5f1": "EndOfOrder", "fullinception6f1": "EndOfOrder",
#            "fullinception7f1": "EndOfOrder", "inception1UAR": "EndOfOrder", "fullinceptionUAR": "EndOfOrder",
#            "fullinception3UAR": "EndOfOrder", "fullinception4UAR": "EndOfOrder", "fullinception5UAR": "EndOfOrder",
#            "fullinception6UAR": "EndOfOrder", "fullinception7UAR": "EndOfOrder", "fullinception8": "EndOfOrder",
#            "fullinception8f1": "EndOfOrder", "fullinception8UAR": "EndOfOrder", "fullinception9": "EndOfOrder",
#            "fullinception9f1": "EndOfOrder", "fullinception9UAR": "EndOfOrder", "fullinception10": "EndOfOrder",
#            "fullinception10f1": "EndOfOrder", "fullinception10UAR": "EndOfOrder", "fullinception12": "EndOfOrder",
#            "fullinception12f1": "EndOfOrder", "fullinception12UAR": "EndOfOrder", "hhf1": "EndOfOrder",
#            "hhUAR": "EndOfOrder", "fullinception13": "EndOfOrder", "fullinception13f1": "EndOfOrder",
#            "fullinception13UAR": "EndOfOrder", "fullinception15": "EndOfOrder", "fullinception15f1": "EndOfOrder",
#            "fullinception15UAR": "EndOfOrder", "fullinception14": "EndOfOrder", "fullinception14UAR": "EndOfOrder",
#            "fullinception14f1": "EndOfOrder", "fullinception16": "EndOfOrder", "fullinception16UAR": "EndOfOrder",
#            "fullinception16f1": "EndOfOrder", "fullinception17": "EndOfOrder", "fullinception17UAR": "EndOfOrder",
#            "fullinception17f1": "EndOfOrder", "icasspUAR": "EndOfOrder", "icasspf1": "EndOfOrder"}
for i in batch_lis:
    print('"' + i + '"' + ": " + '"EndOfOrder",', end='')
    print('"' + i + 'f1"' + ": " + '"EndOfOrder",', end='')
    print('"' + i + 'UAR"' + ": " + '"EndOfOrder",', end='')
    print('"' + i + 'wa"' + ": " + '"EndOfOrder",', end='')
