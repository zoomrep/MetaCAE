import json
import os
import time
import tarfile

txt_path = "E:\\干工作\\2022年论文\\2023ncca\\审稿回复以及提交\\实验\\ACOCTE\\4.txt"
with open(txt_path, "r", encoding="utf-8") as fr:
    lines = fr.readlines()
fr.close()


# for line in lines:
#     # line = line.replace(r"[\"\',]", '')
#     line = line.replace(',',' ')
#     line = line.replace('\'',' ')
#     # index = line.index('precision')
#     # print(line[index + 12:index + 18])  # precision  12-17
#     # index = line.index('recall')
#     # print(line[index + 9:index + 15])  # recall 9-12
#     # index = line.index('shd')
#     # print(line[index + 6: index+8])  # shd 6
#     # index = line.index('F1')
#     # print(line[index + 5: index + 11])  # F1 5-11
#     index = line.index('tpr')
#     print(line[index + 6: index + 11])  # tpr 6-11
#     # index = line.index('fpr')
#     # print(line[index + 6: index + 11])  # fpr 6-11

for line in lines:
    # line = line.replace(r"[\"\',]", '')
    line = line.replace(',',' ')
    line = line.replace('\'',' ')
    index = line.index('Precission')
    print(line[index + 11:index + 16])  # precision  12-17
    # index = line.index('Recall')
    # print(line[index + 7:index + 12])  # recall 9-12
    # index = line.index('SHD')
    # print(line[index + 4: index+6])  # shd 6
    # index = line.index('F')
    # print(line[index + 2: index + 6])  # F1 5-11
    # index = line.index('tpr')
    # print(line[index + 6: index + 11])  # tpr 6-11
    # index = line.index('fpr')
    # print(line[index + 6: index + 11])  # fpr 6-11