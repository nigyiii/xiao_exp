import os
import random
import copy
from tqdm import tqdm

train_ratio = 0.9
val_ratio = 1.0 - train_ratio

txt_file = 'data.txt'
train_txt = open('data_train.txt', 'w+')
val_txt = open('data_val.txt', 'w+')
data_list = [line.strip().split() for line in open(txt_file)][29999:50000]
length = len(data_list)
idx_list = [i for i in range(0, length)]
val_idx_list = random.sample(idx_list, int(val_ratio * length))
print('random idx generated')
val_list = []
train_list = copy.deepcopy(data_list)
print('splitting set')
for idx in tqdm(val_idx_list):
    val_list.append(data_list[idx])
    train_list.remove(data_list[idx])
print('train len:', len(train_list))
print('val len:', len(val_list))
print('writting into txt')
for data in tqdm(train_list):
    this_line = ''
    for ele in data:
        this_line += (str(ele) + ' ')
    train_txt.write(this_line + '\n')
for data in tqdm(val_list):
    this_line = ''
    for ele in data:
        this_line += (str(ele) + ' ')
    val_txt.write(this_line + '\n')

train_txt.close()
val_txt.close()