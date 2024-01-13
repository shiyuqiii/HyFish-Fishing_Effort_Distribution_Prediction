import numpy as np
import torch
import pandas as pd
import os
from parameters import GetParameters

# 参数读取
parameters = GetParameters()
pre_len = parameters['pre_len']
seq_len = parameters['seq_len']

# 读取数据
def GetHeatmapData(name):
    read_path = f'../../output/days_matrix/{name}'
    file_list = os.listdir(read_path)
    file_list.sort(key=lambda x:int(x[9:-5]))                           # 将文件名按顺序排列
    data =[]
    for file in file_list:
        # print(file)
        day_data = pd.read_csv(f'{read_path}/{file}', header=None, dtype=np.float32)
        day_data = torch.from_numpy(day_data.values).view(1, 100, 100)  # torch.Size([1, 100, 100])
        data.append(day_data)
    data = torch.stack(data, dim=0)                                     # torch.Size([n, 1, 100, 100])
    
    mid_data = []                                                       #将原始数据组织成序列数据并存储, 每21天为一个序列，前14天预测后7天
    for i in range(data.size(0) - seq_len - pre_len + 1):
        partial = data[i]
        for j in range(seq_len + pre_len - 1):
            partial = torch.cat([partial, data[i+j+1]], dim=0)
        mid_data.append(partial)
    mid_data = torch.stack(mid_data, dim=0)                             #torch.Size([n-20, 21, 100, 100])   
    return mid_data                                                         


def GetOceanData():
    oceandata = np.load('./factors_and_/ocean_data.npy').astype(np.float32) #(day, 6, 80, 80)
    oceandata = oceandata.reshape(-1, 1, 6, 80, 80)
    oceandata = torch.from_numpy(oceandata)
    
    mid_data = []                                                       #将原始数据组织成序列数据并存储, 每6天为一个序列
    for i in range(oceandata.size(0) - seq_len + 1):
        partial = oceandata[i]
        for j in range(seq_len - 1):
            partial = torch.cat([partial, oceandata[i+j+1]], dim=0)
        mid_data.append(partial)
    mid_data = torch.stack(mid_data, dim=0)                             #torch.Size([n-13, 14, 6, 80, 80])  
    return mid_data[0:-7,:,:,:,:]                                       #和热度矩阵保持一样的数据集长度            
    
    
# 归一化 
def Normalization(data):
    for i in range(data.size(0)):

        '''设置阈值，并进行归一化'''
        new_data = data[i].ravel()[np.flatnonzero(data[i])]   # ndarray
        new_data = abs(np.sort(-new_data))
        threshold = new_data[int(len(new_data) * 0.25)]

        data[i][data[i]>threshold] = torch.tensor(threshold)
        data[i] = data[i] / threshold * 10

    return data

def CalRMSR(x, y):
    x = x.cpu().flatten()
    y = y.cpu().flatten()
    rmse = np.linalg.norm(y-x, ord=2)/(len(y)**0.5)
    return rmse

# weekly_loss calculate
def CalWeeklyLoss(pre_result, label_result):
    weekly_loss = []
    pre = torch.stack(pre_result).reshape(-1, 7, 100, 100)
    label = torch.stack(label_result).reshape(-1, 7, 100, 100)
    print(pre.shape, label.shape)
    for i in range(0, len(pre), 7):
        weekly_loss.append(CalRMSR(pre[i], label[i]))
    return weekly_loss
