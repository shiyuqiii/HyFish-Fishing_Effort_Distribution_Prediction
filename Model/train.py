import math
import torch
import os
import torch.nn as nn
from utils import Normalization, GetHeatmapData, GetOceanData
from parameters import GetParameters
from DataLoader import getDataloader
from EvaLayerModel import HeatMatrixPrediction
from tqdm import tqdm
import numpy as np

parameters = GetParameters()
device = parameters['device']
batch_size = parameters['batch_size']
learning_rate = parameters['learning_rate']
Layer = parameters['Layer']

def train(epoch, Htrain_loader, Otrain_loader, model):
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if os.path.exists(f'./factors_and_/Layer{Layer}_best_model.pth'):
        model.load_state_dict(torch.load('./factors_and_/Layer{Layer_h}_best_model.pth'))
        train_loss=list(np.load('./factors_and_/Layer{Layer_h}_train_loss.npy'))
        print('there has a well-trained model.\n'
              'loading and continue training\n')
    else:
        train_loss = []

    best_loss = np.inf
    for epoch in range(epoch):
        running_loss = 0.0
        batch_num = 0
        for batch_idx, (heatmap_data, ocean_data) in tqdm(enumerate(zip(Htrain_loader, Otrain_loader)), total=len(Htrain_loader)):
            heatmap_data = Normalization(heatmap_data)
            heatmap_data = heatmap_data.to(device)
            ocean_data = ocean_data.to(device) #[16,14,6,80,80]

            heatmap_inputs = heatmap_data[:, 0:14, :, :]                                     #inputs size: [batch_size, 14, 100, 100]
            labels = heatmap_data[:, 14:21, :, :]                                             #torch.size([batch_size, 3, 100, 100])
            decoder_inputs = heatmap_data[:, 13:20, :, :]             

            pre = model(heatmap_inputs, ocean_data, decoder_inputs, ground_truth=1)    #ground_truth=1表示使用真实值在训练
            loss = criterion(pre, labels)
            optimizer.zero_grad()

            # for name, params in model.named_parameters():
            #     print("name--->", name)
            #     print("params--->", params.size())
            #     print("grad-requires--->", params.requires_grad)
            #     print("grad-value--->", params.grad.size())

            loss.backward()
            optimizer.step()

            '''打印模型参数梯度等信息'''
            # print("======================更新之后=======================")
            # for name, params in model.named_parameters():
            #     print("name--->", name)
            #     print("params--->", params.size())
            #     print("grad-requires--->", params.requires_grad)
            #     print("grad-value--->", params.grad.size())

            running_loss += math.sqrt(loss.item())
            batch_num = batch_idx
        # finishi an epoch training
        cur_loss = running_loss / (batch_num + 1)
        train_loss.append(cur_loss)
        np.save('./factors_and_/Layer{Layer_h}_train_loss', train_loss) #每一轮都保存一下loss
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_epoch = epoch
            torch.save(model.state_dict(), './factors_and_/Layer{Layer_h}_best_model.pth')
            torch.save(model, './factors_and_/Layer{Layer_h}_model_save.pt')
        print('[epoch %d] training loss: %.5f' % (epoch + 1, cur_loss))
    #sava final loss and model
    np.save('./factors_and_/Layer{Layer_h}_train_loss', train_loss)
    torch.save(model, './factors_and_/Layer{Layer_h}_model_save.pt')
    return train_loss

if __name__ == '__main__':
    heatmap_data = GetHeatmapData('v2_data')               # torch.size[n, 21, 100, 100]
    ocean_data = GetOceanData()                            # torch.size[n, 14, 6, 80, 80] 第一个6代表序列的长度，第二个6代表的是海洋特征的维度
    Htrain_loader, Htest_loader = getDataloader(heatmap_data, batch_size)          
    Otrain_loader, Otest_loader = getDataloader(ocean_data, batch_size)
    heat_matrix_prediction = HeatMatrixPrediction().to(device)

    epoch = 5000 #1500 还没收敛
    # train(epoch, Htrain_loader, Otrain_loader, heat_matrix_prediction)
    print('In train')