import torch
import torch.nn as nn
import math
from parameters import GetParameters
from DataLoader import getDataloader
from utils import Normalization, GetHeatmapData, GetOceanData

parameters = GetParameters()
device = parameters['device']
batch_size = parameters['batch_size']

def test(Htest_loader, Otest_loader,  heat_matrix_prediction, criterion):
    running_loss = 0.0
    batch_num = 0
    mini_loss = 0.0
    pre_result = []
    label_result = []
    with torch.no_grad():
        for batch_idx, (heatmap_data, ocean_data) in enumerate(zip(Htest_loader, Otest_loader)): 
            heatmap_data = Normalization(heatmap_data)
            heatmap_data = heatmap_data.to(device)
            ocean_data = ocean_data.to(device)

            heatmap_inputs = heatmap_data[:, 0:14, :, :]                                     #inputs size: [batch_size, 6, 100, 100]
            labels = heatmap_data[:,14:21, :, :]                                             #torch.size([batch_size, 3, 100, 100])
            decoder_inputs = heatmap_data[:, 13:14, :, :]

            pre = heat_matrix_prediction(heatmap_inputs, ocean_data, decoder_inputs, ground_truth=0)
            loss = criterion(pre, labels)
            running_loss += math.sqrt(loss.item())                                          #用RMSE作为误差的计算
            batch_num = batch_idx
            pre_result.append(pre)
            label_result.append(labels)
            if batch_idx == 0:
                mini_loss = math.sqrt(loss.item())
                mini_pre = pre
                mini_label = labels
                mini_batchidx = batch_idx                
            elif mini_loss > math.sqrt(loss.item()):
                mini_loss = math.sqrt(loss.item())
                mini_pre = pre
                mini_label = labels
                mini_batchidx = batch_idx
        test_loss = running_loss / (batch_num + 1)                                          #所有batch的平均loss

        print(' test RMSE: %.5f' % test_loss)
    return test_loss, pre_result, label_result, mini_pre, mini_label, mini_batchidx

if __name__ == '__main__':
    model = torch.load('model_save.pt')
    criterion = nn.MSELoss()
    heatmap_data = GetHeatmapData('v2_data')               # torch.size[n, 21, 100, 100]
    ocean_data = GetOceanData()
    Htrain_loader, Htest_loader = getDataloader(heatmap_data, batch_size)          
    Otrain_loader, Otest_loader = getDataloader(ocean_data, batch_size)
    test_loss, _, _, _, _, _ = test(Htest_loader, Otest_loader, model, criterion)