import torch.nn as nn
import torch
import torch.nn.functional as F
from parameters import GetParameters
'''EvaLayerModel 是评估网络深度的模型
是含attetion的Encoder-Decoder模型:
'''

#参数读取
parameters = GetParameters()
batch_size = parameters['batch_size']
hidden_size = parameters['hidden_size']
device = parameters['device']
envs_width = parameters['envs_width']
envs_heigh = parameters['envs_heigh']
heat_width = parameters['heat_width']
heat_heigh = parameters['heat_heigh']
seq_len = parameters['seq_len']
envs_feature_len = parameters['envs_feature_len']
heat_feature_len = parameters['heat_feature_len']

class ResidualBlock(torch.nn.Module):
    def __init__(self, channel):
        super(ResidualBlock,self).__init__()
        
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
    
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)

class HeatmapExtraction(nn.Module): 
    def __init__(self, C_in):                                   # C_in = 14
        super(HeatmapExtraction, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(C_in, 64, kernel_size=1),                 #output:[batch_size, 64, 100, 100] 残差层外的卷积主要用来升维 降维
            nn.ReLU(),                                
            ResidualBlock(64),                                  #残差层负责卷积
            # ResidualBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.MaxPool2d(2),                                    #池化层用来减小图片尺寸，减少参数
            nn.ReLU(),
            ResidualBlock(128),
            # ResidualBlock(128),
            nn.Conv2d(128, C_in, kernel_size=1),
            nn.ReLU()
        )
    def forward(self, x):
        input_len = x.shape[1]
        x = x.reshape(-1, 1, heat_width, heat_heigh)
        x = self.cnn(x)
        x = x.reshape(batch_size, input_len, -1)                #torch.Size([batch_size, 14, 2500])
        return x
 
class CNN(nn.Module):
    def __init__(self, C_in):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(C_in, 32, kernel_size=1),
            nn.ReLU(),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            # ResidualBlock(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # ResidualBlock(32),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)                                     #[batch_size, 14, 400]
        return x

class OceanFeatureExtraction(nn.Module):                                        #torch.size [batch_size, 14, 6, 80, 80]
    def __init__(self):
        super(OceanFeatureExtraction, self).__init__()
        self.cnn1 = CNN(1)
        self.cnn2 = CNN(1)
        self.cnn3 = CNN(1)
        self.cnn4 = CNN(3)
        self.attention = torch.nn.Linear(envs_feature_len, 1)

    def forward(self, x):                                                       #(batchsize, 14, 6, 80, 80) 6表示有六个海洋特征
        x = x.view(-1, 6, envs_width, envs_heigh)        
        y1 = self.cnn1(x[:, 0, :, :].reshape(-1, 1, envs_width, envs_heigh))    #[batch_size, 14, 400]
        y2 = self.cnn2(x[:, 1, :, :].reshape(-1, 1, envs_width, envs_heigh))
        y3 = self.cnn3(x[:, 2, :, :].reshape(-1, 1, envs_width, envs_heigh))
        y4 = self.cnn4(x[:, 3:6, :, :])
 
        y = torch.stack([y1, y2, y3, y4], dim=2)                                          #[batch_size, 14, 4, 400]
        weights = F.softmax(torch.tanh(self.attention(y)).squeeze(), dim=2)               #[batch_size, 14, 4]

        weight_y1 = y1 * weights[:, :, 0].unsqueeze(2)
        weight_y2 = y2 * weights[:, :, 1].unsqueeze(2)
        weight_y3 = y3 * weights[:, :, 2].unsqueeze(2)
        weight_y4 = y4 * weights[:, :, 3].unsqueeze(2)

        y = torch.cat([weight_y1, weight_y2, weight_y3, weight_y4], dim=2)
        return y


class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(heat_feature_len + envs_feature_len * 4, 1024),
            nn.ReLU(),
        )

    def forward(self, x, y):
        feature = torch.cat([x, y], dim=2)
        feature = self.mlp(feature) #[batch_size, 14, 1024]
        return feature
    
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1) :
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return hidden

    def init_C0(self, batch_size):
        C_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return C_0

    def forward(self, feature):
        hidden = self.init_hidden(batch_size)
        C_0 = self.init_C0(batch_size)
        _, (hidden, C_0) = self.lstm(feature, (hidden, C_0))

        return hidden, C_0
    
class MLP_model(nn.Module):
    def __init__(self, hidden_size):
        super(MLP_model, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10000),
            nn.ReLU()
        )
    
    def forward(self, inputs):              # inputs: [batch_size, hidden_size] 线性层默认只对最后一维进行处理
        pre = self.mlp(inputs)              # pre: [batch_size, 10000]
        pre = pre.reshape(batch_size, 1, 100, 100)
        return pre
        
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cnn = HeatmapExtraction(1)
        self.reduction = nn.Linear(heat_feature_len, 1024)
        self.lstmcell = nn.LSTMCell(input_size, hidden_size)
        self.mlp = MLP_model(hidden_size)
        #self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, decoder_inputs, decoder_h0, decoder_C0, ground_truth):        #当使用lstmcell时，decoder_inputs传入的只是一天的图       
        if ground_truth == 1:                                                       #训练
            cnn_feature1 = self.reduction(self.cnn(decoder_inputs[:, 0:1, :, :]).squeeze())   
            cnn_feature2 = self.reduction(self.cnn(decoder_inputs[:, 1:2, :, :]).squeeze())
            cnn_feature3 = self.reduction(self.cnn(decoder_inputs[:, 2:3, :, :]).squeeze())
            cnn_feature4 = self.reduction(self.cnn(decoder_inputs[:, 3:4, :, :]).squeeze())
            cnn_feature5 = self.reduction(self.cnn(decoder_inputs[:, 4:5, :, :]).squeeze())
            cnn_feature6 = self.reduction(self.cnn(decoder_inputs[:, 5:6, :, :]).squeeze())
            cnn_feature7 = self.reduction(self.cnn(decoder_inputs[:, 6:7, :, :]).squeeze())

            hidden_pre1, C = self.lstmcell(cnn_feature1, (decoder_h0.squeeze(), decoder_C0.squeeze()))  #output[batch_size, hidden_size]
            hidden_pre2, C = self.lstmcell(cnn_feature2, (hidden_pre1, C))
            hidden_pre3, C = self.lstmcell(cnn_feature3, (hidden_pre2, C))
            hidden_pre4, C = self.lstmcell(cnn_feature4, (hidden_pre3, C))
            hidden_pre5, C = self.lstmcell(cnn_feature5, (hidden_pre4, C))
            hidden_pre6, C = self.lstmcell(cnn_feature6, (hidden_pre5, C))
            hidden_pre7, C = self.lstmcell(cnn_feature7, (hidden_pre6, C))

            pre1 = self.mlp(hidden_pre1)
            pre2 = self.mlp(hidden_pre2)
            pre3 = self.mlp(hidden_pre3) 
            pre4 = self.mlp(hidden_pre4)
            pre5 = self.mlp(hidden_pre5)
            pre6 = self.mlp(hidden_pre6)
            pre7 = self.mlp(hidden_pre7)

        else:                                                                       #测试
            cnn_feature = self.cnn(decoder_inputs).squeeze()
            cnn_feature = self.reduction(cnn_feature)
            hidden_pre1, C = self.lstmcell(cnn_feature, (decoder_h0.squeeze(), decoder_C0.squeeze()))
            pre1 = self.mlp(hidden_pre1)

            cnn_feature = self.cnn(pre1).squeeze()
            cnn_feature = self.reduction(cnn_feature)
            hidden_pre2, C = self.lstmcell(cnn_feature, (hidden_pre1, C))
            pre2 = self.mlp(hidden_pre2)

            cnn_feature = self.cnn(pre2).squeeze()
            cnn_feature = self.reduction(cnn_feature)
            hidden_pre3, C = self.lstmcell(cnn_feature, (hidden_pre2, C))
            pre3 = self.mlp(hidden_pre3)

            cnn_feature = self.cnn(pre3).squeeze()
            cnn_feature = self.reduction(cnn_feature)
            hidden_pre4, C = self.lstmcell(cnn_feature, (hidden_pre3, C))
            pre4 = self.mlp(hidden_pre4)

            cnn_feature = self.cnn(pre4).squeeze()
            cnn_feature = self.reduction(cnn_feature)
            hidden_pre5, C = self.lstmcell(cnn_feature, (hidden_pre4, C))
            pre5 = self.mlp(hidden_pre5)

            cnn_feature = self.cnn(pre5).squeeze()
            cnn_feature = self.reduction(cnn_feature)
            hidden_pre6, C = self.lstmcell(cnn_feature, (hidden_pre5, C))
            pre6 = self.mlp(hidden_pre6)

            cnn_feature = self.cnn(pre6).squeeze()
            cnn_feature = self.reduction(cnn_feature)
            hidden_pre7, C = self.lstmcell(cnn_feature, (hidden_pre6, C))
            pre7 = self.mlp(hidden_pre7)

        pre = torch.cat([pre1, pre2, pre3, pre4, pre5, pre6, pre7], dim=1)
        return pre
    
'''封装全部类'''
class HeatMatrixPrediction(nn.Module):
    def __init__(self):
        super(HeatMatrixPrediction, self).__init__()
        self.cnn1 = HeatmapExtraction(1)
        self.cnn2 = OceanFeatureExtraction()
        self.fusion = FeatureFusion()
        self.encoder = Encoder(1024, hidden_size)
        self.decoder = Decoder(1024,  hidden_size) 

    def forward(self, heatmap_inputs, ocean_data, decoder_inputs, ground_truth):

        feature1 = self.cnn1(heatmap_inputs)                                # [batch_size, 14, 100, 100] --> [batch_size, 14, 2500]
        feature2 = self.cnn2(ocean_data)
        feature = self.fusion(feature1, feature2)                           # torch.size([batch_size, 14, 1024])

        decoder_h0, decoder_C0 = self.encoder(feature)                      # encoder是batch fisrt的

        pre = self.decoder(decoder_inputs, decoder_h0, decoder_C0, ground_truth)
        return pre

