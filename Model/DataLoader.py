from torch.utils.data.dataloader import Dataset, DataLoader
from utils import GetHeatmapData, GetOceanData
from parameters import GetParameters

parameters = GetParameters()
batch_size = parameters['batch_size']

# 构建数据集
class Dataset(Dataset):
    def __init__(self, data):                                   #传入的原始数据集是：若干天的热度矩阵 [n,100,100]
        self.data = data
        self.len = data.size(0)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index]        

def getDataloader(data, batch_size):                            # data : torch.Size([])
    data = Dataset(data)
    length = len(data)

    train = data[0:273,]
    test = data[273:-1,]

    train_loader = DataLoader(dataset=train,batch_size=batch_size,shuffle=False,drop_last=True)

    test_loader = DataLoader(dataset=test,batch_size=batch_size,shuffle=False,drop_last=True)

    return train_loader, test_loader

if __name__ == '__main__':
    heatmap_data = GetHeatmapData('v2_data')               # torch.size[n, 21, 100, 100]
    ocean_data = GetOceanData()                            # torch.size[n, 14, 6, 80, 80] 第一个6代表序列的长度，第二个6代表的是海洋特征的维度
    Htrain_loader, Htest_loader = getDataloader(heatmap_data, batch_size)          
    Otrain_loader, Otest_loader = getDataloader(ocean_data, batch_size)
