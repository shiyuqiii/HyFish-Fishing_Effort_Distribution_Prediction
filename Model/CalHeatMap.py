import pandas as pd
import time
import datetime
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os


RESOLUTION = 0.1
# calculate heat_matrix and save
def get_heat_matrix(start_year, start_month, start_day, end_year, end_month, end_day):
    heat = np.zeros(shape=[int((EAST-WEST)/RESOLUTION), int((NORTH-SOUTH) / RESOLUTION)])
    readpath = 'D:\BIgDataAnalysis\VMSDATA\metadatatest'
    filelist = os.listdir(readpath)

    start_time = time.mktime(datetime.datetime.strptime(f'{start_year}-{start_month}-{start_day} 0:0:0', '%Y-%m-%d %H:%M:%S').timetuple())
    end_time = time.mktime(datetime.datetime.strptime(f'{end_year}-{end_month}-{end_day} 0:0:0', '%Y-%m-%d %H:%M:%S').timetuple())

    for file in filelist:
        # print(file)

        data = pd.read_csv(f'{readpath}/{file}')

        df = data[(data.timestamp >= start_time) & (data.timestamp < end_time)].copy()
        df = df[(df['lon'] >= 120) & (df['lon'] < 130) & (df['lat'] >= 25) & (df['lat'] < 35)]          #筛选掉异常数据，左闭右开
        df = df[(df.velocity != 0) & (df['head'] != 0)]                                                 #去掉了停在港口的船的数据
        df = df[df['velocity'] >= 0.5]                                                                  #去掉可能停靠在港口，没有在捕捞的数据
        df = df[df['velocity'] <= 5.5]                                                                  #去掉在航行的数据

        #筛选近港口的数据
        # df['isport'] = 0
        # df.loc[df.lat <= split_line, 'isport'] = df.loc[df.lat <= split_line, 'lat']  - (df.loc[df.lat <= split_line, 'lon'] * k1 + b1) 
        # df.loc[df.lat > split_line, 'isport'] = df.loc[df.lat > split_line, 'lon'] * k2 + b2 - df.loc[df.lat > split_line, 'lat'] 
        # df = df.loc[df.isport > 0].reset_index(drop=True)

        df = df.reset_index(drop=True)
        effort = df.timestamp.diff()[1:]
        effort = effort[effort <= 1800]
        df = df.loc[effort.index - 1]
        row = ((df.lat.values - SOUTH) / RESOLUTION).astype(int)
        column = ((df.lon.values - WEST) / RESOLUTION).astype(int)  
        for i in range(len(row)):
            flip_row = int(100  - 1 - row[i])    #python矩阵与地图存储是翻转的
            heat[flip_row, column[i]] += effort.iloc[i]
        
    np.savetxt(f'D:/BIgDataAnalysis/RESOLUTION=0.125/{start_year}{start_month}{start_day}--{end_year}{end_month}{end_day}.csv', heat, delimiter=',', fmt='%d')
    #fig = sns.heatmap(np.mat(heat), cmap = 'rainbow').invert_yaxis()
    #fig.get_figure().savefig(f'D:/BIgDataAnalysis/output/{start_year}-{start_month}_{end_year}-{end_month}.png')

# get_heat_matrix(2017, 9, 1, 2018, 6, 1)