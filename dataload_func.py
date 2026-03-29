import argparse
import torch
import random
import numpy as np

import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from scipy.io import loadmat
from scipy import interpolate
import os
import functools
import pandas as pd
import pywt
from scipy import stats

def get_time_domain_feature(data):
    """
    提取 15个 时域特征
    
    @param data: shape 为 (m, n) 的 2D array 数据，其中，m 为样本个数， n 为样本（信号）长度
    @return: shape 为 (m, 15)  的 2D array 数据，其中，m 为样本个数。即 每个样本的16个时域特征
    """
    rows, cols = data.shape
    
    # 有量纲统计量
    max_value = np.amax(data, axis=1)  # 最大值
    peak_value = np.amax(abs(data), axis=1)  # 最大绝对值
    min_value = np.amin(data, axis=1)  # 最小值
    mean = np.mean(data, axis=1)  # 均值
    p_p_value = max_value - min_value  # 峰峰值
    abs_mean = np.mean(abs(data), axis=1)  # 绝对平均值
    rms = np.sqrt(np.sum(data**2, axis=1) / cols)  # 均方根值
    square_root_amplitude = (np.sum(np.sqrt(abs(data)), axis=1) / cols) ** 2  # 方根幅值
    # variance = np.var(data, axis=1)  # 方差
    std = np.std(data, axis=1)  # 标准差
    kurtosis = stats.kurtosis(data, axis=1)  # 峭度
    skewness = stats.skew(data, axis=1)  # 偏度
    # mean_amplitude = np.sum(np.abs(data), axis=1) / cols  # 平均幅值 == 绝对平均值
    
    # 无量纲统计量
    clearance_factor = peak_value / square_root_amplitude  # 裕度指标
    shape_factor = rms / abs_mean  # 波形指标
    impulse_factor = peak_value  / abs_mean  # 脉冲指标
    crest_factor = peak_value / rms  # 峰值指标
    # kurtosis_factor = kurtosis / (rms**4)  # 峭度指标
    
    features = [max_value, peak_value, min_value, mean, p_p_value, abs_mean, rms, square_root_amplitude,
                std, kurtosis, skewness,clearance_factor, shape_factor, impulse_factor, crest_factor]
    #column=['最大值','最大绝对值','最小值','均值','峰峰值','绝对平均值','均方根值','方根幅值','标准差'
    #,'峭度','偏度','裕度指标','波形指标','脉冲指标','峰值指标']
    
    return np.array(features).T

def get_frequency_domain_feature(data, sampling_frequency=1):
    """
    提取 4个 频域特征
    
    @param data: shape 为 (m, n) 的 2D array 数据，其中，m 为样本个数， n 为样本（信号）长度
    @param sampling_frequency: 采样频率
    @return: shape 为 (m, 4)  的 2D array 数据，其中，m 为样本个数。即 每个样本的4个频域特征
    """
    data_fft = np.fft.fft(data, axis=1)
    m, N = data_fft.shape  # 样本个数 和 信号长度
    
    # 傅里叶变换是对称的，只需取前半部分数据，否则由于 频率序列 是 正负对称的，会导致计算 重心频率求和 等时正负抵消
    mag = np.abs(data_fft)[: , : N // 2]  # 信号幅值
    freq = np.fft.fftfreq(N, 1 / sampling_frequency)[: N // 2]
    # mag = np.abs(data_fft)[: , N // 2: ]  # 信号幅值
    # freq = np.fft.fftfreq(N, 1 / sampling_frequency)[N // 2: ]
    
    ps = mag ** 2 / N  # 功率谱
    
    fc = np.sum(freq * ps, axis=1) / np.sum(ps, axis=1)  # 重心频率
    mf = np.mean(ps, axis=1)  # 平均频率
    rmsf = np.sqrt(np.sum(ps * np.square(freq), axis=1) / np.sum(ps, axis=1))  # 均方根频率
    
    freq_tile = np.tile(freq.reshape(1, -1), (m, 1))  # 复制 m 行
    fc_tile = np.tile(fc.reshape(-1, 1), (1, freq_tile.shape[1]))  # 复制 列，与 freq_tile 的列数对应
    vf = np.sum(np.square(freq_tile - fc_tile) * ps, axis=1) / np.sum(ps, axis=1)  # 频率方差
    features = [fc, mf, rmsf, vf]
    
    return np.array(features).T

def get_wavelet_packet_feature(data, wavelet='db3', mode='symmetric', maxlevel=3):
    """
    提取 小波包特征
    
    @param data: shape 为 (n, ) 的 1D array 数据，其中，n 为样本（信号）长度
    @return: 最后一层 子频带 的 能量百分比
    """
    wp = pywt.WaveletPacket(data, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
    
    nodes = [node.path for node in wp.get_level(maxlevel, 'natural')]  # 获得最后一层的节点路径
    
    e_i_list = []  # 节点能量
    for node in nodes:
        e_i = np.linalg.norm(wp[node].data, ord=None) ** 2  # 求 2范数，再开平方，得到 频段的能量（能量=信号的平方和）
        e_i_list.append(e_i)
    
    # 以 频段 能量 作为特征向量
    # features = e_i_list
        
    # 以 能量百分比 作为特征向量，能量值有时算出来会比较大，因而通过计算能量百分比将其进行缩放至 0~100 之间
    e_total = np.sum(e_i_list)  # 总能量
    features = []
    for e_i in e_i_list:
        features.append(e_i / e_total * 100)  # 能量百分比
    
    return np.array(features)

def get_all_time_domain_feature(data,keep_columns,args):
    ''' 得到时频域特征 '''
    data_values=data.values
    time_dict={}
    for i in tqdm.tqdm(set(data.time)):
        i_data=data_values[data_values[::,-1]==i]
        all_columns=[]
        all_featrues=[]
        for index,column in enumerate(keep_columns):
            column_data=i_data[:,index].reshape([1,-1])
            #-------提取特征---------
            time_feat=get_time_domain_feature(column_data)#时域特征
            freq_feat=get_frequency_domain_feature(column_data, sampling_frequency=10)#频域特征
            tife_feat=get_wavelet_packet_feature(column_data[0], wavelet='db3', 
                                                 mode='symmetric', maxlevel=3).reshape([1,-1])#时频域特征
            
            feat_columns=['最大值','最大绝对值','最小值','均值','峰峰值','绝对平均值','均方根值','方根幅值','标准差'
                                      ,'峭度','偏度','裕度指标','波形指标','脉冲指标','峰值指标']+['重心频率','平均频率','均方根频率',
                                                                               '频率方差']+[f'wave-{ii}' for ii in range(8)]
            all_columns+=[column+'-'+ii for ii in feat_columns]
            all_feature=np.concatenate([time_feat,freq_feat,tife_feat],axis=1)
            all_featrues+=all_feature[0].tolist()
    
        time_dict[i]=all_featrues

    time_dict_df=pd.DataFrame(time_dict,index=all_columns).T
    return time_dict_df

def get_dataset(args):
    """ 得到指定重采样频率、输入特征的数据 """
    save_path=args.process_data_path
    if os.path.exists('/'.join(args.process_data_path.split('/')[:-1])+'/')==False:
        os.mkdir('/'.join(args.process_data_path.split('/')[:-1])+'/')
    if os.path.exists(save_path)==False:#文件不存在，重新生成
        data=pd.read_csv(args.data_path)#,sep=';')
        # data=data.iloc[:-100000]
        mul=int(args.process_data_path.split('/')[2][-1])
        start=1000000*mul
        end=1000000*(mul+1)
        print(args.process_data_path.split('/')[2],start,'--->',end)
        data=data.iloc[start:end]
        print('data.shape',data.shape)
        
        if args.features=='MS':#使用多个特征输入
            keep_columns=[i for i in data.columns if i not in [args.target]]+[args.target]
        else:#使用单个特征输入
            keep_columns=[args.time_feature,args.target]

        print(1111,data.columns)
        print(1111,keep_columns)
        data=data[keep_columns]
        keep_columns.remove(args.time_feature)
        data=data.set_index(keys=args.time_feature)
        
        ## 重采样，并生成新特征        
        data['time']=data.index//args.resample_freq#重采样
        feature_1=data.groupby('time').mean()#重采样的特征之一
        if args.get_diff:#是否生成多阶差作为特征
            new_feat=[]
            for column in feature_1.columns:
                column_df=feature_1[[column]]
                new_feat.append(column_df)
                for i in range(1,args.get_diff_num+1):
                    nf=column_df.diff(i)
                    nf.columns=[column+f'_{i}']
                    new_feat.append(nf)
            feature_1=pd.concat(new_feat,axis=1)
        
        if args.if_creat_feature:
            time_dict_df=get_all_time_domain_feature(data,keep_columns,args)
            feature_1=pd.concat([feature_1,time_dict_df],axis=1)
        feature_1.to_csv(save_path)
    else:
        feature_1=pd.read_csv(save_path,index_col=0)
    return feature_1

def fill_full_time(all_feature):
    #填充缺失时间及其数值
    a=pd.DataFrame({},index=np.arange(all_feature.index.min(),all_feature.index.max()))
    all_feature=pd.concat([all_feature,a],axis=1)
    all_feature=all_feature.sort_index()
    all_feature=all_feature.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    return all_feature


class Dataset_Custom(Dataset):
    def __init__(self,args, flag='train', size=None,scale=True, timeenc=0, freq='h'):
        self.args=args
        if size==None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = args.features
        self.target = args.target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = args.resample_freq*10
        
        self.data_path = args.process_data_path
        self.data=self.__read_data__()
        
    def  __read_data__(self):
        global df_raw
        self.scaler = StandardScaler()

        #=============加载指定数据==============
        all_feature=get_dataset(self.args)#加载指定数据
        all_feature=fill_full_time(all_feature)#填充缺失值、补齐时间
        if 'date' not in all_feature.columns:
            all_feature['date']=all_feature.index.astype(int)
        df_raw=all_feature

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        df_raw=df_raw.reset_index()
        # print('------------',df_raw.shape)
        # print(df_raw.columns)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            cols_data = [i for i in df_raw.columns if (self.target in i)]
            # df_data = df_raw[[self.target]]
            cols_data.remove(self.target)
            cols_data+=[self.target]
            df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            # print('------------',train_data.shape)
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]

        if self.args.time_type=='abs':#绝对时间
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
                
        else:#相对时间的处理方式
            if self.timeenc == 0:
                # data_stamp = df_stamp.drop(['date'], 1).values
                data_stamp = df_stamp.values
            elif self.timeenc == 1:
                data_stamp = data_stamp.transpose(1, 0)

            data_stamp = np.arange(0,len(data_stamp)).reshape(data_stamp.shape)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        return df_data
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        if self.args.time_type=='rel':#补充的时间特征处理方式
            seq_x_mark = self.data_stamp[:len(seq_x_mark)]
            seq_y_mark = self.data_stamp[len(seq_x_mark):len(seq_x_mark)+len(seq_y_mark)]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def data_provider(args, flag,batch_size_=None):
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        if batch_size_ is not None:
            batch_size=batch_size_
        freq = 'h'
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = 'h'

    data_set = Dataset_Custom(
        args=args,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        scale=True,
        timeenc=0,
        freq=freq,
    )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader