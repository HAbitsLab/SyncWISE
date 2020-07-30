import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import re
import os


def timestamp2Sec(strTime):
    def t2s(tStr):
        if tStr:
            tList = tStr.split('_')
            return float(tList[0]) * 3600 + float(tList[1])*60 + float(tList[2]) + float(tList[3]) / (10 ** (len(tStr) - 9))
        return None
    secTime = [t2s(t) for t in strTime]
    return secTime
    
def read_sensor(file):
    file_hd = open(file)
    sensor_info = file_hd.readline()
    sensor_id = sensor_info.split()[1]
    col_names = file_hd.readline().split()
    data = file_hd.read().split('\n')
    data = [line.split('\t') for line in data]
    df_sensor = pd.DataFrame(data=data, columns=col_names)
    for col in col_names[:-2]:
        df_sensor[col] = pd.to_numeric(df_sensor[col], errors='coerce')
    df_sensor[col_names[-2]] = pd.to_numeric(df_sensor[col_names[-2]].str.extract('^(\d+)', expand=False), errors='coerce', downcast='integer')
    strTime = df_sensor[col_names[-1]].values.tolist()
    secTime = timestamp2Sec(strTime)
    df_sensor[col_names[-1]] = secTime
    df_sensor.dropna(inplace=True)
    df_sensor[col_names[-2]] = df_sensor[col_names[-2]].astype(int)
    return df_sensor

def read_video(file):
    file_hd = open(file)
    data = file_hd.readlines()
    data = [[line.split(' ')[0].split(':')[1], line.split()[-1]] for line in data]
    df_video = pd.DataFrame(data=data, columns=['Frame', 'SysTime'])
    assert df_video['Frame'][0] == str(1)
    last_frame_num = len(df_video) - 1
    while df_video.iloc[last_frame_num]['Frame'] == 'NaN':
        last_frame_num -= 1
    assert df_video['Frame'][last_frame_num] == str(last_frame_num + 1)
    df_video['Frame'] = np.arange(1, len(df_video)+1)
    strTime = df_video['SysTime'].values.tolist()
    secTime = timestamp2Sec(strTime)
    df_video['SysTime'] = secTime    
    return df_video

def check_df(df, delta=0.008, verbose=True):
    print('checking dataframe')
    deltaT = df['SysTime'].diff()
    if verbose:
        gap_index = np.argwhere(abs(deltaT.values[1:] - delta) > 1e-10) + 1
        gap_index = gap_index.squeeze().reshape(-1)
        print(deltaT.iloc[gap_index])
    neg_index = np.argwhere(deltaT.values[1:] < 0) + 1
    data = np.delete(df.values, neg_index.squeeze(), 0)
    df = pd.DataFrame(data=data, columns=df.columns)
    print('drop rows', neg_index)
    print('finish checking')
    return df

if __name__ == '__main__':
    file = '/home/yun/Downloads/CMU/sensor/S07_Brownie_3DMGX1/2794_01-30_16_30_49-time.txt'
    df = read_sensor(file)
    df = check_df(df, delta=0.008)
    file = '/home/yun/Downloads/CMU/video/S07_Brownie_Video/STime7150991-time-synch.txt'
    df_video = read_video(file)
    check_df(df_video, delta=0.033332)
