# TODO: change resample method to the same as in syncWISE

import os
import sys
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter

from load_sensor_data import read_data_datefolder_hourfile
from resample import resample

sys.path.append('..')
from utils import create_folder
from settings import settings

sys.path.append(os.path.join(os.path.dirname(__file__), "../syncwise"))
from cross_correlation_func import compute_shift, cross_correlation_using_fft


def plot_MIT_baseline(df_resample, len_raw_sensor, fps, out_path):
    fps=29.969664
    df_resample = df_resample.reset_index()
    df_resample['time'] -= df_resample['time'][0]
    df_resample['time'] /= 1000
    df_resample = df_resample.set_index('time')
    fig, axes = plt.subplots(4, 1, figsize=(20, 10))
    grid = plt.GridSpec(2, 3)
    axes[0].plot(df_resample['accx'],  label='accx')
    axes[0].plot(df_resample['accy'], label='accy')
    axes[0].plot(df_resample['accz'], label='accz')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Accelerometer data')

    # plt.set_color_cycle(['red', 'black', 'blue', 'yellow', 'grey'])
    axes[1].plot(df_resample['diff_flowx'], 'b', label='diff_flowx')
    axes[2].plot(df_resample['diff_flowy'], 'orange', label='diff_flowy')
    #ax2.legend(loc='upper right')
    axes[1].set_title('diff flow x')
    axes[2].set_title('diff flow y')

    fftshift = cross_correlation_using_fft(df_resample['diff_flowx'].values, df_resample['accx'].values)
    shift = compute_shift(fftshift)
    axes[3].plot(fftshift)
    axes[3].set_title('diff flowx/accx delta={:.1f} ms'.format(shift * 1000/fps))
    
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(out_path)
    plt.close()

    
def baseline_MIT_video_MD2K(window_size_sec, stride_sec, num_offsets, max_offset, window_criterion, offset_sec = 0, plot=0):

    df_start_time = pd.read_csv(settings['STARTTIME_TEST_FILE'])
    qualify_videos  = df_start_time["video_name"].tolist()
    print(qualify_videos)
    print(len(qualify_videos))
    subjects = list(set([vid[:3] for vid in qualify_videos]))
    print(subjects)
    print(len(subjects))
    fps = settings['FPS']

    DEVICE = 'CHEST'
    SENSOR = 'ACCELEROMETER'
    SENSORS = ['ACCELEROMETER_X', 'ACCELEROMETER_Y', 'ACCELEROMETER_Z']
    sensor_col_header = ['accx', 'accy', 'accz']
    start_time_file = settings['STARTTIME_FILE']
    RAW_PATH = settings['raw_path']
    RESAMPLE_PATH = settings['reliability_resample_path']

    df_subjects = []
    for sub in subjects:        
        flow_dir = os.path.join(settings['flow_path'], 'sub{}'.format(sub))
        flow_files = [f for f in os.listdir(flow_dir) if os.path.isfile(os.path.join(flow_dir, f))]
        flow_files = [f for f in flow_files if f.endswith('.pkl')]
        print("# of flow_files: ", len(flow_files))
        
        video_list = []
        offset_list = []
        for f in flow_files:
            vid_name = f[:-4]
            if vid_name not in qualify_videos:
                continue
            vid_path = os.path.join(flow_dir, vid_name+'.pkl')
            
            # load start end time
            offset = 0
            df_start_time = pd.read_csv(start_time_file, index_col='video_name')
            if vid_name not in df_start_time.index:
                continue
            start_time = df_start_time.loc[vid_name]['start_time']+offset
            vid_max_len = (17*60+43)*1000 # TODO: go to settings
            interval = [int(start_time), int(start_time) + vid_max_len]

            # load sensor data for drawing
            # raw_path = settings['raw_path']
            df = read_data_datefolder_hourfile(RESAMPLE_PATH, sub, DEVICE, SENSOR, *interval)
            len_raw_sensor = len(df)

            # load sensor reliability data
            df_rel = read_data_datefolder_hourfile(RESAMPLE_PATH, sub, DEVICE, SENSOR + '_reliability', *interval)
            # use the threshold ">=7Hz" to select 'good' seconds
            rel_seconds = df_rel[df_rel['SampleCounts'] > 7].sort_values(by='Time')['Time'].values

            # load optical flow data and assign unixtime to each frame
            motion = pickle.load(open(vid_path, 'rb'))
            step = 1000.0/fps
            length = motion.shape[0]
            timestamps_int = np.arange(start_time, start_time + length * step, step).astype(int)
            timestamps_int = timestamps_int[:min(len(timestamps_int), motion.shape[0])]
            motion = motion[:min(len(timestamps_int), motion.shape[0]), :]
            assert len(timestamps_int) == motion.shape[0]
            df_flow = pd.DataFrame({'time': timestamps_int, 'flowx': motion[:, 0], 'flowy': motion[:, 1]})
            df_flow['second'] = (df_flow['time']/1000).astype(int)

            # extract the optical flow frames of the good seconds according to sensor data
            df_flow_rel = pd.concat([df_flow[df_flow['second']==i] for i in rel_seconds]).reset_index()            
            fixed_time_col = df_flow_rel['time'].values
            df_flow_rel = df_flow_rel[['flowx', 'flowy', 'time']].set_index('time')
            
            # extract the data of consecutive chunk and resample according to video frame timestamp
            df_list = []
            for S, col in zip(SENSORS, sensor_col_header):
                df = read_data_datefolder_hourfile(RAW_PATH, sub, DEVICE, S, fixed_time_col[0], fixed_time_col[-1])
                df = df[['time', col]]
                df_sensor_resample = resample(df, 'time', samplingRate=0,
                                              gapTolerance=200, fixedTimeColumn=fixed_time_col).set_index('time')
                df_list.append(df_sensor_resample)

            df_list.append(df_flow_rel)
            df_resample = pd.concat(df_list, axis=1)
            df_resample = df_resample.dropna(how='any')
            df_resample['accx'] -= df_resample['accx'].mean()
            df_resample['accy'] -= df_resample['accy'].mean()
            df_resample['accz'] -= df_resample['accz'].mean()
            df_resample['diff_flowx'] = df_resample['flowx'].diff()
            df_resample['diff_flowy'] = df_resample['flowy'].diff()
            l1 = len(df_resample)
            df_resample = df_resample.dropna(how='any')
            l2 = len(df_resample)
            assert l2 + 1 == l1

            fftshift = cross_correlation_using_fft(df_resample['diff_flowx'].values, df_resample['accx'].values)
            shift = compute_shift(fftshift)
            shift_ms = shift * 1000 / fps
            # print('diff_flowx accx delta={:.1f} ms'.format(shift_ms))
            video_list.append(vid_name)
            offset_list.append(shift_ms)

            if plot:
                out_path = 'figures/figures_MIT_xx/corr_flow_averaged_acc_{}.png'.format(vid_name)
                plot_MIT_baseline(df_resample, len_raw_sensor, fps, out_path)
                
        df_subj = pd.DataFrame({'video': video_list, 'offset': offset_list})
        df_subjects.append(df_subj)
    result_df = pd.concat(df_subjects)
    result_df = result_df.reset_index()
    ave_error = np.mean(np.abs(result_df['offset'].values))
    PV300 = np.sum(np.abs(result_df['offset'].values) < 300) / len(result_df) * 100
    PV700 = np.sum(np.abs(result_df['offset'].values) < 700) / len(result_df) * 100
    result_df.to_csv('result/baseline_xx/baseline_MIT_entirevideo_MD2K_offset.csv', index=None)
    # result_df.to_csv('result/baseline_xx/baseline_MIT_entirevideo_MD2K_offset' + title_suffix + '.csv', index=None)
    print(ave_error, PV300, PV700)
    return result_df


def summarize_batch_result():
    file_path = './file_list_random.txt'
    summ_path = './result/batch_result_baseline_MIT_random.csv'
    stride_sec = 1
    num_offsets = 20
    with open(file_path) as f:
        params = f.readlines()
    params = params[:1]
    fout = open(summ_path, 'w')
    fout.write('window_size_sec, stride_sec, num_offsets, window_criterion, max_offset, num_videos, ave_offset, num_1000ms, num_700ms, num_300ms\n')
    for line in params:
        window_size_sec, window_criterion, max_offset, num_offsets, offset_sec = line.split()
        title_suffix = '_win{}_str{}_rdoffset{}_maxoffset{}_wincrt{}_pca'\
            .format(window_size_sec, stride_sec, num_offsets, max_offset, window_criterion)
        result_df = baseline_MIT_video_MD2K(window_size_sec, stride_sec, num_offsets, max_offset, window_criterion, offset_sec=offset_sec)
        print(result_df)
        error_abs = abs(result_df['offset'].to_numpy())
        ave_offset = np.mean(error_abs[~np.isnan(error_abs.astype(np.float))])
        num_1000ms = len(error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 1000)])
        num_700ms = len(error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 700)])
        num_300ms = len(error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 300)])
        num_videos = len(result_df)
        assert len(error_abs) == len(result_df)
        fout.write(','.join(map(str, [window_size_sec, stride_sec, num_offsets, window_criterion, max_offset, num_videos, ave_offset, num_1000ms, num_700ms, num_300ms]))+'\n')
    fout.close()


if __name__ == "__main__":
    window_size_sec = 10
    stride_sec = 1
    num_offsets = 20
    max_offset = 5000
    window_criterion = 0.8
    plot = 0
    offset_sec = 1.9898898767622777

    create_folder("result/baseline_xx/")
    summarize_batch_result()



