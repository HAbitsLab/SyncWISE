import os
import pandas as pd
from collections import Counter
from drift_confidence import drift_confidence
from AbsErrorROC import gaussianVotingPerVideo


def drift_windows_for_all_subjects(df_dataset, info_dataset,window_size_sec=20, stride_sec=5, offset_sec=0, kde_num_offset=1, max_offset=20000, \
                                window_criterion=0.8, fps=29.969664, draw=0):
    video_all = []
    for info in info_dataset:
        video_all.append(info[0])
    counter = Counter(video_all)

    # select the qualified videos with more than 20 windows
    qualify_videos = []
    # set the parameter number of qualified windows 
    qualified_window_num = 20
    for i in counter:
        if counter[i] > qualified_window_num:
            qualify_videos.append(i)
    # print(len(qualify_videos), 'videos have more than ', qualified_window_num, ' qualified windows.')

    all_offset_list = []
    all_drift_list = []
    all_conf_list = []
    all_video_list = []
    all_starttime_list = []

    if draw:
        if not os.path.exists('figures/MD2K_cross_corr_win' + str(window_size_sec) + '_str' + str(stride_sec)):
            os.makedirs('figures/MD2K_cross_corr_win' + str(window_size_sec) + '_str' + str(stride_sec))

    # iterate through all the qualified videos
    for video in qualify_videos:
        # for all the qualified windows in this video:
        for df, info in zip(df_dataset, info_dataset):
            if info[0] == video:
                out_path = 'figures/MD2K_cross_corr_win' + str(window_size_sec) + '_str' + str(stride_sec)\
                           + '/corr_flow_acc_{}_{}'.format(video, info[1])
                drift, conf = drift_confidence(df, out_path, fps, save_fig=draw)

                if kde_num_offset: # if conf > 4:
                    all_drift_list.append(drift-info[3])
                else:
                    all_drift_list.append(drift)
                all_offset_list.append(info[3])
                all_conf_list.append(conf)
                all_video_list.append(video)
                all_starttime_list.append(info[1])

    df = pd.DataFrame({'confidence': all_conf_list, 'offset': all_offset_list, 'drift': all_drift_list, 'video': all_video_list,
                       'starttime': all_starttime_list})
    return df


def calc_drift_all_windows(df_dataset, info_dataset, vid_target, window_size_sec, stride_sec, offset_sec, kde_num_offset, max_offset, window_criterion, 
                            result_dir, kernel_var=500, fps=29.969664, draw=0):
    folder = result_dir + '/result' + vid_target
    if not os.path.exists(folder):
        os.makedirs(folder)
    title_suffix = '{}_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}_pca_sigma500'.\
        format(vid_target, window_size_sec, stride_sec, offset_sec, kde_num_offset, max_offset, window_criterion)
    scores_dataframe = drift_windows_for_all_subjects(df_dataset=df_dataset, info_dataset=info_dataset, window_size_sec=window_size_sec, stride_sec=stride_sec, offset_sec=offset_sec,\
                                                      kde_num_offset=kde_num_offset, max_offset=max_offset, window_criterion=window_criterion,\
                                                      fps=fps, draw=draw)
    scores_dataframe.to_csv(folder+'/conf_drift_video' + title_suffix + '.csv', index=None)
    if draw:
        if not os.path.exists('figures/MD2K_cross_corr' + title_suffix):
            os.makedirs('figures/MD2K_cross_corr' + title_suffix)
    offset_df, _ = gaussianVotingPerVideo(scores_dataframe, kernel_var=500, thresh=0, draw=draw, folder='figures/MD2K_cross_corr' + title_suffix)
    offset_df = offset_df.sort_values(by=['offset'])
    offset_df.to_csv(folder+'/final_result_per_video' + title_suffix + '.csv', index=None)
