import os
import sys
import pandas as pd
from collections import Counter

from settings import settings

sys.path.append(os.path.join(os.path.dirname(__file__), "../../syncwise"))
from drift_confidence import drift_confidence
from abs_error_ROC import gaussian_voting_per_video


KERNEL_VAR = settings["kernel_var"]

def drift_windows_for_all_subjects(df_dataset, info_dataset, window_size_sec=20, stride_sec=5, kde_num_offset=1,
                                   qualified_window_num=200, fps=29.969664, draw=0):
    """
    Calculate window offset for all videos

    Args:
        df_dataset: dataframe, all videos dataset
        info_dataset: dataframe, all videos information
        window_size_sec: int, window size
        stride_sec: int, stride size
        kde_num_offset: int, in KDE algorithm number of offsets
        qualified_window_num: int, number of qualified windows
        fps: float
        draw: draw flag, default = 0

    Returns:
        dataframe, the starttime, offset and confidence for each window in video

    """
    video_all = []
    for info in info_dataset:
        video_all.append(info[0])
    counter = Counter(video_all)

    # select the qualified videos with a sufficient number of windows
    qualify_videos = []
    for i in counter:
        if counter[i] > qualified_window_num:
            qualify_videos.append(i)

    if qualify_videos == 0:
        print("ERROR:", video_all, 'video have less than ', qualified_window_num, ' qualified windows.')

    all_offset_list = []
    all_drift_list = []
    all_conf_list = []
    all_video_list = []
    all_starttime_list = []

    if draw and not os.path.exists('figures/MD2K_cross_corr_win' + str(window_size_sec) + '_str' + str(stride_sec)):
        os.makedirs('figures/MD2K_cross_corr_win' + str(window_size_sec) + '_str' + str(stride_sec))

    # iterate through all the qualified videos
    for video in qualify_videos:
        # for all the qualified windows in this video:
        for df, info in zip(df_dataset, info_dataset):
            if info[0] == video:
                out_path = 'figures/MD2K_cross_corr_win' + str(window_size_sec) + '_str' + str(stride_sec) \
                           + '/corr_flow_acc_{}_{}'.format(video, info[1])
                drift, conf = drift_confidence(df, out_path, fps, save_fig=draw)

                if kde_num_offset:
                    all_drift_list.append(drift - info[3])
                else:
                    all_drift_list.append(drift)
                all_offset_list.append(info[3])
                all_conf_list.append(conf)
                all_video_list.append(video)
                all_starttime_list.append(info[1])

    df = pd.DataFrame(
        {'confidence': all_conf_list, 'offset': all_offset_list, 'drift': all_drift_list, 'video': all_video_list,
         'starttime': all_starttime_list})
    return df


def calc_drift_all_windows(df_dataset, info_dataset, vid_target, window_size_sec, stride_sec, offset_sec,
                           kde_num_offset, kde_max_offset, window_criterion, qualified_window_num,
                           result_dir, kernel_var=KERNEL_VAR, fps=29.969664, draw=0):
    """
    calculate video offset for all videos

    Args:
        df_dataset: dataframe, data for all videos
        info_dataset: dataframe, information for all videos
        vid_target: str, video of target
        window_size_sec: int,  window size
        stride_sec: int, stride
        offset_sec: float, offset
        kde_num_offset: int, KDE number of offset
        kde_max_offset: int, KDE max offset
        window_criterion: float, window criterion
        qualified_window_num: int, number of qualified window
        save_dir: str, save directory
        pca: boolean, use pca or not
        kernel_var: int, kernel variance
        fps: float
        draw: boolean, draw figure or not

    Returns:
        None
    """
    folder = result_dir + '/result' + vid_target
    if not os.path.exists(folder):
        os.makedirs(folder)
    title_suffix = '{}_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}_pca_sigma{}'. \
        format(
            vid_target,
            window_size_sec,
            stride_sec,
            offset_sec,
            kde_num_offset,
            kde_max_offset,
            window_criterion,
            kernel_var
        )
    scores_dataframe = drift_windows_for_all_subjects(
        df_dataset=df_dataset,
        info_dataset=info_dataset,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        kde_num_offset=kde_num_offset,
        qualified_window_num=qualified_window_num,
        fps=fps,
        draw=draw
    )
    scores_dataframe.to_csv(folder + '/conf_drift_video' + title_suffix + '.csv', index=None)
    if draw:
        if not os.path.exists('figures/MD2K_cross_corr' + title_suffix):
            os.makedirs('figures/MD2K_cross_corr' + title_suffix)
    offset_df, _ = gaussian_voting_per_video(scores_dataframe, kernel_var=kernel_var, thresh=0, draw=draw,
                                          folder='figures/MD2K_cross_corr' + title_suffix)
    offset_df = offset_df.sort_values(by=['offset'])
    offset_df.to_csv(folder + '/final_result_per_video' + title_suffix + '.csv', index=None)
