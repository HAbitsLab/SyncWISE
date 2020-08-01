import os
import sys
from collections import Counter
import pickle

import numpy as np
import pandas as pd

from settings import settings
from utils import create_folder

sys.path.append(os.path.join(os.path.dirname(__file__), "../../syncwise"))
from AbsErrorROC import gaussianVotingPerVideo as gaussian_voting_per_video
from drift_confidence import drift_confidence


FPS = settings["FPS"]
KERNEL_VAR = settings["kernel_var"]
temp_dir = settings["TEMP_DIR"]

def calc_win_offset_all(
        df_dataset_all,
        info_dataset_all,
        window_size_sec,
        stride_sec,
        kde_num_offset,
        qualified_window_num,
        pca,
        fps,
        draw=0
):
    """

    Args:
        df_dataset_all:
        info_dataset_all:
        window_size_sec:
        stride_sec:
        kde_num_offset:
        qualified_window_num:
        pca:
        fps:

    Returns:

    """
    video_all = []
    for info in info_dataset_all:
        video_all.append(info[0])
    counter = Counter(video_all)
    # select the qualified videos with a sufficient number of windows
    qualify_videos = []
    for i in counter:
        if counter[i] > qualified_window_num:
            qualify_videos.append(i)

    all_offset_list = []
    all_drift_list = []
    all_conf_list = []
    all_video_list = []
    all_starttime_list = []

    if draw and not os.path.exists("figures/MD2K_cross_corr_win" + str(window_size_sec) + "_str" + str(stride_sec)):
        os.makedirs("figures/MD2K_cross_corr_win" + str(window_size_sec) + "_str" + str(stride_sec))
    # iterate through all the qualified videos
    for video in qualify_videos:
        # for all the qualified windows in this video:
        for df, info in zip(df_dataset_all, info_dataset_all):
            if info[0] == video:
                out_path = (
                        "figures/MD2K_cross_corr_win"
                        + str(window_size_sec)
                        + "_str"
                        + str(stride_sec)
                        + "/corr_flow_acc_{}_{}".format(video, info[1])
                )
                drift, conf = drift_confidence(df, out_path, fps, pca=pca, save_fig=0)

                if kde_num_offset:
                    all_drift_list.append(drift - info[3])
                else:
                    all_drift_list.append(drift)

                all_offset_list.append(info[3])
                all_conf_list.append(conf)
                all_video_list.append(video)
                all_starttime_list.append(info[1])

    df = pd.DataFrame(
        {
            "confidence": all_conf_list,
            "offset": all_offset_list,
            "drift": all_drift_list,
            "video": all_video_list,
            "starttime": all_starttime_list,
        }
    )
    return df


def print_offset_summary(offset_df):
    """

    Args:
        offset_df:

    Returns:

    """
    l = len(offset_df)
    l1 = len(offset_df[(offset_df["offset"] > -700) & (offset_df["offset"] < 700)])
    l2 = len(offset_df[(offset_df["offset"] > -300) & (offset_df["offset"] < 300)])
    offset_abs = abs(offset_df["offset"].to_numpy())
    ave_offset = np.mean(offset_abs[~np.isnan(offset_abs.astype(np.float))])
    ave_segs = np.mean(offset_df["num_segs"])
    print(
        "{}/{} videos with error < 700 ms, {}/{} videos with error time < 300 ms".format(
            l1, l, l2, l
        )
    )
    print("average offset = ", ave_offset)
    print("ave segs = ", ave_segs)


def calc_video_offset_all(
        # df_dataset_all,
        # info_dataset_all,
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
        qualified_window_num,
        save_dir,
        pca,
        kernel_var=KERNEL_VAR,
        fps=FPS,
        draw=0
):
    """

    Args:
        kde_max_offset:
        window_criterion:
        offset_sec:
        df_dataset_all:
        info_dataset_all:
        window_size_sec:
        stride_sec:
        kde_num_offset:
        qualified_window_num:
        save_dir:
        pca:
        kernel_var:
        fps:
        draw:

    Returns:

    """
    title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
    )
    print('\noffset_sec', offset_sec, '\n')
    with open(
            os.path.join(temp_dir, "all_video" + title_suffix + "_df_dataset.pkl"), "rb"
    ) as handle:
        df_dataset_all = pickle.load(handle)
    with open(
            os.path.join(temp_dir, "all_video" + title_suffix + "_info_dataset.pkl"), "rb"
    ) as handle:
        info_dataset_all = pickle.load(handle)
    scores_dataframe = calc_win_offset_all(
        df_dataset_all=df_dataset_all,
        info_dataset_all=info_dataset_all,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        kde_num_offset=kde_num_offset,
        qualified_window_num=qualified_window_num,
        pca=pca,
        fps=fps,
        draw=0
    )
    scores_dataframe.to_csv(save_dir + '/conf_offset_win' + title_suffix + '.csv', index=None)

    if draw:
        create_folder("figures/MD2K_cross_corr" + title_suffix)

    offset_df, _ = gaussian_voting_per_video(
        scores_dataframe,
        kernel_var=kernel_var,
        thresh=0,
        draw=draw,
        folder="figures/MD2K_cross_corr" + title_suffix,
    )
    offset_df = offset_df.sort_values(by=["offset"])
    offset_df.to_csv(
        save_dir + "/final_result_per_video" + title_suffix + ".csv", index=None
    )
