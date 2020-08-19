import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from utils import create_folder
from settings import settings

# =====================================================================================
FPS = settings["FPS"]
FRAME_INTERVAL = settings["FRAME_INTERVAL"]
STARTTIME_FILE = settings['STARTTIME_TEST_FILE']
reliability_resample_path = settings['reliability_resample_path']
raw_path = settings['raw_path']
flow_path = settings['flow_path']
# =====================================================================================

def read_batch_final_results(
    window_size_sec,
    stride_sec,
    offset_sec,
    kde_num_offset,
    kde_max_offset,
    window_criterion,
    folder='./result/',
):
    """
    Read batch final results.

    Args:
        window_size_sec: int, window size
        stride_sec: int, stride
        offset_sec: float, offset in seconds
        kde_num_offset: int, KDE algorithm number of offset
        kde_max_offset: int, KDE algorithm max offset
        window_criterion: float, window criterion
        folder: str

    Returns:
        float, ave offset
        float, PV1000
        float, PV700
        float, PV300
        float, ave_conf
        int, num of videos
        int, ave_num_segments
        float, confidence

    """
    title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
    )
    result_file = folder + "/final_result_per_video" + title_suffix + ".csv"
    result_df = pd.read_csv(result_file)
    num_videos = len(result_df)
    error_abs = abs((result_df["offset"] + offset_sec * 1000).to_numpy())
    num_segs = result_df["num_segs"].to_numpy()
    try:
        ave_offset = np.mean(error_abs[~np.isnan(error_abs.astype(np.float))])
    except ZeroDivisionError:
        ave_offset = 0
        print("Error @offset_sec:", offset_sec, "kde_max_offset", kde_max_offset)
    try:
        ave_num_segs = np.mean(num_segs[~np.isnan(num_segs.astype(np.float))])
        PV1000 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 1000)]
        ) / float(num_videos)
        PV700 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 700)]
        ) / float(num_videos)
        PV300 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 300)]
        ) / float(num_videos)
        conf = 200000 / (result_df["sigma"].to_numpy() * result_df["mu_var"].to_numpy())
        ave_conf = np.mean(conf[~np.isnan(conf.astype(np.float))])
        return (
            ave_offset,
            PV1000,
            PV700,
            PV300,
            ave_conf,
            num_videos,
            ave_num_segs,
            conf,
        )
    except:
        print(title_suffix)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def summarize_xaxis_batch_to_csv(summ_path):
    """
    summarize xaxis batch and save to csv

    Args:
        summ_path: str, summary path

    Returns:
        None

    """
    file_path = "./file_list_random.txt"
    stride_sec = 1
    with open(file_path) as f:
        params = f.readlines()
    fout = open(summ_path, "w")
    fout.write(
        "window_size_sec,stride_sec,window_criterion,max_offset,kde_num_offset,offset_sec,num_videos,ave_num_segs,ave_offset,ave_conf,PV1000,PV700,PV300\n"
    )
    for line in params:
        (
            window_size_sec,
            window_criterion,
            max_offset,
            kde_num_offset,
            offset_sec,
        ) = line.split()
        title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
            window_size_sec,
            stride_sec,
            offset_sec,
            kde_num_offset,
            max_offset,
            window_criterion,
        )
        result_file = "result/summary_xx/final_result_per_video" + title_suffix + ".csv"
        result_df = pd.read_csv(result_file)
        num_videos = len(result_df)
        error_abs = abs((result_df["offset"] + float(offset_sec) * 1000).to_numpy())
        ave_offset = np.mean(error_abs[~np.isnan(error_abs.astype(np.float))])
        PV1000 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 1000)]
        ) / float(num_videos)
        PV700 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 700)]
        ) / float(num_videos)
        PV300 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 300)]
        ) / float(num_videos)
        conf = 200000 / (result_df["sigma"].to_numpy() * result_df["mu_var"].to_numpy())
        ave_conf = np.mean(
            conf[~np.isnan(conf.astype(np.float)) & (conf < 10000)]
        )  # dive into the problem there are two >10000 cases
        assert len(error_abs) == len(result_df)
        ave_num_segs = np.mean(result_df["num_segs"].to_numpy())
        fout.write(
            ",".join(
                map(
                    str,
                    [
                        window_size_sec,
                        stride_sec,
                        window_criterion,
                        max_offset,
                        kde_num_offset,
                        offset_sec,
                        num_videos,
                        ave_num_segs,
                        ave_offset,
                        ave_conf,
                        PV1000,
                        PV700,
                        PV300,
                    ],
                )
            )
            + "\n"
        )
    fout.close()
    
    result_df = pd.read_csv(summ_path)
    with open('final/syncwise_xaxis_final_result.txt', 'w') as f:
        print("Ave #Win Pairs: ", result_df["ave_num_segs"].mean(), file=f)
        print("Ave Error (ms): ", result_df["ave_offset"].mean(), file=f)
        print("PV700 (%): ", result_df["PV700"].mean(), file=f)
        print("PV300 (%): ", result_df["PV300"].mean(), file=f)

    print("Ave #Win Pairs: ", result_df["ave_num_segs"].mean())
    print("Ave Error (ms): ", result_df["ave_offset"].mean())
    print("PV700 (%): ", result_df["PV700"].mean())
    print("PV300 (%): ", result_df["PV300"].mean())


def summarize_pca_batch_to_csv(summ_path):
    """
    summarize pca batch and save to csv

    Args:
        summ_path: str, summary path

    Returns:
        None

    """
    file_path = "./file_list_random.txt"
    stride_sec = 1
    with open(file_path) as f:
        params = f.readlines()
    fout = open(summ_path, "w")
    fout.write(
        "window_size_sec,stride_sec,window_criterion,max_offset,kde_num_offset,offset_sec,num_videos,ave_num_segs,ave_offset,ave_conf,PV1000,PV700,PV300\n"
    )
    for line in params:
        (
            window_size_sec,
            window_criterion,
            max_offset,
            kde_num_offset,
            offset_sec,
        ) = line.split()
        title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
            window_size_sec,
            stride_sec,
            offset_sec,
            kde_num_offset,
            max_offset,
            window_criterion,
        )
        result_file = (
            "result/summary_pca/final_result_per_video" + title_suffix + ".csv"
        )
        try:
            result_df = pd.read_csv(result_file)
        except:
            continue
        num_videos = len(result_df)
        error_abs = abs((result_df["offset"] + float(offset_sec) * 1000).to_numpy())
        ave_offset = np.mean(error_abs[~np.isnan(error_abs.astype(np.float))])
        PV1000 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 1000)]
        ) / float(num_videos)
        PV700 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 700)]
        ) / float(num_videos)
        PV300 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 300)]
        ) / float(num_videos)
        conf = 200000 / (result_df["sigma"].to_numpy() * result_df["mu_var"].to_numpy())
        ave_conf = np.mean(
            conf[~np.isnan(conf.astype(np.float)) & (conf < 10000)]
        )  # dive into the problem there are two >10000 cases
        assert len(error_abs) == len(result_df)
        ave_num_segs = np.mean(result_df["num_segs"].to_numpy())
        fout.write(
            ",".join(
                map(
                    str,
                    [
                        window_size_sec,
                        stride_sec,
                        window_criterion,
                        max_offset,
                        kde_num_offset,
                        offset_sec,
                        num_videos,
                        ave_num_segs,
                        ave_offset,
                        ave_conf,
                        PV1000,
                        PV700,
                        PV300,
                    ],
                )
            )
            + "\n"
        )
    fout.close()

    result_df = pd.read_csv(summ_path)
    with open("final/syncwise_pca_final_result.txt", 'w') as f:
        print("Ave #Win Pairs: ", result_df["ave_num_segs"].mean(), file=f)
        print("Ave Error (ms): ", result_df["ave_offset"].mean(), file=f)
        print("PV700 (%): ", result_df["PV700"].mean(), file=f)
        print("PV300 (%): ", result_df["PV300"].mean(), file=f)

    print("Ave #Win Pairs: ", result_df["ave_num_segs"].mean())
    print("Ave Error (ms): ", result_df["ave_offset"].mean())
    print("PV700 (%): ", result_df["PV700"].mean())
    print("PV300 (%): ", result_df["PV300"].mean())

#
# def summarize_ablation_augmentation_maxoffset():
#     window_size_sec = 10
#     stride_sec = 1
#     kde_num_offset = 20
#     window_criterion = 0.8
#     offset_secs = [0.0, 2.0, 4.0]
#     kde_max_offsets = [2000, 4000, 6000, 8000, 10000]
#     kde_max_offset_list = []
#     ave_offset_list = []
#     PV1000_list = []
#     PV700_list = []
#     PV300_list = []
#     ave_conf_list = []
#     ave_num_segs_list = []
#     first_time_flag = 1
#
#     latex = "./final/ablation_maxoffset_result_win{}_str{}_wincrt{}_pca_sigma500_latex.txt".format(
#         window_size_sec, stride_sec, window_criterion
#     )
#     f = open(latex, "w+")
#
#     for offset_sec in offset_secs:
#         for kde_max_offset in kde_max_offsets:
#             (
#                 ave_offset,
#                 PV1000,
#                 PV700,
#                 PV300,
#                 ave_conf,
#                 num_videos,
#                 ave_num_segs,
#                 conf,
#             ) = read_batch_final_results(
#                 window_size_sec,
#                 stride_sec,
#                 offset_sec,
#                 kde_num_offset,
#                 kde_max_offset,
#                 window_criterion,
#             )
#             if first_time_flag:
#                 last_num_videos = num_videos
#             else:
#                 assert last_num_videos == num_videos
#                 last_num_videos = num_videos
#
#             kde_max_offset_list.append(kde_max_offset)
#             ave_offset_list.append(ave_offset)
#             PV1000_list.append(PV1000)
#             PV700_list.append(PV700)
#             PV300_list.append(PV300)
#             ave_conf_list.append(ave_conf)
#             ave_num_segs_list.append(ave_num_segs)
#             f.write(
#                 " & {} & {} & {} & {} & {} & {}\\\ \n".format(
#                     offset_sec,
#                     int(kde_max_offset / 1000),
#                     int(ave_offset),
#                     round(PV700, 2),
#                     round(PV300, 2),
#                     int(ave_conf),
#                 )
#             )
#
#         summ_path = "./final/ablation_max_offset_result_win{}_str{}_offset{}_rdoffset{}_wincrt{}_pca_sigma500_{}videos.csv".format(
#             window_size_sec,
#             stride_sec,
#             offset_sec,
#             kde_num_offset,
#             window_criterion,
#             num_videos,
#         )
#         data = {
#             "kde_max_offset": kde_max_offset_list,
#             "ave_offset": ave_offset_list,
#             "PV1000": PV1000_list,
#             "PV700": PV700_list,
#             "PV300": PV300_list,
#             "ave_num_segs": ave_num_segs_list,
#         }
#         df = pd.DataFrame(
#             data,
#             columns=[
#                 "kde_max_offset",
#                 "ave_offset",
#                 "PV1000",
#                 "PV700",
#                 "PV300",
#                 "ave_num_segs",
#             ],
#         )
#         df.to_csv(summ_path, index=None)
#
#         df_plot = df[["kde_max_offset", "ave_offset"]]
#         fig, ax = plt.subplots()
#         ax.plot(
#             df_plot["kde_max_offset"] / 1000, df_plot["ave_offset"] / 1000
#         )
#         ax.set_xlabel("max range of random offset in wKDE / sec", fontsize=14)
#         ax.set_ylabel("average absolute error / sec", fontsize=14)
#         plt.grid()
#         fig.savefig(
#             os.path.join(
#                 "./figures/",
#                 "ablation_result_augmentation_win{}_ave_offset{}.eps".format(
#                     window_size_sec, offset_sec
#                 ),
#             ),
#             format="eps",
#             dpi=100,
#             bbox_inches="tight",
#         )
#     f.close()
#     f = open(latex, "r")
#     print(f.read())
#
#
# def summarize_ablation_augmentation_numoffset():
#     window_size_sec = 10
#     stride_sec = 1
#     window_criterion = 0.8
#     kde_max_offset = 3000
#     offset_secs = [0.0, 2.0, 4.0]
#     kde_num_offsets = [1, 20, 40, 60, 80, 100]
#     first_time_flag = 1
#
#     my_dpi = 100
#     fig = plt.figure(figsize=(500 / my_dpi, 300 / my_dpi))
#     ax = fig.add_subplot(111)
#
#     for offset_sec in offset_secs:
#         ave_offset_list = []
#         PV1000_list = []
#         PV700_list = []
#         PV300_list = []
#         ave_conf_list = []
#         ave_num_segs_list = []
#         for kde_num_offset in kde_num_offsets:
#             (
#                 ave_offset,
#                 PV1000,
#                 PV700,
#                 PV300,
#                 ave_conf,
#                 num_videos,
#                 ave_num_segs,
#                 conf,
#             ) = read_batch_final_results(
#                 window_size_sec,
#                 stride_sec,
#                 offset_sec,
#                 kde_num_offset,
#                 kde_max_offset,
#                 window_criterion,
#             )
#             if first_time_flag:
#                 last_num_videos = num_videos
#             else:
#                 assert last_num_videos == num_videos
#                 last_num_videos = num_videos
#             if ave_offset:
#                 ave_offset_list.append(ave_offset)
#                 PV1000_list.append(PV1000)
#                 PV700_list.append(PV700)
#                 PV300_list.append(PV300)
#                 ave_conf_list.append(ave_conf)
#                 ave_num_segs_list.append(ave_num_segs)
#
#         summ_path = "./result/ablation_num_offset_result_win{}_str{}_offset{}_max_offset{}_wincrt{}_pca_sigma500_{}videos.csv".format(
#             window_size_sec,
#             stride_sec,
#             offset_sec,
#             kde_max_offset,
#             window_criterion,
#             num_videos,
#         )
#         data = {
#             "kde_num_offset": kde_num_offsets,
#             "ave_offset": ave_offset_list,
#             "PV1000": PV1000_list,
#             "PV700": PV700_list,
#             "PV300": PV300_list,
#             "ave_num_segs": ave_num_segs_list,
#         }
#         df = pd.DataFrame(
#             data,
#             columns=[
#                 "kde_num_offset",
#                 "ave_offset",
#                 "PV1000",
#                 "PV700",
#                 "PV300",
#                 "ave_num_segs",
#             ],
#         )
#         df.to_csv(summ_path, index=None)
#
#         df_plot = df[["kde_num_offset", "ave_offset"]]
#         ax.plot(
#             df_plot["kde_num_offset"],
#             df_plot["ave_offset"],
#             label="input shift: {}s".format(offset_sec),
#         )
#
#     ax.set_xlabel("augmentation ratio in wKDE", fontsize=14)
#     ax.set_ylabel("average error / ms", fontsize=14)
#     ax.set_xticks([1, 20, 40, 60, 80, 100])
#     plt.grid()
#     plt.tight_layout()
#     fig.savefig(
#         os.path.join(
#             "./figures/",
#             "ablation_result_num_offset_win{}_maxoffset{}_offset{}.eps".format(
#                 window_size_sec, kde_max_offset, offset_sec
#             ),
#         ),
#         format="eps"
#     )
#     # plt.show()
#     plt.close()


if __name__ == "__main__":
    summarize_xaxis_batch_to_csv('./result/batch_result_xx_sigma500_flow_w_random.csv')
    summarize_pca_batch_to_csv('./result/batch_result_pca_sigma500_flow_w_random.csv')
    # summarize_ablation_augmentation_numoffset()
    # summarize_ablation_augmentation_maxoffset()
