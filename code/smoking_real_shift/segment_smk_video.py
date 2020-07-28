import os
import csv
import time
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statistics import mean, stdev, median
from scipy.stats import skew, kurtosis

from cross_correlation_func import cross_correlation_using_fft, compute_shift
from load_sensor_data import read_data_datefolder_hourfile
# from update_starttime import update_starttime
from settings import settings
from utils import csv_read

# ========================================================================
#TODO: go to settings
FPS = 29.969664
STARTTIME_FILE = "start_time_test.csv"
reliability_resample_path = "../../data/RESAMPLE200/wild/"
raw_path = "../../data/RAW/wild/"
flow_path = "../../data/flow_pwc/"
# ========================================================================


def load_start_time(df_start_time, vid_name):
    if vid_name not in df_start_time.index:
        return None
    start_time = df_start_time.loc[vid_name]["start_time"]
    return int(start_time)


def reliability_df_to_consecutive_seconds(df_sensor_rel, window_size_sec, stride_sec):
    # use the threshold ">=8Hz" criterion to select 'good' seconds
    rel_seconds = (
        df_sensor_rel[df_sensor_rel["SampleCounts"] > 7] #TODO: go to settings
        .sort_values(by="Time")["Time"]
        .values
    )
    # print('There are {0:.2f} % reliable seconds in sensor data.'.format(
    #     len(rel_seconds) / len(df_sensor_rel) * 100))
    win_start_end = consecutive_seconds(rel_seconds, window_size_sec, stride_sec)
    return win_start_end


def consecutive_seconds(rel_seconds, window_size_sec, stride_sec=1):
    """
    Function:
        return a list of all the possible [window_start, window_end] pairs containing consecutive seconds of length window_size_sec inside.
    Args:
        rel_seconds: a list of qualified seconds
        window_size_sec: int
        stride_sec: int
    Returns:
        win_start_end: a list of all the possible [window_start, window_end] pairs that meets the requirement.

    Test:
        >>> rel_seconds = [2,3,4,5,6,7,9,10,11,12,16,17,18]; window_size_sec = 3; stride_sec = 1
        >>> print(consecutive_seconds(rel_seconds, window_size_sec))
        >>> [[2, 4], [3, 5], [4, 6], [5, 7], [9, 11], [10, 12], [16, 18]]
    """
    win_start_end = []
    for i in range(0, len(rel_seconds) - window_size_sec + 1, stride_sec):
        if rel_seconds[i + window_size_sec - 1] - rel_seconds[i] == window_size_sec - 1:
            win_start_end.append([rel_seconds[i], rel_seconds[i + window_size_sec - 1]])
    return win_start_end


def load_flow(vid_path, fps, start_time, offset_sec=0):
    motion = pickle.load(open(vid_path, "rb"))
    step = 1000.0 / fps
    length = motion.shape[0]
    timestamps_int = np.arange(
        start_time + offset_sec * 1000,
        start_time + offset_sec * 1000 + length * step,
        step,
    ).astype(int)
    l = min(len(timestamps_int), motion.shape[0])
    timestamps_int = timestamps_int[:l]
    motion = motion[:l, :]
    df_flow = pd.DataFrame(
        {"time": timestamps_int, "flowx": motion[:, 0], "flowy": motion[:, 1]}
    )
    df_flow["second"] = (df_flow["time"] / 1000).astype(int)
    df_flow["diff_flowx"] = df_flow["flowx"].diff()
    df_flow["diff_flowy"] = df_flow["flowy"].diff()
    df_flow = df_flow.reset_index()
    return df_flow


def load_merge_sensors_cubic_interp(
    raw_path, sub, device, sensors, sensor_col_header, start_time, end_time, fps
):
    df_list = []
    for s, col in zip(sensors, sensor_col_header):
        df_sensor = read_data_datefolder_hourfile(
            raw_path, sub, device, s, start_time, end_time
        )
        df_sensor = df_sensor[["time", col]]
        # cubic spline interpolation
        df_sensor["time"] = pd.to_datetime(df_sensor["time"], unit="ms")
        df_sensor = df_sensor.set_index("time")
        if fps == FPS:
            df_resample = df_sensor.resample(
                "0.03336707S"
            ).mean()  # 0.033333333S is the most closest value to 1/30 pandas accepts
        df_resample = df_resample.interpolate(method="spline", order=3)
        df_list.append(df_resample)
    df_sensors = pd.concat(df_list, axis=1)
    return df_sensors


def pca_sensor_flow(df_sensor, df_flow):
    pca_sensor = PCA(n_components=1)
    df_sensor[["accx", "accy", "accz"]] -= df_sensor[["accx", "accy", "accz"]].mean()
    df_sensor["acc_pca"] = pca_sensor.fit_transform(
        df_sensor[["accx", "accy", "accz"]].to_numpy()
    )
    diffflow_mat = df_flow[["diff_flowx", "diff_flowy"]].to_numpy()
    diffflow_mat -= np.mean(diffflow_mat, axis=0)
    pca_diffflow = PCA(n_components=1)
    df_flow["diffflow_pca"] = pca_diffflow.fit_transform(diffflow_mat)
    return df_sensor, df_flow


def shift_video_w_random_offset(
    df_sensor,
    df_flow,
    vid_name,
    win_start_end,
    start_time,
    end_time,
    kde_num_offset,
    kde_max_offset,
    window_size_sec,
    window_criterion,
    fps,
):
    df_dataset_vid = []
    info_dataset_vid = []
    cnt_windows = 0
    # add an offset to each window sensor-video pair
    for pair in win_start_end:
        start = pair[0] * 1000
        end = pair[1] * 1000 + 1000
        df_window_sensor = df_sensor[
            (df_sensor["time"] >= pd.to_datetime(start, unit="ms"))
            & (df_sensor["time"] < pd.to_datetime(end, unit="ms"))
        ]
        for i in range(kde_num_offset):
            # match video df
            offset = random.randint(-kde_max_offset, kde_max_offset)
            offset = (
                min(offset, end_time - end)
                if offset > 0
                else max(offset, start_time - start)
            )
            df_window_flow = df_flow[
                (df_flow["time"] >= pd.to_datetime(start + offset, unit="ms"))
                & (df_flow["time"] < pd.to_datetime(end + offset, unit="ms"))
            ]
            pd.options.mode.chained_assignment = None
            df_window_flow.loc[:, "time"] = df_window_flow.loc[
                :, "time"
            ] - pd.Timedelta(offset, unit="ms")
            df_window = pd.merge_asof(
                df_window_sensor,
                df_window_flow,
                on="time",
                tolerance=pd.Timedelta("29.969664ms"),
                direction="nearest",
            ).set_index("time")
            df_window = df_window.dropna(how="any")
            if len(df_window) > fps * window_size_sec * window_criterion:
                cnt_windows += 1
                df_dataset_vid.append(df_window)
                info_dataset_vid.append(
                    [vid_name, start, end, offset]
                )  # video name, sensor start time, sensor end time, video offset
    return cnt_windows, df_dataset_vid, info_dataset_vid


def seg_smk_video(
    vid_target,
    window_size_sec=20,
    stride_sec=5,
    offset_sec=0,
    kde_num_offset=20,
    window_criterion=0.8,
    kde_max_offset=60000,
    fps=FPS,
):
    video_qualified_window_num_list = []
    df_dataset = []
    info_dataset = []

    device = "CHEST"
    sensor = "ACCELEROMETER"
    sensors = ["ACCELEROMETER_X", "ACCELEROMETER_Y", "ACCELEROMETER_Z"]
    sensor_col_header = ["accx", "accy", "accz"]

    # update start_time.csv, disable the update when start_time.csv is intentionally manually modified.
    # update_starttime(STARTTIME_FILE)

    df_start_time = csv_read(STARTTIME_FILE).set_index("video_name") # method pd.read_csv() may induce bug in extreme batch processing
    video_names = df_start_time.index.tolist()
    subjects = list(set([vid.split(" ")[0] for vid in video_names]))

    # for offset_sec in offset_secs:
    for sub in subjects:
        flow_dir = flow_path + "sub{}".format(sub)
        flowfiles = [
            f
            for f in os.listdir(flow_dir)
            if os.path.isfile(os.path.join(flow_dir, f))
        ]
        flowfiles = [f for f in flowfiles if f.endswith(".pkl")]

        for f in flowfiles:
            # get video name, also used as flow file name
            vid_name = f[:-4]
            if vid_name != vid_target:
                continue

            # load start end time
            start_time = load_start_time(df_start_time, vid_name)
            if start_time == None:
                print(vid_name, "not included in ", STARTTIME_FILE)
                continue
            video_len_ms = (17 * 60 + 43) * 1000
            end_time = int(start_time) + video_len_ms

            # load sensor reliability data
            df_sensor_rel = read_data_datefolder_hourfile(
                reliability_resample_path,
                sub,
                device,
                sensor + "_reliability",
                start_time,
                end_time,
            )

            # record consecutive seconds of a window length
            win_start_end = reliability_df_to_consecutive_seconds(
                df_sensor_rel, window_size_sec, stride_sec
            )

            # load optical flow data and assign unixtime to each frame
            df_flow = load_flow(
                os.path.join(flow_dir, vid_name + ".pkl"),
                fps,
                start_time,
                offset_sec,
            )

            ## extract the optical flow frames of the good seconds according to sensor data
            # df_flow_rel = pd.concat([df_flow[df_flow['second'] == i] for i in rel_seconds]).reset_index()
            # print('There are {0:.2f} % reliable seconds in optical flow data.'.format(
            # len(df_flow_rel) / len(df_flow) * 100))
            df_flow["time"] = pd.to_datetime(df_flow["time"], unit="ms")

            df_flow = df_flow[
                ["flowx", "flowy", "diff_flowx", "diff_flowy", "time"]
            ].set_index("time")

            # extract the raw data 'ACCELEROMETER_X' (,'ACCELEROMETER_Y', 'ACCELEROMETER_Z') of consecutive chunk and resample
            #   according to video frame timestamp.
            df_sensors = load_merge_sensors_cubic_interp(
                raw_path,
                sub,
                device,
                sensors,
                sensor_col_header,
                start_time,
                end_time,
                fps,
            )

            # concatenate df_sensors and df_flow
            df_list = [df_sensors, df_flow]
            # cubic spline interpolation
            df_resample = pd.merge_asof(
                df_list[1],
                df_list[0],
                on="time",
                tolerance=pd.Timedelta("30ms"),
                direction="nearest",
            ).set_index("time")

            df_resample = df_resample.dropna(how="any")
            df_sensor = df_resample[["accx", "accy", "accz"]]
            df_flow = df_resample[["flowx", "flowy", "diff_flowx", "diff_flowy"]]
            df_sensor = df_sensor.reset_index()
            df_flow = df_flow.reset_index()

            # PCA
            df_sensor, df_flow = pca_sensor_flow(df_sensor, df_flow)

            ## select anchor windows from sensor, apply shifts in videos
            (
                cnt_windows,
                df_dataset_vid,
                info_dataset_vid,
            ) = shift_video_w_random_offset(
                df_sensor,
                df_flow,
                vid_name,
                win_start_end,
                start_time,
                end_time,
                kde_num_offset,
                kde_max_offset,
                window_size_sec,
                window_criterion,
                fps,
            )
            df_dataset += df_dataset_vid
            info_dataset += info_dataset_vid
            print(
                cnt_windows,
                "/",
                len(win_start_end),
                "windows left for this video.\n",
            )
            video_qualified_window_num_list.append((vid_name, cnt_windows))

    title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
    )
    pd.DataFrame(
        video_qualified_window_num_list, columns=["vid_name", "window_num"]
    ).to_csv("./data/num_valid_windows" + title_suffix + ".csv", index=None)

    return df_dataset, info_dataset

