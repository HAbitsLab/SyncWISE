import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from load_sensor_data import read_data_datefolder_hourfile
from settings import settings
from utils import csv_read


FPS = settings["FPS"]
FRAME_INTERVAL = settings["FRAME_INTERVAL"]
sample_counts = settings["sample_counts"]


def load_start_time(start_time_file, vid):
    df_start_time = csv_read(start_time_file).set_index("video_name")
    if vid not in df_start_time.index:
        print("Error: ", vid, " not in ", start_time_file)
        exit()
    start_time = df_start_time.loc[vid]["start_time"]
    return int(start_time)


def reliability_df_to_consecutive_seconds(
    df_sensor_rel, window_size_sec, stride_sec, threshold=sample_counts
):
    # use the threshold ">=7Hz" criterion to select 'good' seconds
    rel_seconds = (
        df_sensor_rel[df_sensor_rel["SampleCounts"] > threshold]
        .sort_values(by="Time")["Time"]
        .values
    )
    win_start_end = consecutive_seconds(rel_seconds, window_size_sec, stride_sec)
    return win_start_end


def consecutive_seconds(rel_seconds, window_size_sec, stride_sec=1):
    """
    Return a list of all the possible [window_start, window_end] pairs 
        containing consecutive seconds of length window_size_sec inside.
    Args:
        rel_seconds: a list of qualified seconds
        window_size_sec: int
        stride_sec: int
    Returns:
        win_start_end: a list of all the possible [window_start, window_end] pairs.

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
    frame_len = 1000.0 / fps
    frames = motion.shape[0]
    len_ms = frames * frame_len
    timestamps_int = np.arange(
        start_time + offset_sec * 1000,
        start_time + offset_sec * 1000 + len_ms,
        frame_len,
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
    return df_flow, len_ms


def load_sensors_cubic(
    raw_path, sub, device, sensors, sensor_col_header, start_time, end_time, fps
):
    df_list = []
    for s, col in zip(sensors, sensor_col_header):
        df_sensor = read_data_datefolder_hourfile(
            raw_path, sub, device, s, start_time, end_time
        )
        df_sensor = df_sensor[["time", col]]
        df_sensor["time"] = pd.to_datetime(df_sensor["time"], unit="ms")
        df_sensor = df_sensor.set_index("time")
        df_resample = df_sensor.resample(FRAME_INTERVAL).mean()  
        # FRAME_INTERVAL as 0.03336707S is the most closest value to 1/29.969664 pandas accepts
        df_resample = df_resample.interpolate(method="spline", order=3) # cubic spline interpolation
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


def add_win_rand_offset(
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
    fps
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
            # match video dataframe
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
            df_window_flow.loc[:, "time"] = df_window_flow.loc[:, "time"] \
                - pd.Timedelta(offset, unit="ms")
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
                )  # relatively video name, sensor starttime, sensor endtime, video offset
    return cnt_windows, df_dataset_vid, info_dataset_vid


def seg_smk_video(
    subject,
    video,
    window_size_sec,
    stride_sec,
    offset_sec,
    kde_num_offset,
    kde_max_offset,
    window_criterion,
    starttime_file,
    fps,
):
    # ==========================================================================================
    reliability_resample_path = settings['reliability_resample_path']
    raw_path = settings['raw_path']
    flow_path = settings['flow_path']
    # ==========================================================================================

    vid_name = subject + " " + video
    vid_qual_win_cnt = []
    df_dataset = []
    info_dataset = []

    device = "CHEST"
    sensor = "ACCELEROMETER"
    sensors = ["ACCELEROMETER_X", "ACCELEROMETER_Y", "ACCELEROMETER_Z"]
    sensor_col_header = ["accx", "accy", "accz"]
    flow_path = os.path.join(
        "../../data/flow_pwc/sub{}".format(subject), vid_name + ".pkl"
    )

    # load start end time
    start_time = load_start_time(starttime_file, vid_name)

    # load optical flow data and assign unixtime to each frame
    df_flow, len_ms = load_flow(flow_path, fps, start_time, offset_sec)
    end_time = int(start_time) + int(len_ms)

    # load sensor reliability data
    df_sensor_rel = read_data_datefolder_hourfile(
        reliability_resample_path,
        subject,
        device,
        sensor + "_reliability",
        start_time,
        end_time,
    )

    # record consecutive seconds of the length the same as window_size
    win_start_end = reliability_df_to_consecutive_seconds(
        df_sensor_rel, window_size_sec, stride_sec, threshold=7
    )

    ## extract the optical flow frames of the good seconds according to sensor data
    df_flow["time"] = pd.to_datetime(df_flow["time"], unit="ms")
    df_flow = df_flow[["flowx", "flowy", "diff_flowx", "diff_flowy", "time"]].set_index(
        "time"
    )

    # extract the raw data 'ACCELEROMETER_X' (,'ACCELEROMETER_Y', 'ACCELEROMETER_Z') of consecutive chunk and resample
    #   according to video frame timestamp.
    # note that we need to resample sensor data according to video time,
    #   so the input of resample function here should be raw data instead of already sampled data to avoid resample twice.

    df_sensors = load_sensors_cubic(
        raw_path,
        subject,
        device,
        sensors,
        sensor_col_header,
        start_time,
        end_time,
        fps,
    )

    # concatenate df_sensors and df_flow
    df_resample = pd.merge_asof(
        df_flow,
        df_sensors,
        on="time",
        tolerance=pd.Timedelta("30ms"),
        direction="nearest",
    ).set_index("time")

    df_resample = df_resample.dropna(how="any")
    # print('There are {} points left (freq=30Hz) for sensor-opticalflow combined.\n'.format(len(df_resample)))
    df_sensor = df_resample[["accx", "accy", "accz"]].reset_index()
    df_flow = df_resample[["flowx", "flowy", "diff_flowx", "diff_flowy"]].reset_index()

    # PCA with zero-mean the data
    df_sensor, df_flow = pca_sensor_flow(df_sensor, df_flow)

    ## select anchor windows from sensor, apply shifts in videos
    cnt_windows, df_dataset_vid, info_dataset_vid = add_win_rand_offset(
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
        fps
    )
    print(cnt_windows, "/", len(win_start_end), "windows left for video ", vid_name)
    df_dataset += df_dataset_vid
    info_dataset += info_dataset_vid
    vid_qual_win_cnt.append((vid_name, cnt_windows))

    return vid_qual_win_cnt, df_dataset, info_dataset


def segment_video_all(
    window_size_sec,
    stride_sec,
    offset_sec,
    kde_num_offset,
    kde_max_offset,
    window_criterion,
    data_dir,
    starttime_file,
    fps=FPS,
):
    df_start_time = csv_read(starttime_file).set_index("video_name")
    video_names = df_start_time.index.tolist()

    df_dataset_all = []
    info_dataset_all = []
    vid_qual_win_cnt_all = []

    for video_name in video_names:
        subject = video_name[:3]
        video = video_name[4:]
        vid_qual_win_cnt, df_dataset, info_dataset = seg_smk_video(
            subject=subject,
            video=video,
            window_size_sec=window_size_sec,
            stride_sec=stride_sec,
            offset_sec=offset_sec,
            kde_num_offset=kde_num_offset,
            kde_max_offset=kde_max_offset,
            window_criterion=window_criterion,
            starttime_file=starttime_file,
            fps=fps,
        )

        vid_qual_win_cnt_all += vid_qual_win_cnt
        df_dataset_all += df_dataset
        info_dataset_all += info_dataset

    title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
    )
    print(
        len(vid_qual_win_cnt_all),
        "videos with valid window(s), # of qualified windows: ",
        vid_qual_win_cnt_all,
    )
    pd.DataFrame(vid_qual_win_cnt_all, columns=["vid_name", "window_num"]).to_csv(
        "./data/num_valid_windows" + title_suffix + ".csv", index=None
    )
    with open(
        os.path.join(data_dir, "all_video" + title_suffix + "_df_dataset.pkl"), "wb"
    ) as handle:
        pickle.dump(df_dataset_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
        os.path.join(data_dir, "all_video" + title_suffix + "_info_dataset.pkl"), "wb"
    ) as handle:
        pickle.dump(info_dataset_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return df_dataset_all, info_dataset_all


def load_windows(
    window_size_sec,
    stride_sec,
    offset_sec,
    kde_num_offset,
    kde_max_offset,
    window_criterion,
    data_dir
    ):
    title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
    )
    pd_window_num = pd.read_csv("./data/num_valid_windows" + title_suffix + ".csv")
    print(pd_window_num)
    with open(os.path.join(data_dir, "all_video" + title_suffix + "_df_dataset.pkl"), "rb") as handle:
        df_dataset = pickle.load(handle)
    print(len(df_dataset))
    with open(os.path.join(data_dir, "all_video" + title_suffix + "_info_dataset.pkl"), "rb") as handle:
        info_dataset = pickle.load(handle)
    print(len(info_dataset))
    return df_dataset, info_dataset, pd_window_num

