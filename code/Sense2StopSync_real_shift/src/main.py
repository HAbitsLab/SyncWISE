import sys

from segment_video import segment_video
from calc_video_offset_all import calc_drift_all_windows
from settings import settings

import random

if __name__ == "__main__":
    random.seed(0)

    parameter_str = sys.argv[1]
    (
        window_size_sec,
        window_criterion,
        kde_max_offset,
        kde_num_offset,
        offset_sec,
        subject,
        video,
    ) = parameter_str.split(" ")
    window_size_sec = int(window_size_sec)
    window_criterion = float(window_criterion)
    kde_max_offset = int(kde_max_offset)
    kde_num_offset = int(kde_num_offset)
    if float(offset_sec).is_integer():
        offset_sec = int(float(offset_sec))
    else:
        offset_sec = float(offset_sec)
    stride_sec = 1
    full_vid_name = subject + " " + video

    # segment videos into windows
    df_dataset, info_dataset = segment_video(
        full_vid_name,
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        window_criterion,
        kde_max_offset,
        29.969664
    )

    # calculate drift for all the windows
    calc_drift_all_windows(
        df_dataset, 
        info_dataset,
        full_vid_name,
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
        qualified_window_num=settings["qualified_window_num"] * kde_num_offset,
        result_dir="final"  # TOOD: change this to "result"
    )
