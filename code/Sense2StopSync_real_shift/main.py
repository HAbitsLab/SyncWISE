import sys

from segment_smk_video import seg_smk_video
from calc_drift_MD2K_all_windows import calc_drift_all_windows


if __name__ == "__main__":

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

    # TODO: explanations
    df_dataset, info_dataset = seg_smk_video(
        full_vid_name,
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        window_criterion,
        kde_max_offset,
        29.969664
    )

    # TODO: explanations
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
        result_dir="final"
    )
