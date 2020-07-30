import os
import sys
from seg_smk_video_all import seg_smk_video_all
from calc_video_offset_all import calc_video_offset_all
from settings import settings


if __name__ == "__main__":
    starttime_file = settings['STARTTIME_VAL_FILE']
    data_dir = os.path.join(settings["TEMP_DIR"], 'sanity')
    stride_sec = settings["STRIDE_SEC"]

    parameter_str = sys.argv[1]
    (
        window_size_sec,
        window_criterion,
        kde_max_offset,
        kde_num_offset,
        offset_sec
    ) = parameter_str.split(" ")
    window_size_sec = int(window_size_sec)
    offset_sec = float(offset_sec)
    # note that when converted to float, an integer will be appended a '.0'
    kde_num_offset = int(kde_num_offset)
    kde_max_offset = int(kde_max_offset)
    window_criterion = float(window_criterion)

    df_dataset_all, info_dataset_all = seg_smk_video_all(
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
        data_dir,
        starttime_file,
    )

    calc_video_offset_all(
        df_dataset_all, 
        info_dataset_all,
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
        qualified_window_num=10*kde_num_offset, 
        save_dir='./result/summary_ablation',
        pca=1
    )



