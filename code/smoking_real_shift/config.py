import pandas as pd
import random

from settings import settings


startime_file = settings["STARTTIME_FILE"]
val_set_ratio = settings["val_set_ratio"]

videos_df = pd.read_csv(startime_file)
len_all= len(videos_df)
len_val= int(round(len_all*val_set_ratio))
random.seed(80)
# indices for validation set (sensitivity study)
val = random.sample(range(0, len_all), len_val)

videos_df = pd.read_csv(settings["STARTTIME_FILE"])

# save start time file for validation videos 
val_videos_df = videos_df.iloc[val]
val_videos_df = val_videos_df.sort_values("video_name")
val_videos_df.to_csv(settings["STARTTIME_VAL_FILE"], index=None)

# save start time file for test videos
test_videos_df = videos_df.iloc[~videos_df.index.isin(val)]
test_videos_df = test_videos_df.sort_values("video_name")
test_videos_df.to_csv(settings["STARTTIME_TEST_FILE"], index=None)

# read original shift for all videos
orig_shift_file = settings["ORIGIN_SHIFT_FILE"]
orig_shift = pd.read_csv(orig_shift_file, names=["videos", "offsets"])
orig_shift = orig_shift.set_index('videos')

task_list_file = settings["TASK_LIST_FILE"]
window_size_sec = settings["window_size_sec"]
window_criterion = settings["window_criterion"]
kde_max_offset = settings["kde_max_offset"]
kde_num_offset = settings["kde_num_offset"]
step_sec = settings["step_sec"]

# create task list file
f = open(task_list_file,"w+")
test_videos = test_videos_df["video_name"].tolist()
for vid in test_videos:
    offset  = orig_shift.loc[vid].values[0]
    range_bidir = settings["range_sec_bidir"]
    step = settings["step_sec"]
    for i in range(int(2 * range_bidir / step) + 1):
        f.write('{} {} {} {} {} {}\n'.format(
            window_size_sec, 
            window_criterion, 
            kde_max_offset, 
            kde_num_offset,
            -range_bidir + i * step_sec + offset, 
            vid
        ))
