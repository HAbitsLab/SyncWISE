import pandas as pd
import random
from settings import settings

# from update_starttime import update_starttime

startime_file = settings["STARTTIME_FILE"]
# print(startime_file)
# update_starttime(startime_file)

videos_df = pd.read_csv(startime_file)
len_all= len(videos_df)
len_val= int(round(len_all*0.2))
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
