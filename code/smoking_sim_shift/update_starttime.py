# TODO: remove this script

# NOTE:
#   run this script whenever change in ANNOTATION occurs

import os
import re
import sys
import json
import time
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time, date
from shutil import copyfile
from utils import datetime_to_unixtime, \
                    parse_timestamp_tz_naive, \
                    list_files_in_directory, \
                    truncate_df_index_dt,\
                    truncate_df_index_str,\
                    parse_timestamp_tz_aware,\
                    csv_read
from settings import settings
from itertools import groupby


def backup_start_time_csv(start_time_file='../../data/start_time.csv'):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    copyfile(start_time_file, start_time_file[:-4]+dt_string+'.csv')
    return 0

def read_start_time_file(start_time_file='../../data/start_time.csv'):
    df_start_time = csv_read(start_time_file)
    return df_start_time

def read_sync_yaml_file(subject, anno_dir='../../data/ANNOTATION/inwild'):
    sync_path = os.path.join(anno_dir, subject, 'sync.yaml')
    with open(sync_path) as f:
        sync = yaml.load(f)
    return sync

def calc_start_time(sync, episode):
    # get start time and end time
    sync_relative = sync[episode]['sync_relative']
    sync_absolute = sync[episode]['sync_absolute']
    video_lead_time = sync[episode]['video_lead_time']
    # print("video_lead_time: ", video_lead_time)
    sync_absolute = parse_timestamp_tz_aware(sync_absolute)
    t = datetime.strptime(sync_relative,"%H:%M:%S")
    startTime = sync_absolute - timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)\
                             + timedelta(seconds=video_lead_time/1000)
    return startTime

def update_starttime(start_time_file='../../data/start_time.csv'):
    video_name_list = []
    start_time_list = []
    start_df = read_start_time_file(start_time_file)
    video_list = start_df['video_name'].tolist()
    # make a copy of original start_time.csv as start_time_
    # datetime object containing current date and time
    # backup_start_time_csv(start_time_file)
    # get subject list from start_time.csv
    data_dir = os.path.dirname(start_time_file)
    anno_dir = os.path.join(data_dir, 'ANNOTATION', 'inwild')
    for subject, group in groupby(video_list, lambda x: x[:3]):
        # print(subject)
        sync = read_sync_yaml_file(subject, anno_dir)
        for item in group:
            episode = item[4:]
            # print(subject, episode)
            video_name_list.append(subject+' '+episode)
            start_time_list.append(datetime_to_unixtime(calc_start_time(sync, episode)))
    df = pd.DataFrame({'video_name': video_name_list, 'start_time': start_time_list})
    df.to_csv(start_time_file, index=None)


if __name__ == '__main__':
    # print('Time zone:', settings['TIMEZONE'])
    
    STARTTIME_FILE = '../../data/start_time.csv'
    update_starttime(STARTTIME_FILE)
