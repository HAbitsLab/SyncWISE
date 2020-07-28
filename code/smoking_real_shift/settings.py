import os
import pytz

# TODO: all single ' to double "
settings = {}
# settings['TEMP_DIR'] = 'tmp_data'
settings['TEMP_DIR'] = '/Users/szh702/Documents/sync_data'
settings['TIMEZONE'] = pytz.timezone('America/Chicago')
settings['FPS'] = 29.969664
settings["FRAME_INTERVAL"] = "0.03336707S"

settings['STARTTIME_FILE'] = "start_time.csv"
settings['STARTTIME_VAL_FILE'] = "start_time_val.csv"
settings['STARTTIME_TEST_FILE'] = "start_time_test.csv"
settings["ORIGIN_SHIFT_FILE"] = 'video_shift_orig_trial.txt'
settings["TASK_LIST_FILE"] = 'temp_file_list_r_orig_trial.txt'

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/")
settings["DATA_DIR"] = DATA_DIR
settings['reliability_resample_path'] = os.path.join(DATA_DIR, "RESAMPLE200/wild/")
settings['raw_path'] = os.path.join(DATA_DIR, "RAW/wild/")
settings['flow_path'] = os.path.join(DATA_DIR, "flow_pwc/")
settings["qualified_window_num"] = 200
settings["range_sec_bidir"] = 180
settings["step_sec"] = 3
settings["window_size_sec"] = 10 
settings["window_criterion"] = 0.8
settings["kde_max_offset"] = 5000
settings["kde_num_offset"] = 20

# settings['file_list'] = "file_list_test.txt"