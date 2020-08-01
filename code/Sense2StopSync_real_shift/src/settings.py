import os
import pytz


settings = {}
settings["TEMP_DIR"] = "tmp_data"
settings["TIMEZONE"] = pytz.timezone("America/Chicago")
settings["FPS"] = 29.969664
settings["FRAME_INTERVAL"] = "0.03336707S"

settings["STARTTIME_FILE"] = "start_time.csv"
settings["STARTTIME_VAL_FILE"] = "start_time_val.csv"
settings["STARTTIME_TEST_FILE"] = "start_time_test.csv"
settings["ORIGIN_SHIFT_FILE"] = "video_shift_orig_trial.txt"
settings["TASK_LIST_FILE"] = "task_list_orig_offset.txt"
settings["LOW_QUALITY_VIDEOS"] = "low_quality_videos.csv"

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../Sense2StopSync/")
settings["DATA_DIR"] = DATA_DIR
settings["reliability_resample_path"] = os.path.join(DATA_DIR, "RESAMPLE200/wild/") #TODO: last step to change name to RESAMPLE
settings["raw_path"] = os.path.join(DATA_DIR, "RAW/wild/") #TODO: last step to remove wild
settings["flow_path"] = os.path.join(DATA_DIR, "flow_pwc/")
settings["qualified_window_num"] = 10
settings["range_sec_bidir"] = 180  # looping search range is [-range_sec_bidir, range_sec_bidir] (seconds)
settings["step_sec"] = 1.5
settings["window_size_sec"] = 10
settings["window_criterion"] = 0.8
settings["kde_max_offset"] = 5000
settings["kde_num_offset"] = 20
settings["sample_counts"] = 7
settings["val_set_ratio"] = 0.2
settings["kernel_var"] = 500
