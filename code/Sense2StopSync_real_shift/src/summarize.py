import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from settings import settings
from utils import list_files_in_directory


def get_test_videos():
    return pd.read_csv(settings["STARTTIME_TEST_FILE"])["video_name"].tolist()


def get_test_videos_boundary_removed():
    """

    Returns:

    """
    test_videos = pd.read_csv(settings["STARTTIME_TEST_FILE"])["video_name"].tolist()
    boundary_set = pd.read_csv(settings["LOW_QUALITY_VIDEOS"])["video_name"].tolist()
    boundary_removed = [vid for vid in test_videos if vid not in boundary_set]
    print([vid for vid in test_videos if vid in boundary_set])
    return boundary_removed


def read_batch_final_result(folder, file):
    """

    Args:
        folder:
        file:

    Returns:

    """
    result_df = pd.read_csv(os.path.join(folder, file))
    offset_sec = float(file.split("_offset",1)[1].split("_rdoffset",1)[0])
    num_videos = len(result_df)
    error_abs = abs((result_df['offset'] + offset_sec*1000).to_numpy())
    num_segs = result_df['num_segs'].to_numpy()
    try:
        ave_offset = np.mean(error_abs[~np.isnan(error_abs.astype(np.float))])
    except ZeroDivisionError:
        ave_offset = 0
        print('Error @offset_sec:', offset_sec)
    try:        
        ave_num_segs = np.mean(num_segs[~np.isnan(num_segs.astype(np.float))])
        PV1000 = len(error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 1000)])/float(num_videos)
        PV700 = len(error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 700)])/float(num_videos)
        PV300 = len(error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 300)])/float(num_videos)
        conf = 200000 / (result_df['sigma'].to_numpy() * result_df['mu_var'].to_numpy())
        # dive into the problem there are two >10000 cases
        ave_conf = np.mean(conf[~np.isnan(conf.astype(np.float)) & (conf<10000)]) 
        return ave_offset, PV1000, PV700, PV300, ave_conf, num_videos, ave_num_segs, conf, offset_sec
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, offset_sec


def summarize_shift_per_video(vid, ini_off, max_shift_bidir, in_folder, out_folder):
    """

    Args:
        vid:
        ini_off:
        max_shift_bidir:
        in_folder:
        out_folder:

    Returns:

    """
    window_size_sec = 10
    stride_sec = 1
    window_criterion = 0.8
    kde_num_offset = 20
    kde_max_offset = 5000
    kde_max_offset_list, ave_offset_list = [], []
    PV300_list, PV700_list, PV1000_list = [], [], []
    ave_conf_list, ave_num_segs_list = [], []
    conf_list, offset_secs = [], []
    first_time_flag = 1

    # read all files in $in_folder$
    files = list_files_in_directory(in_folder)
    files = [f for f in files if f.startswith("final_result_per_video")]
    print(vid)

    # note: need to clear folder "./result" each time before running
    range_sec_bidir = settings["range_sec_bidir"]
    step_sec = settings["step_sec"]
    maxfiles = 2 * range_sec_bidir / step_sec + 1
    print("max number of files: ", maxfiles)
    assert len(files) <= maxfiles
    files.sort()

    for file in files:
        ave_offset, PV1000, PV700, PV300, ave_conf, _, ave_num_segs, conf, offset_sec = read_batch_final_result(
            in_folder, file)
        kde_max_offset_list.append(kde_max_offset)
        ave_offset_list.append(ave_offset)
        PV1000_list.append(PV1000)
        PV700_list.append(PV700)
        PV300_list.append(PV300)
        ave_conf_list.append(ave_conf)
        ave_num_segs_list.append(ave_num_segs)
        if np.isnan(conf):
            conf_list.append(np.nan)
        else:
            conf_list.append(conf[0])
        offset_secs.append(offset_sec)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    summ_path = out_folder+'/final_shift_result_win{}_str{}_rdoffset{}_wincrt{}_pca_sigma500_1videos.csv'.\
                format(window_size_sec, stride_sec, kde_num_offset, window_criterion)
    data = {'offset_sec': offset_secs, 'kde_max_offset': kde_max_offset_list, 'ave_offset': ave_offset_list, \
            'PV1000': PV1000_list, 'PV700': PV700_list, 'PV300': PV300_list, 'ave_num_segs': ave_num_segs_list,\
            'conf':conf_list
        }
    data_df = pd.DataFrame(data, columns = ['offset_sec','kde_max_offset','ave_offset','PV1000','PV700','PV300',\
        'ave_num_segs','conf']).sort_values("offset_sec")
    data_df.to_csv(summ_path, index=None)


def summarize_shift_all_videos(result_dir):
    """

    Args:
        result_dir:

    Returns:

    """
    videos = get_test_videos()
    vid_shift = pd.read_csv(orig_shift_file, names=["videos", "offsets"])
    vid_shift = vid_shift.set_index("videos")
    offsets = []
    for vid in videos:
        offsets.append(vid_shift.loc[vid].values[0])
    subdirs = [o for o in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir,o))]
    print(len(subdirs))
    
    for vid, ini_off in zip(videos, offsets):
        if 'result' + vid in subdirs:
            summarize_shift_per_video(vid, ini_off, max_shift_bidir=180, in_folder=result_dir+'/result'+vid, 
                out_folder=result_dir+'/result_summary_'+vid)
        else:
            print('result for {} not exists.'.format(vid))


def success_Ntrials(metric_ms, result_dir, videos):
    """

    Args:
        metric_ms:
        result_dir:
        videos:

    Returns:

    """
    total = len(videos)
    success_1, success_2, success_3, success_4, success_5, success_6 = 0, 0, 0, 0, 0, 0
    success_7, success_8, success_9, success_10, no_file = 0, 0, 0, 0, 0
    success1_offsets, success2_offsets, success3_offsets, success4_offsets = [], [], [], []
    success5_offsets, success6_offsets, success7_offsets, success8_offsets = [], [], [], []
    success9_offsets, success10_offsets = [], []

    for vid in videos:
        folder = result_dir + '/result_summary_'+vid
        try:
            df= pd.read_csv(folder+'/final_shift_result_win10_str1_rdoffset20_wincrt0.8_pca_sigma500_1videos.csv')
            df_new = df.sort_values(by='conf', ascending=False)

            if df_new['ave_offset'].iloc[0] < metric_ms:
                success_1 += 1
                success1_offsets.append(df_new['ave_offset'].iloc[0])

            elif df_new['ave_offset'].iloc[1] < metric_ms:
                success_2 += 1
                success2_offsets.append(df_new['ave_offset'].iloc[1])

            elif df_new['ave_offset'].iloc[2] < metric_ms:
                success_3 += 1
                success3_offsets.append(df_new['ave_offset'].iloc[2])

            elif df_new['ave_offset'].iloc[3] < metric_ms:
                success_4 += 1
                success4_offsets.append(df_new['ave_offset'].iloc[3])

            elif df_new['ave_offset'].iloc[4] < metric_ms:
                success_5 += 1
                success5_offsets.append(df_new['ave_offset'].iloc[4])

            elif df_new['ave_offset'].iloc[5] < metric_ms:
                success_6 += 1
                success6_offsets.append(df_new['ave_offset'].iloc[5])

            elif df_new['ave_offset'].iloc[6] < metric_ms:
                success_7 += 1
                success7_offsets.append(df_new['ave_offset'].iloc[6])

            elif df_new['ave_offset'].iloc[7] < metric_ms:
                success_8 += 1
                success8_offsets.append(df_new['ave_offset'].iloc[7])

            elif df_new['ave_offset'].iloc[8] < metric_ms:
                success_9 += 1
                success9_offsets.append(df_new['ave_offset'].iloc[8])

            elif df_new['ave_offset'].iloc[9] < metric_ms:
                success_10 += 1
                success10_offsets.append(df_new['ave_offset'].iloc[9])

            else: 
                print('fail ', vid)

        except:
            no_file += 1
            print(vid, ' not existing! ', vid)

    failed = total - no_file - success_1 - success_2 - success_3 - success_4 - success_5 - success_6 - success_7 -\
     success_8 - success_9- success_10    
    print('\n\nMetric: PV{}'.format(metric_ms))
    print('success_1: ', success_1, success_1/(total - no_file))
    print('success_2: ', success_2, success_2/(total - no_file))
    print('success_3: ', success_3, success_3/(total - no_file))
    print('success_4: ', success_4, success_4/(total - no_file))
    print('success_5: ', success_5, success_5/(total - no_file))
    print('success_6: ', success_6, success_6/(total - no_file))
    print('success_7: ', success_7, success_7/(total - no_file))
    print('success_8: ', success_8, success_8/(total - no_file))
    print('success_9: ', success_9, success_9/(total - no_file))
    print('success_10: ', success_10, success_10/(total - no_file))
    print('Failed: ', failed, failed/(total - no_file))
    print('no file: ', no_file)
    print('num:', len(videos))
    print('One time PV{}: '.format(metric_ms), success_1/(total - no_file))
    print('Three time PV{}: '.format(metric_ms), (success_1 + success_2 + success_3)/(total - no_file))
    print('Five time PV{}:  '.format(metric_ms), (success_1 + success_2 + success_3 + success_4 + success_5)/(total - no_file))
    print('Ten time PV{}:   '.format(metric_ms), 1 - failed/(total - no_file), '\n\n')

    # plot and save cumulative distribution function (CDF) 
    topK = 10
    Y = np.array([success_1,success_2,success_3,success_4,success_5,success_6,success_7,success_8,success_9,success_10])
    Y = Y/len(videos)*100
    Y_cdf = [Y[0]]
    for i in range(1, topK):
        Y_cdf.append(Y_cdf[-1] + Y[i])
    X = np.array(list(range(1, topK + 1)))
    fig, ax= plt.subplots(figsize =(4.5, 2.2))
    ax.plot(X, Y_cdf, 'k.')
    ax.plot(X, Y_cdf, 'k', linewidth=2.0)
    plt.xlabel('Top K Proposal Examined')
    plt.ylabel('PV-300')
    plt.ylim([0, 100])
    # locs, labels = plt.xticks()
    plt.xticks(np.arange(1, topK + 1, step=1)) 
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    plt.savefig('cdf_PV{}_{}.eps'.format(metric_ms, total), dpi=fig.dpi)
    # plt.show()


if __name__ == '__main__':
    orig_shift_file = settings["ORIGIN_SHIFT_FILE"]
    # result_dir = "../final_result_real/final_all_1st"
    result_dir = "final"
    
    summarize_shift_all_videos(result_dir)
    
    print("Before removing low-quality videos:")
    success_Ntrials(300, result_dir, videos=get_test_videos())

    print("After removing low-quality videos:")
    success_Ntrials(300, result_dir, videos=get_test_videos_boundary_removed())

