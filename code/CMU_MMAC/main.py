from AbsErrorROC import gaussianVoting
from resample import resample
from load_time import *
import pickle
from cross_correlation_func import cross_correlation_using_fft, compute_shift
from sklearn.decomposition import PCA
import glob
import random
from statistics import mean, stdev, median
import time

def pca_sensor(df_sensor):
    pca_sensor = PCA(n_components=1)
    df_sensor[['Accel_X', 'Accel_Y', 'Accel_Z']] -= df_sensor[['Accel_X', 'Accel_Y', 'Accel_Z']].mean()
    df_sensor['Accel_PCA'] = pca_sensor.fit_transform( df_sensor[['Accel_X', 'Accel_Y', 'Accel_Z']].to_numpy())
    return df_sensor

def pca_flow(df_flow):
    diffflow_mat = df_flow[['diff_flowx', 'diff_flowy']].to_numpy()
    diffflow_mat -= np.mean(diffflow_mat, axis=0)
    pca_diffflow = PCA(n_components=1)
    df_flow['diff_flowPCA'] = pca_diffflow.fit_transform(diffflow_mat)
    return df_flow    

def drift_confidence(ts1, ts2):
    fftshift = cross_correlation_using_fft(ts1, ts2)
    dist = max(abs(fftshift-median(fftshift)))
    shift = compute_shift(fftshift)
    conf = dist/stdev(fftshift)
    return conf, shift

def compute_error(offsets, shifts):
    sensors = shifts.keys()
    error = {}
    FPS =30
    for sensor in sensors:
        valid_index = ~np.isnan(shifts[sensor])
        num_vids = sum(valid_index)
        abs_error = abs(offsets[valid_index] - shifts[sensor][valid_index]* 1000 /FPS)
        PC300 = sum(abs_error < 300)/num_vids * 100
        PC700 = sum(abs_error < 700)/num_vids * 100
        error[sensor] = [sum(abs_error)/num_vids, PC300, PC700, num_vids]
        df_error = pd.DataFrame.from_dict(error, orient='index', columns=['Abs Error', 'PC300', 'PC700', 'num vids'])
    return df_error


def baseline(mode):
    # load video data
    # session_list = open('../../CMU/session_list').readlines()
    video_dir = '../../CMU/video'
    opt_dir = '../../CMU/opt_flow'
    IMU_dir = '../../CMU/sensor/'
    data_dir = '../../CMU/data'
    FPS = 30
    baseline_dir = os.path.join(data_dir, 'baseline')
    # offsets = np.zeros(len(session_list))
    sensors = ['2794', '2795', '2796', '3261', '3337']
    video = '7150991'
    load_df = True
    # verbose = True
    draw = True
    sensor_dict = {'2794': 'Left Arm', '2795': 'Back', '2796': 'Left Leg', '3261': 'Right Leg', '3337': 'Right Arm'}
    #########################################################
    ##                   Baseline                           #
    #########################################################
    valid_session_file = '../../CMU/valid_sessions_win30_max60.pkl'
    offsets, session_list = pickle.load(open(valid_session_file, 'rb'))
    session_list = session_list['2794']
    if mode == 'x':
        mode_video = 'x'
        mode_imu = 'X'
    if mode == 'PCA':
        mode_video = 'PCA'
        mode_imu = 'PCA'
    shifts_baseline = {}
    skipped_sessions = {}
    valid_sessions = {}
    for sensor in sensors:
        shifts_baseline[sensor] = np.zeros(len(session_list))
        skipped_sessions[sensor] = []
        valid_sessions[sensor] = []
        os.makedirs('figures/baseline_{}_{}/{}'.format(mode_video, mode_imu, sensor), exist_ok=True)
    # load ground truth
    offsets = np.zeros(len(session_list))
    for i, session in enumerate(session_list):
        session = session.strip()
        opt_file = glob.glob(os.path.join(opt_dir, session+'_Video', session+'_7150991-*.pkl'))
        if len(opt_file) > 0:
            opt_file = opt_file[0]
            offsets[i] = int(opt_file[opt_file.find('-')+1:-4])
    offsets = offsets * 1000/FPS
    for i, session in enumerate(session_list):
        session = session.strip()
        print('======== Procession session {} ========'.format(session))
        file = glob.glob(os.path.join(video_dir, session+'_Video', 'STime{}-time-*synch.txt'.format(video)))
        out_dir = os.path.join(baseline_dir, session)
        os.makedirs(out_dir, exist_ok=True)
        df_file = os.path.join(out_dir, 'df_video_{}.pkl'.format(video))
        print('Processing video {}'.format(video))
        if load_df and os.path.exists(df_file):
            print('    loading video df')
            df_video = pd.read_pickle(df_file)
        else:
            print('    creating video df')
            if file:
                df_video = read_video(file[0])
                check_df(df_video, delta=0.033332)
            else:
                print('skipping session{}, video sync doesn\'t exist'.format(session))
                for sensor in sensors:
                    shifts_baseline[sensor][i] = np.nan
                    skipped_sessions[sensor].append(session)
                continue
            opt_file = glob.glob(os.path.join(opt_dir, session+'_Video', session+'_7150991-*.pkl'))
            if opt_file:
                opt_file = opt_file[0]
                #offsets[i] = int(opt_file[opt_file.find('-')+1:-4])
            else:
                print('skipping session{}, optical flow doesn\'t exist'.format(session))
                for sensor in sensors:
                    shifts_baseline[sensor][i] = np.nan
                    skipped_sessions[sensor].append(session)
                continue
            motion = pickle.load(open(opt_file, 'rb'))
            if len(df_video) < len(motion):
                print('skipping session{}, sync stamps less than frames'.format(session))
                for sensor in sensors:
                    shifts_baseline[sensor][i] = np.nan
                    skipped_sessions[sensor].append(session)
                continue                
            df_video = df_video[:len(motion)]
            df_video['flowx'] = motion[:, 0]
            df_video['flowy'] = motion[:, 1]
            df_video['diff_flowx'] = df_video['flowx'].diff()
            df_video['diff_flowy'] = df_video['flowy'].diff()
            df_video = df_video[1:]
            df_video = pca_flow(df_video)
            # df_video.to_pickle(df_file)
        # load sensor data
        for sensor in sensors:
            print('Processing sensor {}'.format(sensor))
            df_file = os.path.join(out_dir, 'df_sensor_{}.pkl'.format(sensor))
            if os.path.exists(df_file):
                print('    loading sensor df')
                df_imu = pd.read_pickle(df_file)
            else:
                print('    creating sensor df')
                sensor_file = glob.glob(os.path.join(IMU_dir, session+'_3DMGX1', '{}_*-time*.txt'.format(sensor)))
                if sensor_file:
                    sensor_file = sensor_file[0]
                else:
                    print('skipping session{}, session time doesn\'t exist'.format(session))
                    skipped_sessions[sensor].append(session)
                    shifts_baseline[sensor][i] = np.nan  
                    continue
                df_imu = read_sensor(sensor_file)
                df_imu = check_df(df_imu, delta=0.008)
                df_imu = pca_sensor(df_imu)
                # df_imu.to_pickle(df_file)
            st_time = max([df_imu.iloc[0]['SysTime'], df_video.iloc[0]['SysTime']])-0.01
            en_time = min([df_imu.iloc[len(df_imu) - 1]['SysTime'], df_video.iloc[len(df_video) - 1]['SysTime']])+0.01
            df_video_tmp = df_video[(df_video['SysTime'] >= st_time) & (df_video['SysTime'] < en_time)]
            df_imu = df_imu[(df_imu['SysTime'] >= st_time) & (df_imu['SysTime'] < en_time)]
            vid_time_stamps = df_video_tmp['SysTime'].values
            df_imu = resample(df_imu, 'SysTime', samplingRate=0,
                                          gapTolerance=200, fixedTimeColumn=vid_time_stamps)
            if df_imu is None:
                print('    no intersection between video and imu, skip session {}'.format(session))
                skipped_sessions[sensor].append(session)
                shifts_baseline[sensor][i] = np.nan 
                continue
            if len(df_video_tmp) != len(df_imu):
                print('    lengths of video and imu not equal, skip session {}'.format(session))
                print('len_vid = {}, len_sen = {}'.format(len(df_video_tmp), len(df_imu)))
                skipped_sessions[sensor].append(session)
                shifts_baseline[sensor][i] = np.nan 
                #df_imu.to_pickle(df_file)
                continue            

            fftshift = cross_correlation_using_fft(df_video_tmp['diff_flow{}'.format(mode_video)].values, \
                                                   df_imu['Accel_{}'.format(mode_imu)].values)
            shifts_baseline[sensor][i] = compute_shift(fftshift)
            if draw:
                path ='figures/baseline_{}_{}/{}/{}.jpg'.format(mode_video, mode_imu, sensor, session)
                plt.figure()
                plt.plot(fftshift[::-1])
                plt.title('Video / {} Sensor, gt = {:.3f}s, predition = {:.3f}s'.format(sensor_dict[sensor], offsets[i]/1000, shifts_baseline[sensor][i] /FPS))
                plt.savefig(path)
                plt.close()  
            valid_sessions[sensor].append(session)
            #plt.figure()
            #plt.plot(fftshift)
            #plt.savefig('{}_{}_{}.png'.format(mode, session, sensor))
    for sensor in sensors:
        shifts_baseline[sensor][67:91] = np.nan    
    error = compute_error(offsets, shifts_baseline)
    error.to_csv(os.path.join(baseline_dir, 'df_error_baseline_{}_{}.csv'.format(mode_video, mode_imu)))
    print(mode_video, mode_imu)
    print(error)
    #print(offsets, shifts_baseline)
    # pickle.dump([offsets, shifts_baseline], open(os.path.join(baseline_dir, 'results_baseline_{}_{}.pkl'.format(mode_video, mode_imu)), 'wb'))
    # pickle.dump(valid_sessions, open(os.path.join(baseline_dir, 'valid_sessions.pkl'), 'wb'))
    #
    # Analysis
    _, shifts_baseline = pickle.load(open(os.path.join(baseline_dir, 'results_baseline_{}_{}.pkl'.format(mode_video, mode_imu)), 'rb'))
    summ_mat = offsets.reshape(-1, 1)
    for sensor in sensors:
        summ_mat = np.concatenate([summ_mat, shifts_baseline[sensor].reshape(-1, 1) - offsets.reshape(-1, 1)], axis=1)
    df_summ = pd.DataFrame(data=summ_mat, columns=['Ground truth', 'imu2794', 'imu2795', 'imu2796', 'imu3261', 'imu3337'], index=session_list)
    df_summ.to_csv(os.path.join(baseline_dir, 'df_summ_baseline_{}_{}.csv'.format(mode_video, mode_imu)))
    print(offsets)
    

def syncwise():
    # load video data
    # session_list = open('../../CMU/session_list').readlines()
    # video_dir = '../../CMU/video'
    opt_dir = '../../CMU/opt_flow'
    # IMU_dir = '../../CMU/sensor/'
    data_dir = '../../CMU/data'
    # FPS = 30
    baseline_dir = os.path.join(data_dir, 'baseline')
    # offsets = np.zeros(len(session_list))
    sensors = ['2794', '2795', '2796', '3261', '3337']
    video = '7150991'
    # load_df = True
    # verbose = True
    # draw = True
    # sensor_dict = {'2794': 'Left Arm', '2795': 'Back', '2796': 'Left Leg', '3261': 'Right Leg', '3337': 'Right Arm'}
    ##############################################
    ##                  SyncWISE                 #
    ##############################################
    FPS = 30
    wind_stride = 1
    wind_length = 60
    num_rand = 10
    max_shift = 60
    mode_video = 'PCA'
    mode_imu = 'PCA'
    load_window = True
    load_score = False
    load_results = False
    shifts_syncwise = {}
    skipped_sessions = {}
    # valid_sessions = {}
    summ_mats = {}
    systematic_window = False
    print('systemic windows', systematic_window)
    suffix = '_win{}_max{}'.format(wind_length, max_shift)
    #sync_data_dir = '/media/yun/08790233DP/SyncWise_subset/SyncWise'+suffix
    sync_data_dir = 'tmp_data/SyncWise' + suffix
    #sync_data_dir = 'SyncWise'+suffix
    print(' mode:', mode_video, mode_imu, wind_length, max_shift)
    os.makedirs('figures/win{}_max{}'.format(wind_length, max_shift), exist_ok=True)
    offsets, valid_sessions = pickle.load(open('../../CMU/valid_sessions_win30_max60.pkl', 'rb'))
    session_list = valid_sessions['2794']
    for sensor in sensors:
        shifts_syncwise[sensor] = np.zeros(len(session_list))    
        skipped_sessions[sensor] = []
        valid_sessions[sensor] = []
        summ_mats[sensor] = []
        os.makedirs('figures/win{}_max{}/{}'.format(wind_length, max_shift, sensor), exist_ok=True)
    # load ground truth
    offsets = np.zeros(len(session_list))
    for i, session in enumerate(session_list):
        session = session.strip()
        opt_file = glob.glob(os.path.join(opt_dir, session+'_Video', session+'_7150991-*.pkl'))
        if len(opt_file) > 0:
            opt_file = opt_file[0]
            offsets[i] = int(opt_file[opt_file.find('-')+1:-4])
    offsets = offsets * 1000/FPS
    for i, session in enumerate(session_list):
        #if i < 87:
            #continue
        st = time.time()
        session = session.strip()
        out_dir = os.path.join(sync_data_dir, session)
        os.makedirs(out_dir, exist_ok=True)
        print('Processing video {}'.format(session))
        # load labels
        #pkl_file = open(os.path.join(baseline_dir, 'results_baseline_x_X.pkl'), 'rb')
        #offsets, shifts_baseline = pickle.load(pkl_file)
        df_file = os.path.join(baseline_dir, session, 'df_video_{}.pkl'.format(video))
        if os.path.exists(df_file):
            df_video = pd.read_pickle(df_file)
        else:
            print('skipping video {}'.format(session))
            for sensor in sensors:
                shifts_syncwise[sensor][i] = np.nan
                skipped_sessions[sensor].append(session)
            continue
        for sensor in sensors:
            # load df
            df_file = os.path.join(baseline_dir, session, 'df_sensor_{}.pkl'.format(sensor))
            df_imu = pd.read_pickle(df_file)
            
            st_time = max([df_imu.iloc[0]['SysTime'], df_video.iloc[0]['SysTime']])
            en_time = min([df_imu.iloc[len(df_imu) - 1]['SysTime'], df_video.iloc[len(df_video) - 1]['SysTime']])
            df_video_tmp = df_video[(df_video['SysTime'] >= st_time) & (df_video['SysTime'] < en_time)]
            vid_time_stamps = df_video_tmp['SysTime'].values
            if len(vid_time_stamps) == 0:
                print('    no intersection between video and imu, skip session {}'.format(session))
                skipped_sessions[sensor].append(session)
                shifts_syncwise[sensor][i] = np.nan
                continue
            df_imu = resample(df_imu, 'SysTime', samplingRate=0,
                                          gapTolerance=200, fixedTimeColumn=vid_time_stamps)
            df_imu = df_imu[(df_imu['SysTime'] >= st_time) & (df_imu['SysTime'] < en_time)]
            if len(df_video_tmp) != len(df_imu):
                print('    lengths of video and imu not equal, skip session {}'.format(session))
                print('len_vid = {}, len_sen = {}'.format(len(df_video_tmp), len(df_imu)))
                skipped_sessions[sensor].append(session)
                shifts_syncwise[sensor][i] = np.nan
                continue            
            
            num_windows = (len(df_video_tmp) - wind_length * FPS) // (FPS * wind_stride) + 1
            if num_windows <= 1:
                print('not enough windows')
                skipped_sessions[sensor].append(session)
                shifts_syncwise[sensor][i] = np.nan
                continue
            valid_sessions[sensor].append(session)
            # one window every win_stride seconds
            st_win = time.time()
            win_path = os.path.join(out_dir, 'win_{}_{}.pkl'.format(sensor, session))
            if load_window and os.path.exists(win_path):
                window_list = pickle.load(open(win_path, 'rb'))
            elif not os.path.exists(win_path):
                window_list = [None] * num_windows * num_rand
                for j in range(num_windows):
                    vid_start_frame = j*FPS*wind_stride
                    df_win_vid = df_video_tmp[vid_start_frame: vid_start_frame + wind_length * FPS]
                    for k in range(num_rand):
                        if systematic_window:
                            rand_offset = int((-max_shift + k * 2 * max_shift / (num_rand - 1)) * FPS)
                        else:
                            rand_offset = random.randint(-max_shift * FPS, max_shift * FPS)
                        rand_offset = np.min([np.max([-vid_start_frame, rand_offset]), len(df_video_tmp) - wind_length * FPS - vid_start_frame])
                        sen_start_frame = vid_start_frame + rand_offset
                        df_win_sen = df_imu[sen_start_frame: sen_start_frame + wind_length * FPS]
                        window_list[j * num_rand + k] = (df_win_vid, df_win_sen, rand_offset)
                # pickle.dump(window_list, open(win_path, 'wb'))
            en_win = time.time()
            print('    Time to get window: ', en_win - st_win)
            
            st_score = time.time()
            score_path = os.path.join(out_dir, 'score_{}_{}_xx.pkl'.format(sensor, session))
            if load_score and os.path.exists(score_path):
                scores_mat = pickle.load(open(score_path, 'rb'))  
            elif not os.path.exists(score_path):
                scores_mat = np.zeros((num_windows * num_rand, 2))
                # calculate conf, drift
                for j in range(len(window_list)):
                    df_win_vid, df_win_sen, rand_offset = window_list[j]
                    scores_mat[j] = drift_confidence(df_win_vid['diff_flow{}'.format(mode_video)].values, \
                        df_win_sen['Accel_{}'.format(mode_imu)].values)
                    scores_mat[j, 1] += rand_offset
                #scores_mat[:, 1] = scores_mat[:, 1] * 1000/FPS
                # pickle.dump(scores_mat, open(score_path, 'wb'))
            else:
                scores_mat = np.zeros((num_windows * num_rand, 2))
                # calculate conf, drift
                for j in range(len(window_list)):
                    df_win_vid, df_win_sen, rand_offset = window_list[j]
                    scores_mat[j] = drift_confidence(df_win_vid['diff_flow{}'.format(mode_video)].values, \
                                                     df_win_sen['Accel_{}'.format(mode_imu)].values)
                    scores_mat[j, 1] += rand_offset
                # scores_mat[:, 1] = scores_mat[:, 1] * 1000/FPS
                # pickle.dump(scores_mat, open(score_path, 'wb'))
            en_score = time.time()
            print(scores_mat[0, :])
            print('    time to get scores: ', en_score - st_score)
            
            st_res = time.time()
            #path = '/home/yun/Dropbox (GaTech)/code/figures/{}_{}.jpg'.format(session, sensor)
            res_path = os.path.join(out_dir, 'res_{}_{}_xx.pkl'.format(sensor, session))
            path = 'figures/win{}_max{}/{}/{}.jpg'.format(wind_length, max_shift, sensor, session)
            if load_results and os.path.exists(res_path):
                res = pickle.load(open(res_path, 'rb'))
            else:
                res = gaussianVoting(scores_mat, kernel_var=90, draw=True, path=path)
                # pickle.dump(res, open(res_path, 'wb'))
            en_res = time.time()
            print('    time to get res:', en_res - st_res)
            # 'session', 'offset', 'abs_offset', 'num_segs', 'conf', 'mu', 'sigma', 'mu_var', 'sigma_var', 'abs_mu'
            summ = [session, res[0], abs(res[0]), num_windows * num_rand, res[1], res[2][0][0], res[2][0][1], res[2][1][0, 0], \
                    res[2][1][1, 1], abs(res[2][0][0])]
            summ_mats[sensor].append(summ)
            shifts_syncwise[sensor][i] = summ[1]
        en = time.time()
        print('  processing time {} seconds'.format(en-st))        
    # result_file = os.path.join(sync_data_dir, 'results_SyncWISE_win{}_max{}_xx.pkl'.format(wind_length, max_shift))
    #offsets, shifts_syncwise = pickle.load(open(result_file, 'rb'))
    # pickle.dump([offsets, shifts_syncwise], open(result_file, 'wb'))
    # valid_session_file = os.path.join(sync_data_dir, 'valid_sessions_win{}_max{}.pkl'.format(wind_length, max_shift))
    # pickle.dump([offsets, valid_sessions], open(valid_session_file, 'wb'))
    for sensor in sensors:
        shifts_syncwise[sensor][67:91] = np.nan
    error = compute_error(offsets, shifts_syncwise)
    print(suffix)
    print(error)
    error.to_csv(os.path.join(sync_data_dir, 'df_error{}.csv'.format(sensor, suffix)))
    cols = ['session', 'offset', 'abs_offset', 'num_segs', 'conf', 'mu', 'sigma', 'mu_var', 'sigma_var', 'abs_mu']
    for sensor in sensors:
        df_summ = pd.DataFrame(data=summ_mats[sensor], columns=cols)
        df_summ.to_csv(os.path.join(sync_data_dir, 'df_summary_sensor{}{}.csv'.format(sensor, suffix)))


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    baseline('x')
    baseline('PCA')
    syncwise()

