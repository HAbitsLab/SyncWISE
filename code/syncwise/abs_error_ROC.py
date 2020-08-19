import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit
from textwrap import wrap


def abs_error_ROC_per_point(scores, mode='mean', draw=True):
    # INPUT: n x 2, cols: conf, abs error
    n = scores.shape[0]
    scores = scores[(-scores[:, 0]).argsort()]  # sort confidence in descending order
    if mode == 'mean':
        # mean of conf
        cum_error = np.cumsum(scores[:, 1], axis=0)
        ave_error = cum_error / (np.arange(n) + 1)
    elif mode == 'median':
        # median of conf
        ave_error = np.zeros((n,))
        for i in range(n):
            ave_error[i] = np.median(scores[:i + 1, 1])
    if draw:
        plt.plot(scores[:, 0], ave_error, '-o')
        plt.grid()
        plt.show()
    return np.stack((scores[:, 0], ave_error), axis=1)


def abs_error_ROC_fixed_steps(scores, num_threth=50, mode='median', draw=True):
    # INPUT: n x 2, cols: conf, abs error
    scores = scores[(-scores[:, 0]).argsort()]  # sort confidence in descending order
    ave_error = np.zeros((num_threth, 2))
    n = scores.shape[0]
    step_size = n / num_threth
    for i in range(num_threth):
        ind = min(np.int((i + 1) * step_size + 1), n - 1)
        ave_error[i, 0] = scores[ind, 0]
        if mode == 'mean':
            ave_error[i, 1] = np.mean(scores[:ind, 1])
        if mode == 'median':
            ave_error[i, 1] = np.median(scores[:ind, 1])
    if draw:
        plt.plot(ave_error[:, 0], ave_error[:, 1], '-o')
        plt.grid()
        plt.show()
    return ave_error


def gaussian(x, mu, sig):
    return 1 / (2 * np.pi * sig) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def scaled_gaussian(x, mu, sig, s):
    return s * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian_voting(scores, kernel_var=500, draw=True, path='../figures/offset.jpg'):
    # INPUT: n x 2, conf, offset
    # OUTPUT: offset
    offset_max = 20000
    x = np.arange(-offset_max, offset_max + 1)
    y = np.zeros(2 * offset_max + 1)
    for i in range(scores.shape[0]):
        y += gaussian(x, scores[i, 1], kernel_var) * scores[i, 0]
    y /= np.sum(scores[:, 0])
    offset = np.argmax(y) - offset_max

    # fit a Gaussian to voted_shift using nonlinear least square
    # confidence of the shift estimation can be described as the variance of the estimated model parameters
    # conf = max(abs(y-median(y)))/stdev(y)
    try:
        popt, pcov = curve_fit(gaussian, x, y, bounds=([-offset_max, 0], [offset_max, np.inf]))
        y_nlm = gaussian(x, *popt)
    except RuntimeError:
        popt, pcov = np.array([np.inf, np.inf, np.inf]), \
                     np.array([[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]])
        y_nlm = np.zeros((len(x)))
    conf = 200000 / popt[1] / pcov[0, 0]
    if draw:
        plt.figure()
        plt.plot(x, y, color='blue', label='weighted kde')
        plt.plot(x, y_nlm, color='red', label='fitted gaussian')
        plt.xlabel('shift/ms')
        plt.ylabel('probability')
        plt.legend(loc='upper right')
        title = '{} windows, offset={}ms, conf={:.2f}'.format(scores.shape[0], int(offset), conf)
        plt.title("\n".join(wrap(title, 60)))
        plt.savefig(path)
        plt.close()

    return offset, conf, [popt, pcov]


def gaussian_voting_per_video(scores_dataframe, kernel_var=100, thresh=0, min_voting_segs=0, draw=True,
                              folder='../figures/cross_corr/'):
    # INPUT: n x 3, conf, offset, video
    # OUTPUT: nv, offset
    scores = scores_dataframe[['confidence', 'drift', 'video']].to_numpy()
    scores = scores[scores[:, 0] > thresh]
    videos = np.unique(scores_dataframe[['video']].to_numpy())
    offset = np.zeros((len(videos)))
    conf = np.zeros((len(videos)))
    nlm_params = np.zeros((len(videos), 4))
    num_valid_segs = np.zeros((len(videos)))
    num_segs = 0
    num_videos = 0
    for i, vid in enumerate(videos):
        path = os.path.join(folder, 'offset_' + vid)
        valid_segs = scores[:, 2] == vid
        num_segs_cur = sum(valid_segs)
        if num_segs_cur > min_voting_segs:
            offset[i], conf[i], p = gaussian_voting(scores[valid_segs, :2], kernel_var, draw, path)
            nlm_params[i, :] = np.concatenate((p[0][:2], np.diag(p[1])[:2]))
            num_valid_segs[i] = num_segs_cur
            num_segs += num_segs_cur
            num_videos += 1
        else:
            offset[i] = np.nan
            conf[i] = np.nan
    try:
        ave_segs = num_segs / num_videos
    except ZeroDivisionError:
        ave_segs = np.nan
    summary_df = pd.DataFrame(np.concatenate(
        [np.stack([videos, offset, abs(offset), num_valid_segs, conf], axis=1), nlm_params, abs(nlm_params[:, :1])],
        axis=1), \
                              columns=['video', 'offset', 'abs_offset', 'num_segs', 'conf', 'mu', 'sigma', 'mu_var',
                                       'sigma_var', 'abs_mu'])

    return summary_df, ave_segs


def video_drift(scores_dataframe, output_file):
    offset, _ = gaussian_voting_per_video(scores_dataframe, draw=True, folder='../figures/cross_corr')
    offset.to_csv(output_file, index=None)
    return offset


def video_drift_ROC(scores_dataframe, num_thresh=50, draw=True, folder='../figures/cross_corr'):
    # INPUT: n x 2, cols: conf, abs error
    # lb = 3
    # ub = 5
    lb = 0
    ub = 7
    conf_threshs = np.linspace(lb, ub, num_thresh)
    offsets = np.zeros((num_thresh,))
    num_videos = np.zeros((num_thresh,))
    num_valid_segs = np.zeros((num_thresh,))
    for i, thresh in enumerate(conf_threshs):
        offset, ave_segs = gaussian_voting_per_video(scores_dataframe, kernel_var=500, thresh=thresh, draw=False)
        offset = offset.to_numpy()
        valid_videos = ~np.isnan(offset[:, 1].astype(np.float))
        offsets[i] = np.mean(abs(offset[valid_videos, 1]))
        num_videos[i] = sum(valid_videos)
        num_valid_segs[i] = ave_segs
    if draw:
        fig, ax = plt.subplots()
        ax.plot(conf_threshs, offsets, color="red", marker="o")
        ax.set_xlabel("confidence threshold", fontsize=14)
        ax.set_ylabel("abs offset", color="red", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(conf_threshs, num_videos, color="blue", marker="o")
        ax2.set_ylabel("number of valid videos", color="blue", fontsize=14)
        plt.grid()
        # plt.show()
        fig.savefig(os.path.join(folder, "video drift ROC curve"), \
                    format='jpeg', \
                    dpi=100, \
                    bbox_inches='tight')
        plt.figure()
        plt.plot(conf_threshs, num_valid_segs, marker="o")
        plt.grid()
        plt.savefig(os.path.join(folder, "num of segs"), \
                    format='jpeg', \
                    dpi=100, \
                    bbox_inches='tight')
        plt.show()
    return offsets, num_videos
