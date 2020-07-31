import numpy as np
from numpy.fft import fft, ifft, fftshift
from statistics import median


def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)


def cross_corr_conv(x, y):
    a = np.convolve(x, y, 'same')
    return a


# shift < 0 means that y starts 'shift' time steps before x # shift &gt; 0 means that y starts 'shift' time steps after x
def compute_shift(c):
    zero_index = int(len(c) / 2) - 1
    shift = zero_index - np.argmax(abs(c - median(c)))
    return shift
