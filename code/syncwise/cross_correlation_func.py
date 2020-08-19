import numpy as np
from numpy.fft import fft, ifft, fftshift
from statistics import median


def cross_correlation_using_fft(x, y):
    """
    calculate cross correlation using fft

    Args:
        x: signal x
        y: signal y

    Returns:
        float, cross correlation

    """
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)


def compute_shift(c):
    """
    Compute shift given the cross correlation signal

    Args:
        c: np array, cross correlation

    Returns:
        float, shift
            if shift < 0 means y starts 'shift' time steps before x
            if shift = 0 means that y starts 'shift' time steps after x
    """
    zero_index = int(len(c) / 2) - 1
    shift = zero_index - np.argmax(abs(c - median(c)))
    return shift

