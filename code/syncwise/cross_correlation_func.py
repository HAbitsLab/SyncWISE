import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
import matplotlib.pylab as plt
from statistics import median


# def cross_correlation_manual(x, y):
#     assert len(x) == len(y)
#     l = len(x)
#     cross_corr = []
#     # s means shift
#     for s in range(l): 
#         v = 0
#         x_s = x[]

#         for i in range(len(x_s)):
#             v += x_s[i] * y_s[l-i]
#         cross_corr.append(v)
#     return

def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

def cross_corr_conv(x, y):
    a = np.convolve(x, y, 'same')
    return a
 
# # shift < 0 means that y starts 'shift' time steps before x # shift &gt; 0 means that y starts 'shift' time steps after x
# def compute_shift(x, y):
#     assert len(x) == len(y)
#     c = cross_correlation_using_fft(x, y)
#     assert len(c) == len(x)
#     zero_index = int(len(x) / 2) - 1
#     shift = zero_index - np.argmax(abs(c))
#     return shift


# shift < 0 means that y starts 'shift' time steps before x # shift &gt; 0 means that y starts 'shift' time steps after x
def compute_shift(c):
    # assert len(c) == len(x)
    zero_index = int(len(c) / 2) - 1
    shift = zero_index - np.argmax(abs(c-median(c)))
    return shift

def test1():
    l = 20
    a = np.random.rand(l,)
    b = np.random.rand(l,)
    assert len(a) == len(b)

    x = cross_correlation_using_fft(a,b)/len(a)


def test2():
    l = 20
    a = np.cos(np.linspace(-np.pi, np.pi, 20)) + 10
    b = np.sin(np.linspace(-np.pi, np.pi, 20)) + 10

    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(-np.pi, np.pi, 20), a)
    plt.title('signal A')
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    # plt.axis('tight')
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(-np.pi, np.pi, 20), b)
    plt.title('signal B')
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    # plt.axis('tight')
    plt.show()

    assert len(a) == len(b)

    x = cross_correlation_using_fft(a,b)/len(a)
    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.title('cross correlation using fft')
    print(len(x),x)

    np_corr = np.correlate(a, b, 'same')
    plt.subplot(2, 1, 2)
    plt.plot(np_corr)
    plt.title('cross correlation using numpy')
    plt.show()
    print(len(np_corr),np_corr)

def test3():
    l = 20
    a = np.cos(np.linspace(-np.pi, np.pi, 20))
    b = np.sin(np.linspace(-np.pi, np.pi, 20))

    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(-np.pi, np.pi, 20), a)
    plt.title('signal A')
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    # plt.axis('tight')
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(-np.pi, np.pi, 20), b)
    plt.title('signal B')
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    # plt.axis('tight')
    plt.show()

    assert len(a) == len(b)

    x = cross_correlation_using_fft(a,b)/len(a)
    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.title('cross correlation using fft')
    print(len(x),x)

    np_corr = np.correlate(a, b, 'same')
    plt.subplot(2, 1, 2)
    plt.plot(np_corr)
    plt.title('cross correlation using numpy')
    plt.show()
    print(len(np_corr),np_corr)

def test4_fft_vs_conv_crosscorr():
    # a = np.cos(np.linspace(-np.pi, np.pi, 20))
    # b = np.sin(np.linspace(-np.pi, np.pi, 20))

    # plt.subplot(2, 1, 1)
    # plt.plot(np.linspace(-np.pi, np.pi, 20), a)
    # plt.title('signal A')
    # plt.xlabel('Angle [rad]')
    # plt.ylabel('sin(x)')
    # # plt.axis('tight')
    # plt.subplot(2, 1, 2)
    # plt.plot(np.linspace(-np.pi, np.pi, 20), b)
    # plt.title('signal B')
    # plt.xlabel('Angle [rad]')
    # plt.ylabel('sin(x)')
    # # plt.axis('tight')
    # plt.show()

    a = np.array([1,1,1,1,1,10,9,8,7,6,5,4,3,2,1,1,1,1,1,1])
    b = a[::-1]


    plt.subplot(2, 1, 1)
    plt.plot(a)
    plt.title('signal A')
    plt.subplot(2, 1, 2)
    plt.plot(b)
    plt.title('signal B')
    plt.show()

    assert len(a) == len(b)

    x = cross_correlation_using_fft(a,b)#/len(a)
    plt.subplot(4, 1, 1)
    plt.plot(x)
    plt.title('cross correlation using fft')
    print(len(x),x)

    np_corr = cross_corr_conv(a, b)
    plt.subplot(4, 1, 2)
    plt.plot(np_corr)
    plt.title('conv using numpy')
    print(len(np_corr),np_corr)

    np_corr = cross_corr_conv(a, b[::-1])
    plt.subplot(4, 1, 3)
    plt.plot(np_corr)
    plt.title('cross correlation using numpy conv')
    print(len(np_corr),np_corr)

    
    np_corr = np.correlate(a, b, 'same')
    plt.subplot(4, 1, 4)
    plt.plot(np_corr)
    plt.title('cross correlation using numpy correlate')
    plt.show()
    print(len(np_corr),np_corr)



if __name__ == '__main__':
    # test2()
    # test3()
    test4_fft_vs_conv_crosscorr()

 