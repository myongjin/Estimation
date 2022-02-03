
# https://gist.github.com/junzis/e06eca03747fc194e322
# https://stackoverflow.com/questions/25191620/
#   creating-lowpass-filter-in-scipy-understanding-methods-and-units

import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
import pandas as pd


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Load data
file_name = 'ID002_filtered'
load_name = '../../Data/study/' + file_name + '.csv'
save_name = '../../Data/study/' + file_name + '_LPF.csv'

# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]

dataSet = pd.read_csv(load_name).to_numpy()
print(dataSet.shape)

plotFreqResponse = False
filterPos = False
filterForce = True
saveFlag = True

time = dataSet[:, 0]
posAndOri = dataSet[:, 1:9]
posAndOri2 = dataSet[:, 9:17]
pressure = dataSet[:, 17:]

fs=len(time)/time[-1]
print(len(time))
print(time[-1])
print(fs)


# Filter requirements.
order = 3 # 필터 함수의 오더로 얼마나 급하게 꺽이는지를 결정함
cutoff = 2  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
if plotFreqResponse:
    w, h = freqz(b, a, worN=8000)
    plt.figure()
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b') # 왜 보드플롯으로 그리지 않았지 ?
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko') # 점찍고
    plt.axvline(cutoff, color='k') # 선긋고
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()


# Demonstrate the use of the filter.
# First make some data to be filtered.
t = time

if filterPos:
    plt.figure()
    for i in range(1,4,1):
        plt.subplot(3,1,i)
        data = dataSet[:, 1+i]
        y = butter_lowpass_filter(data, cutoff, fs, order)
        dataSet[:, 1 + i] = y
        plt.plot(t, data, 'b-', label='data')
        plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()

    plt.figure()
    for i in range(1, 4, 1):
        plt.subplot(3, 1, i)
        data = dataSet[:, 8 + i]
        y = butter_lowpass_filter(data, cutoff, fs, order)
        dataSet[:, 8 + i] = y
        plt.plot(t, data, 'b-', label='data')
        plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()

if filterForce:
    plt.figure()
    for i in range(1,3,1):
        plt.subplot(2, 1, i)
        data = dataSet[:, 8*i]
        y = butter_lowpass_filter(data, cutoff, fs, order)

        plt.plot(t, data, 'b-', label='data')
        plt.plot(t, y, 'g--', label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
        dataSet[:, 8 * i] = y

    plt.subplots_adjust(hspace=0.35)
    plt.show()

if saveFlag:
    np.savetxt(save_name, dataSet, delimiter=',')
    print("Filtered data saved")