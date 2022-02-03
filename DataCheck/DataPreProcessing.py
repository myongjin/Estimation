import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, freqz
import datetime
import os
from matplotlib import pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def MyFilter(data, threshold):
    new_data = []
    prevalue = data[0]
    ref_point = 0
    stop = False
    new_data.append(data[0])
    for i in range(1, len(data), 1):
        # 만약 차이가 크면 이전 값을 대입함// 다시 값이 정상범위에 돌아올때까지 기다려야함

        # 한번 튄적이 있나 ?
        if stop:
            # 값이 정상범위로 돌아왔나?
            if abs(data[i] - prevalue) < threshold:
                # 정상 범위면 다시 데이터 리딩 시작
                stop = False
                new_data.append(data[i])
            else:
                # 정상 범위 아니면 이전 튀기전 값 대입
                new_data.append(ref_point)
        else:
            # 튄적이 없다면 값이 튄적이 있는지 확인
            if abs(data[i] - prevalue) > threshold:
                # 값이 튀었으면 튀었음을 저장 하고 이전 값을 대입
                stop = True
                ref_point = prevalue
                new_data.append(prevalue)
            else:
                # 안 튀었으면 정상적으로 값 대입
                new_data.append(data[i])
        prevalue = data[i]

    return new_data


# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
name = 'PalpationTwoFinger_Train 10-02-2021 11-16'
path = '../../Data/'
startIdx = 500
dataSet = pd.read_csv(path + name + '.csv').to_numpy()[startIdx:, :]
print(dataSet.shape)
t = dataSet[:, 0]
fs=len(t)/t[-1]

# filter position
# Filter requirements.
order = 10 # 필터 함수의 오더로 얼마나 급하게 꺽이는지를 결정함
cutoff = 1  # desired cutoff frequency of the filter, Hz
plt.figure(1)
for i in range(1,4,1):
    plt.subplot(2,2,i)
    data = dataSet[:, 1+i]
    y = butter_lowpass_filter(data, cutoff, fs, order)

    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    dataSet[:, 1 + i] = y

data=dataSet[:,8]
y=MyFilter(data,2)
plt.subplot(2, 2, 4)
plt.plot(t, data, 'b-',marker='o',  label='data')
plt.plot(t, y, 'g-', linewidth=2,marker='o',  label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()
dataSet[:,8] = y

plt.figure(2)
for i in range(1, 4, 1):
    plt.subplot(2, 2, i)
    data = dataSet[:, 8 + i]
    y = butter_lowpass_filter(data, cutoff, fs, order)

    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    dataSet[:, 8 + i] = y

data=dataSet[:,16]
y=MyFilter(data,2)
plt.subplot(2, 2, 4)
plt.plot(t, data, 'b-',marker='o',  label='data')
plt.plot(t, y, 'g-', linewidth=2, marker='o',  label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()
dataSet[:,16] = y




# draw a graph
plt.subplots_adjust(hspace=0.35)
plt.show()

# save
pd.DataFrame(dataSet).to_csv(path + name + '_filtered.csv', header=False,index=False)