
# https://gist.github.com/junzis/e06eca03747fc194e322
# https://stackoverflow.com/questions/25191620/
#   creating-lowpass-filter-in-scipy-understanding-methods-and-units

import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
import pandas as pd

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
            if abs(data[i] - ref_point) < threshold:
                # 정상 범위면 다시 데이터 리딩 시작
                stop = False
                new_data.append(data[i])
            else:
                # 정상 범위 아니면 이전 튀기전 값 대입
                new_data.append(ref_point)
        else:
            # 튄적이 없다면 값이 튀는지 확인
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

# Load data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
dataSet = pd.read_csv('../Data/PalpationTwoFinger_Train 10-02-2021 11-16_manualFilter.csv').to_numpy()
print(dataSet.shape)

# Demonstrate the use of the filter.
# First make some data to be filtered.
t = dataSet[:, 0]
data=dataSet[:,8]
y=MyFilter(data,2)

# plt.plot(t, data, 'b-',marker='o',  label='data')
# plt.plot(t, y, 'g-', linewidth=2,marker='o',  label='filtered data')
# plt.xlabel('Time [sec]')
# plt.grid()
# plt.legend()
# plt.subplots_adjust(hspace=0.35)

plt.figure(1)
threshold = 10
for i in range(1,4,1):
    print(i)
    plt.subplot(3,1,i)
    data = dataSet[:, i]
    y = MyFilter(data, threshold)

    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

plt.figure(2)
for i in range(1, 4, 1):
    plt.subplot(3, 1, i)
    data = dataSet[:, 8 + i]
    y = MyFilter(data, threshold)

    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()



plt.show()

# time_pd.to_csv("filename.csv", mode='w')
pd.DataFrame(dataSet).to_csv('test4.csv', header=False,index=False)
