import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# 회전은 건들면 안된다 - 변화가 선형으로 일어나지 않으므로 이상한 움직임을 만들어냄

def movingAverage1D(data, size):
    movingAvg = []
    newData = []

    for i in range(len(data)):
        movingAvg.append(data[i])
        if len(movingAvg)>=size:
            #데이터가 특정갯수 이상 쌓이면 평균 계산 및 새로운 데이터 대입
            newData.append(np.mean(np.array(movingAvg)))
            # 그리고 처음값 제거
            movingAvg.pop(0)
        else:
            newData.append(data[i])

    return np.array(newData)


# Load data
file_name = 'ID024_filtered'
load_name = '../../Data/study/' + file_name + '.csv'


# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]

dataSet = pd.read_csv(load_name).to_numpy()
print(dataSet.shape)

plotFreqResponse = False
filterPos = True
filterOri = False
filterForce = True
saveFlag = True

time = dataSet[:, 0]
posAndOri = dataSet[:, 1:9]
posAndOri2 = dataSet[:, 9:17]
pressure = dataSet[:, 17:]


# Filter requirements.
windowSize = 3
save_name = '../../Data/study/' + file_name + '_MAF_pos' + str(windowSize) +'.csv'

# Demonstrate the use of the filter.
# First make some data to be filtered.
t = time

if filterPos:
    plt.figure()
    for i in range(1,4,1):
        plt.subplot(3,1,i)
        print(i)
        data = dataSet[:, i]
        y = movingAverage1D(data, windowSize)

        plt.plot(t, data, 'b-', label='data')
        plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
        dataSet[:, i] = y

    plt.figure()
    for i in range(1, 4, 1):
        plt.subplot(3, 1, i)
        data = dataSet[:, 8 + i]
        y = movingAverage1D(data, windowSize)

        plt.plot(t, data, 'b-', label='data')
        plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
        dataSet[:, 8 + i] = y

if filterOri:
    plt.figure()
    for i in range(1, 5, 1):
        plt.subplot(4,1,i)
        data = dataSet[:, 3 + i]
        y = movingAverage1D(data, windowSize)

        plt.plot(t, data, 'b-', label='data')
        plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
        dataSet[:, 3 + i] = y

    plt.figure()
    for i in range(1, 5, 1):
        plt.subplot(4, 1, i)
        data = dataSet[:, 12 + i]
        y = movingAverage1D(data, windowSize)

        plt.plot(t, data, 'b-', label='data')
        plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
        dataSet[:, 12 + i] = y


if filterForce:
    plt.figure()
    for i in range(1,3,1):
        plt.subplot(2, 1, i)
        data = dataSet[:, 8*i]
        y = movingAverage1D(data, windowSize)
        plt.plot(t, data, 'b-', label='data')
        plt.plot(t, y, 'g--', label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()
        dataSet[:, 8 * i] = y

    plt.subplots_adjust(hspace=0.35)


if saveFlag:
    np.savetxt(save_name, dataSet, delimiter=',')
    print("Filtered data saved")

plt.show()