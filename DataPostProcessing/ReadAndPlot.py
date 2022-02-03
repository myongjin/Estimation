import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import copy

# 데이터 읽어와서 앞뒤 짜르고 bias 잡는 코드
# 데이터 읽어와서 압력 정규화를 진행함(의도치 않았지만)
# 필터 후 뒤에 _filtered 가 붙음


def manualNormalization1D(min, max, data):
    for i in range(len(data)):
        data[i] = (data[i] - min) / (max - min)
    return data

def manualNormalization2D(min, max, data):
    print(len(data[0,:]), len(data[:,0]))
    for i in range(len(data[:,0])):
        for j in range(len(data[0,:])):
            data[i,j] = (data[i,j] - min)/(max - min)
    return data

def minMax2DArray(array):
    minValue = 10000000000
    maxValue = -10000000000
    for i in range(len(array[:,0])):
        pminValue = min(array[i, :])
        pmaxValue = max(array[i, :])
        if pminValue<minValue:
            minValue = pminValue

        if pmaxValue>maxValue:
            maxValue = pmaxValue

    return minValue, maxValue

# train_file_name = '../Data/Traindata16082021.csv'
# train_file_name = '../Data/Traindata2 16-08-2021 15-37.csv'
# train_file_name = '../Data/train 01-09-2021 15-47_filtered.csv'
# train 01-09-2021 15-51_filtered

file_name = 'ID024'
load_name = '../../Data/study/original/' + file_name + '.csv'
# load_name = '../../Data/Jennifer_pos_force.csv'
filterForce = 3.8
filterDiffForce = 50

manualForceBias = False
biasLeft = 0.71
biasRight = 0.48

cutStart = 100
cutEnd=-100

drawPos = False
filter = True
saveFlag = True
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2288)[17~]
dataSet = pd.read_csv(load_name).to_numpy()



dataSet = dataSet[:,:2288+17]
dataSet = dataSet[20:-40]
x = dataSet[:,0]
ref = 1


# pos check
manualNorm = [[-100, 100],[10, 150],[-200, 200]]
if drawPos:
    for i in range(3):
        y = dataSet[:,ref + i]
        print(min(y), max(y))
        plt.figure()
        plt.plot(y)

        y = manualNormalization1D(manualNorm[i][0],manualNorm[i][1],y)
        print(min(y), max(y))
    ref = 9
    for i in range(3):
        y = dataSet[:, ref + i]
        print(min(y), max(y))
        plt.figure()
        plt.plot(y)
        y = manualNormalization1D(manualNorm[i][0], manualNorm[i][1], y)
        print(min(y), max(y))


ref = 1
plt.figure()
plt.subplot(121)
y = dataSet[:, ref + 7]
plt.plot(dataSet[:, 0], y)

ref = 9
plt.subplot(122)
y = dataSet[:, ref + 7]
plt.plot(dataSet[:, 0], y)


# filter
if filter:
    if manualForceBias:
        ref = 1
        dataSet[:, ref + 7] = dataSet[:, ref + 7] + biasLeft
        ref = 9
        dataSet[:, ref + 7] = dataSet[:, ref + 7] + biasRight
    else:
        ref = 1
        dataSet[:, ref + 7] = dataSet[:, ref + 7] - np.mean(dataSet[:10, ref + 7])
        ref = 9
        dataSet[:, ref + 7] = dataSet[:, ref + 7] - np.mean(dataSet[:10, ref + 7])

    dataSet = dataSet[cutStart:cutEnd, :]

    ref = 1
    print("before Filter")
    ori = len(dataSet)
    print(ori)

    newDataSet = []
    for i in range(len(dataSet[:,ref + 7])):
        if dataSet[i,1 + 7] < filterForce and dataSet[i,9 + 7] < filterForce:
            newDataSet.append(dataSet[i,:])
    dataSet = np.array(newDataSet)
    print("after max force Filter")
    print(len(dataSet), len(dataSet) / ori)

    ref = 1
    plt.figure()
    plt.subplot(121)
    y = dataSet[:, ref + 7]
    plt.plot(dataSet[:, 0], y)

    ref = 9
    plt.subplot(122)
    y = dataSet[:, ref + 7]
    plt.plot(dataSet[:, 0], y)


    newDataSet = []
    newDataSet.append(dataSet[0, :])
    for i in range(1, len(dataSet[:, ref + 7])):
        if abs(dataSet[i - 1, 1 + 7] - dataSet[i, 1 + 7]) < filterDiffForce and \
                abs(dataSet[i - 1, 1 + 9] - dataSet[i, 1 + 9]) < filterDiffForce:
            newDataSet.append(dataSet[i, :])
    dataSet = np.array(newDataSet)
    print("after max diff force Filter")
    print(len(dataSet), len(dataSet)/ori)

    newDataSet = []
    newDataSet.append(dataSet[0, :])
    for i in range(1, len(dataSet[:, ref + 7])):
        if abs(dataSet[i - 1, 1 + 7] - dataSet[i, 1 + 7]) < filterDiffForce and \
                abs(dataSet[i - 1, 1 + 9] - dataSet[i, 1 + 9]) < filterDiffForce:
            newDataSet.append(dataSet[i, :])
    dataSet = np.array(newDataSet)
    print("after max diff force Filter")
    print(len(dataSet), len(dataSet) / ori)

    ref = 1
    plt.figure()
    plt.subplot(121)

    y = dataSet[:, ref + 7]
    plt.plot(dataSet[:, 0], y)

    ref = 9
    plt.subplot(122)

    y = dataSet[:, ref + 7]
    plt.plot(dataSet[:, 0], y)





if saveFlag:
    save_name = '../../Data/study/' + file_name + '_filtered.csv'
    np.savetxt(save_name, dataSet, delimiter=',')
    print("Data is saved")

plt.show()