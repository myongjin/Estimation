import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
import pandas as pd
import math
from matplotlib import pyplot as plt
import random



def computeDTW(sequenceA, sequenceB):
    # sequence A should be shorter than B
    n = len(sequenceA) # n행 1열
    m = len(sequenceB) # m행 1열
    # print(n,m)
    inf = 10000000000000
    DTW = np.zeros((n+1,m+1))

    for i in range(0,n+1):
        for j in range(0,m+1):
            DTW[i,j] = inf
    DTW[0,0] = 0

    # compute DTW Matrix, n<m
    for i in range(1,n+1):
        for j in range(1,m+1):
            vectorA = sequenceA[i-1]
            vectorB = sequenceB[j-1]
            cost = abs(vectorA-vectorB)
            data = [DTW[i-1,j], DTW[i,j-1], DTW[i-1,j-1]]
            DTW[i, j] = cost + min(data)


    return DTW[n, m]

def computeDTWIndex(sequenceA, sequenceB):
    # sequence A should be shorter than B
    n = len(sequenceA) # n행 1열
    m = len(sequenceB) # m행 1열
    # print(n,m)
    inf = 10000000000000
    DTW = np.zeros((n+1,m+1))

    for i in range(0,n+1):
        for j in range(0,m+1):
            DTW[i,j] = inf
    DTW[0,0] = 0

    # compute DTW Matrix, n<m
    for i in range(1,n+1):
        for j in range(1,m+1):
            vectorA = sequenceA[i-1]
            vectorB = sequenceB[j-1]
            cost = abs(vectorA-vectorB)
            data = [DTW[i-1,j], DTW[i,j-1], DTW[i-1,j-1]]
            DTW[i, j] = cost + min(data)

    # Searching
    # 0 부터 m-1 까지
    Index_i = n
    Index_j = m
    listIndex = []
    listIndex.append(Index_i-1)
    for i in range(0,m-1):
        # find next Min pos
        values=[DTW[Index_i-1, Index_j], DTW[Index_i, Index_j-1], DTW[Index_i-1, Index_j-1]]
        # 최소값의 인덱스 확인
        # print(values)
        minIndex = values.index(min(values))

        if minIndex == 0:
            Index_i -= 1
        if minIndex == 1:
            Index_j -= 1
        if minIndex == 2:
            Index_i -= 1
            Index_j -= 1
        # print(Index_i, Index_j)
        listIndex.append(Index_i-1)
    listIndex.reverse()

    return listIndex

def computeDTWAvgDis(sequenceA, sequenceB):
    n = len(sequenceA)
    m = len(sequenceB)
    sameLength = False
    if n>m:
        shortS = sequenceB
        longS = sequenceA
    if m>n:
        shortS = sequenceA
        longS = sequenceB
    if n == m:
        sameLength = True
        convertedA = sequenceA
        convertedB = sequenceB

    if not sameLength:
        Index = computeDTWIndex(shortS,longS)
        # print(Index)
        convertedA = []
        for i in Index:
            convertedA.append(shortS[i])
        # print(convertedA)

        convertedB = longS
    print(convertedA)
    print(convertedB)

    return computeAvgDistance(convertedA, convertedB)


def generateSequence(startIdx, length, data):
    output = []
    for i in range(0,length):
        output.append(data[startIdx + i]-data[startIdx])

    return np.array(output)


def computeDistance(sequenceA, sequenceB):
    n = sequenceA.shape[0]  # n행
    normSum=0
    for i in range(0,n):
        vectorA = sequenceA[i]
        vectorB = sequenceB[i]
        normSum += abs(vectorA-vectorB)
    return normSum

def drawSquare(time,data,margin):
    upper = -10000000
    lower = 10000000
    for i in range(len(data)):
        if data[i] > upper:
            upper = data[i]
        if data[i] < lower:
            lower = data[i]
    length = upper - lower
    upper = upper + length * margin
    lower = lower - length * margin

    datax=[time[0], time[0], time[-1], time[-1], time[0]]
    datay=[lower, upper, upper, lower, lower]

    plt.plot(datax,datay)

def drawSquare(time,data,margin,color):
    upper = -10000000
    lower = 10000000
    for i in range(len(data)):
        if data[i] > upper:
            upper = data[i]
        if data[i] < lower:
            lower = data[i]
    length = upper - lower
    upper = upper + length * margin
    lower = lower - length * margin

    datax=[time[0], time[0], time[-1], time[-1], time[0]]
    datay=[lower, upper, upper, lower, lower]

    plt.plot(datax,datay,color=color)


def computeAvgDistance(sequenceA, sequenceB):
    n = len(sequenceA)  # n행
    normSum=0
    for i in range(0,n):
        vectorA = sequenceA[i]
        vectorB = sequenceB[i]
        normSum += abs(vectorA-vectorB)
    return normSum/n

def checkOverlapping(IndexA, IndexB, margin):
    # 오버래핑 검사 경우는 두가지
    # Index A가 앞서는 경우
    overlapping = False
    if IndexA[0]<IndexB[0] and IndexA[1]>IndexB[0]:
        if IndexA[1] - IndexB[0] > margin:
            overlapping = True
    # Index B가 앞서는 경우
    if IndexB[0]<IndexA[0] and IndexB[1]>IndexA[0]:
        if IndexB[1] - IndexA[0] > margin:
            overlapping = True

    return overlapping
# Load data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
dataSet = pd.read_csv('../../Data/Palpation_one_finger_Train 14-04-2021 13-14.csv').to_numpy()
print(dataSet.shape)

# 데이터 생성 우선 y축 데이터로 테스트해보자
time = dataSet[:,0]
testdata = dataSet[:,2]
testdata = testdata[100:-100]
time = time[100:-100]

# 전체 데이터 길이
total_length = testdata.shape[0]
print(total_length)

# 균일 간격 생성, 균일 길이, 하지만 겹치게 생성
l_min = 20
l_max = 30
R = 15
interval = 5

# 시퀀스 생성
startIdx = 0
sequences=[]
startIdxList = []
endIdxList = []
total_nbSequence = 0

figureIdx = 0
for _startIdx in range(0,l_min,10):
    for l_Sequence in range(l_min, l_max + 5,5):
        startIdx = _startIdx
        tempStartIdx = []
        tempEndIdx = []
        while startIdx+l_Sequence<total_length:
            startIdxList.append(startIdx)
            endIdxList.append(startIdx+l_Sequence)

            tempStartIdx.append(startIdx)
            tempEndIdx.append(startIdx + l_Sequence)
            sequences.append(generateSequence(startIdx,l_Sequence,testdata))
            total_nbSequence += 1
            startIdx += interval



        # 생성된 시퀀스 그리기
        if True:
            plt.figure(figureIdx)
            plt.plot(time, testdata)
            plt.title('Start Index: ' + str(_startIdx) + ' Sequence Length: ' + str(l_Sequence))
            plt.xlabel('Time (s)')
            plt.ylabel('Y Position (mm)')
            for i in range(len(tempStartIdx)):
                # 가장 많이 일치한 시퀀스의 Idx로 부터 그 시퀀스의 time 데이터에서 시작 위치 찾음
                startIdx = tempStartIdx[i]
                endIdx = tempEndIdx[i]
                drawSquare(time[startIdx:endIdx], testdata[startIdx:endIdx], 0.1, 'r')

            figureIdx += 1
plt.show()

print("Total sequences: ", total_nbSequence)



# 서치 하기
# 저장할 값, 매치 Index
matchedArray = []
# 각 i에 대해
for i in range(total_nbSequence):
    matchedIdx = []
    match = 0
    # 다른 모든 시퀀스와 비교
    for j in range(total_nbSequence):
        if i != j:
            # 만약 두 시퀀스가 겹치치 않으면서
            if not checkOverlapping([startIdxList[i], endIdxList[i]],[startIdxList[j], endIdxList[j]],l_min/3):
                # 만약 둘 사이 거리가 일정 이하라면
                # print(computeDTW(sequences[i], sequences[j]))
                if computeDTW(sequences[i], sequences[j]) < R:
                    matchedIdx.append(j)
    print(len(matchedIdx))
    matchedArray.append(matchedIdx)

# 각 행기준 매치되는 다른 시퀀스를 찾음
matchedArray = np.array(matchedArray, dtype=object)

bestIdx = -1
max_repeat = -100000

for i in range(nbSequence):

    if len(matchedArray[i])>max_repeat:
        max_repeat = len(matchedArray[i])
        bestIdx = i

plt.figure(0)
for i in range(len(matchedArray[bestIdx])):
    plt.plot(sequences[matchedArray[bestIdx][i]])

# 전체 데이터 내에서 반복된 시퀀스 보기
plt.figure(1)
plt.plot(time,testdata)
plt.xlabel('Time (s)')
plt.ylabel('Y Position (mm)')
for i in range(len(matchedArray[bestIdx])):
    # 가장 많이 일치한 시퀀스의 Idx로 부터 그 시퀀스의 time 데이터에서 시작 위치 찾음
    startIdx = startIdxList[matchedArray[bestIdx][i]]
    endIdx = endIdxList[matchedArray[bestIdx][i]]
    drawSquare(time[startIdx:endIdx],testdata[startIdx:endIdx],0.1,'r')



plt.show()