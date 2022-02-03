import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
import pandas as pd
import math
from matplotlib import pyplot as plt
import matplotlib.patches as patches
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

def computeDTWVector(sequenceA, sequenceB):
    # sequence A should be shorter than B
    n = sequenceA.shape[0] # n행 1열
    m = sequenceB.shape[0] # m행 1열
    # print(n,m)
    inf = 10000000000000
    DTW = np.zeros((n+1,m+1))

    for i in range(0, n+1):
        for j in range(0, m+1):
            DTW[i,j] = inf
    DTW[0,0] = 0

    # compute DTW Matrix, n<m
    for i in range(1, n+1):
        for j in range(1, m+1):
            vectorA = sequenceA[i-1, :]
            vectorB = sequenceB[j-1, :]
            cost = np.linalg.norm(vectorA-vectorB)
            data = [DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1]]
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

def drawSquareColor(time,data,margin,color):
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
    if IndexA[0]<=IndexB[0] and IndexA[1]>=IndexB[0]:
        if IndexA[1] - IndexB[0] > margin:
            overlapping = True
        # A 안에 B가 있는 경우
        if IndexA[1]>=IndexB[1]:
            overlapping = True

    # Index B가 앞서는 경우
    if IndexB[0]<=IndexA[0] and IndexB[1]>=IndexA[0]:
        if IndexB[1] - IndexA[0] > margin:
            overlapping = True
        # B 안에 A가 있는 경우
        if IndexA[1] <= IndexB[1]:
            overlapping = True


    return overlapping

def mergeTwoSequences(IndexA, IndexB):
    # 오버래핑 검사 경우는 두가지
    # Index A가 앞서는 경우
    Index = []
    if IndexB[0] < IndexA[0]:
        Index.append(IndexB[0])
    else:
        Index.append(IndexA[0])

    if IndexA[1] < IndexB[1]:
        Index.append(IndexA[1])
    else:
        Index.append(IndexB[1])

    return Index

def getSequencefromIdx(seq_Idx, _startIdxList, _endIdxList, data):
    startIdx = _startIdxList[seq_Idx]
    endIdx = _endIdxList[seq_Idx]
    return data[startIdx:endIdx]


def draw3DLine(x,y,z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x,y,z)

def draw3DLineOnFig(fig, x, y, z):
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z)

def sortInDecendingOrder(list):
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            if len(list[j]) > len(list[i]):
                # 기준 i보다 j가 더 크다면 i와 j 스왑
                temp = list[i]
                list[i] = list[j]
                list[j] = temp

    return list

def checkDuplication(listA, listB):
    numDup = 0
    for a in listA:
        for b in listB:
            if a==b:
                numDup += 1

    return numDup

def addSequenceList(listA, listB, startIdxList, endIdxList):
    # 각 B에 대해
    for b in listB:
        dup = False

        # 기존 A에 B요소가 있나 확인
        for a in listA:
            if a == b:
                dup = True
        # 중복되는게 없다면
        if not dup:
            # 시퀀스 간의 오버래핑 검사
            for a in listA:
                overlapping = False
                length = (endIdxList[a] - startIdxList[a])*2/3
                if checkOverlapping([startIdxList[b], endIdxList[b]], [startIdxList[a], endIdxList[a]], length):
                    overlapping = True
            # 최종적으로 중복된 시퀀스가 없고 기존 시퀀스와 중복 많이 안되면 추가
            if not overlapping:
                listA.append(b)
    return listA

def removeDuplicatedPattern(matchedArray, duprate, startIdxList, endIdxList):
    # 각 패턴 사이 중복되는 시퀀스 수가 많으면 그 패턴을 삭제하고 합쳐버리기
    # 기준은 합쳐지는 쪽 시퀀스의 80프로가 겹치면 합침
    new_matchedArray = []
    removedIdx = []
    for i in range(len(matchedArray)):
        # 패턴 중복 검사에서 이미 제거된 패턴인지 확인
        removed = False
        for idx in removedIdx:
            if i == idx:
                removed = True
        # 제거된 패턴이 아니면
        if not removed:
            # 다른 모든 패턴과 비교 분석
            listA = matchedArray[i]
            for j in range(len(matchedArray)):
                if i != j:
                    listB = matchedArray[j]
                    dup = checkDuplication(listA, listB)
                    # B가 A와 겹치는게 80프로 이상이면 B를 A와 합침
                    if dup > len(listB) * duprate:
                        # print(dup, len(listB)*8/10)
                        # 합치기
                        removedIdx.append(j)
                        listA = addSequenceList(listA, listB, startIdxList, endIdxList)
            # 2번 이상 반복된 것만 찾기
            if len(listA) > 3:
                # print(listA)
                new_matchedArray.append(listA)

    return new_matchedArray

def drawPattern4D(fig, i, matchedArray,startIdxList, endIdxList, testdata):
    ax = fig.gca(projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    x = getSequencefromIdx(matchedArray[i][0], startIdxList, endIdxList, testdata[:, 0])
    y = getSequencefromIdx(matchedArray[i][0], startIdxList, endIdxList, testdata[:, 1])
    z = getSequencefromIdx(matchedArray[i][0], startIdxList, endIdxList, testdata[:, 2])
    avgF = 0
    avgDepth = 0
    for idx in matchedArray[i]:
        f = getSequencefromIdx(idx, startIdxList, endIdxList, testdata[:, 3])
        _y = getSequencefromIdx(idx, startIdxList, endIdxList, testdata[:, 1])
        max_force = max(f)
        avgF += max_force
        depth = max(_y) - min(_y)
        avgDepth += depth

    avgF /= len(matchedArray[i])
    avgDepth /= len(matchedArray[i])
    avgF = round(avgF, 2)
    avgDepth = round(avgDepth, 2)

    ax.set_title("Pattern Index: " + str(i) + ", The number of repetition: " + str(
        len(matchedArray[i])) + "\nAvg. Max. Force (N): " + str(avgF) + ", Avg. Depth (mm): " + str(avgDepth))

    draw3DLineOnFig(fig, x, z, y)

def drawPattern3D(fig, i, matchedArray, startIdxList, endIdxList, testdata):
    ax = fig.gca(projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    x = getSequencefromIdx(matchedArray[i][0], startIdxList, endIdxList, testdata[:, 0])
    y = getSequencefromIdx(matchedArray[i][0], startIdxList, endIdxList, testdata[:, 1])
    z = getSequencefromIdx(matchedArray[i][0], startIdxList, endIdxList, testdata[:, 2])

    avgDepth = 0
    for idx in matchedArray[i]:
        _y = getSequencefromIdx(idx, startIdxList, endIdxList, testdata[:, 1])
        depth = max(_y) - min(_y)
        avgDepth += depth

    avgDepth /= len(matchedArray[i])
    avgDepth = round(avgDepth, 2)

    ax.set_title("Pattern Index: " + str(i) + ", The number of repetition: " + str(
        len(matchedArray[i])) + "\nAvg. Depth (mm): " + str(avgDepth))

    draw3DLineOnFig(fig, x, z, y)

def drawPatternPlane(fig, idx, matchedArray, startIdxList, endIdxList, testdata):
    seqList = matchedArray[idx]
    # x-z 평면에서 패턴 보기
    ax = fig.gca()
    ax.grid()

    ax.set_xlabel('Z Position (mm)')
    ax.set_ylabel('X Position (mm)')

    ax.set_xlim([min(testdata[:, 2]), max(testdata[:, 2])])
    ax.set_ylim([min(testdata[:, 0]), max(testdata[:, 0])])


    ax.invert_yaxis()

    avgDepth = 0
    for i in range(len(seqList)):
        # 가장 많이 일치한 시퀀스의 Idx로 부터 그 시퀀스의 time 데이터에서 시작 위치 찾음
        x = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 0])
        z = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 2])
        _y = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 1])
        depth = max(_y) - min(_y)
        avgDepth += depth

        ax.plot(z, x)

    avgDepth /= len(seqList)
    avgDepth = round(avgDepth, 2)

    ax.set_title("Pattern Index: " + str(idx) + ", The number of repetition: " + str(
        len(seqList)) + "\nAvg. Depth (mm): " + str(avgDepth))

def drawPatternPlaneLimited(fig, idx, matchedArray, startIdxList, endIdxList, testdata, _xlim, _ylim):
    seqList = matchedArray[idx]
    # x-z 평면에서 패턴 보기
    ax = fig.gca()
    ax.grid()

    ax.set_xlabel('Z Position (mm)')
    ax.set_ylabel('X Position (mm)')

    ax.set_xlim(_xlim)
    ax.set_ylim(_ylim)


    ax.invert_yaxis()

    avgDepth = 0
    for i in range(len(seqList)):
        # 가장 많이 일치한 시퀀스의 Idx로 부터 그 시퀀스의 time 데이터에서 시작 위치 찾음
        x = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 0])
        z = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 2])
        _y = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 1])
        depth = max(_y) - min(_y)
        avgDepth += depth

        ax.plot(z, x)

    avgDepth /= len(seqList)
    avgDepth = round(avgDepth, 2)

    ax.set_title("Pattern Index: " + str(idx) + ", The number of repetition: " + str(
        len(seqList)) + "\nAvg. Depth (mm): " + str(avgDepth))

def draw6DPatternPlaneLimited(fig, idx, matchedArray, startIdxList, endIdxList, testdata, _xlim, _ylim):
    seqList = matchedArray[idx]
    # x-z 평면에서 패턴 보기

    axes = fig.subplots(nrows = 1, ncols = 2)


    axes[0].grid()
    axes[0].set_title('Left index')
    axes[0].set_xlabel('Z Position (mm)')
    axes[0].set_ylabel('X Position (mm)')
    axes[0].set_xlim(_xlim)
    axes[0].set_ylim(_ylim)
    axes[0].invert_yaxis()

    axes[1].grid()
    axes[1].set_title('Right index')
    axes[1].set_xlabel('Z Position (mm)')
    axes[1].set_ylabel('X Position (mm)')
    axes[1].set_xlim(_xlim)
    axes[1].set_ylim(_ylim)
    axes[1].invert_yaxis()

    avgDepth = 0
    for i in range(len(seqList)):
        # 가장 많이 일치한 시퀀스의 Idx로 부터 그 시퀀스의 time 데이터에서 시작 위치 찾음
        x = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 0])
        z = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 2])
        _y = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 1])
        depth = max(_y) - min(_y)
        avgDepth += depth
        axes[0].plot(z, x)

        x = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 3])
        z = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 5])
        axes[1].plot(z, x)

    avgDepth /= len(seqList)
    avgDepth = round(avgDepth, 2)

    plt.suptitle("Pattern Index: " + str(idx) + ", The number of repetition: "+ str(len(seqList)))

def draw1DPatternInTime(fig, seqList, startIdxList, endIdxList, totalTime, targetData):

    # x-z 평면에서 패턴 보기
    ax = fig.gca()
    ax.grid()


    for i in range(len(seqList)):
        # 가장 많이 일치한 시퀀스의 Idx로 부터 그 시퀀스의 time 데이터에서 시작 위치 찾음
        data = getSequencefromIdx(seqList[i], startIdxList, endIdxList, targetData)
        time = totalTime[startIdxList[seqList[i]]:endIdxList[seqList[i]]]
        #move time to see all forces on the origin
        time = time-time[0]
        ax.plot(time, data)

def draw2ForcePatternInTime(fig, seqList, startIdxList, endIdxList, totalTime, targetData):
    axes = fig.subplots(nrows=1, ncols=2)

    axes[0].grid()
    axes[0].set_title('Left index')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Force (N)')

    axes[1].grid()
    axes[1].set_title('Right index')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Force (N)')

    for i in range(len(seqList)):
        # 가장 많이 일치한 시퀀스의 Idx로 부터 그 시퀀스의 time 데이터에서 시작 위치 찾음
        data = getSequencefromIdx(seqList[i], startIdxList, endIdxList, targetData)
        time = totalTime[startIdxList[seqList[i]]:endIdxList[seqList[i]]]
        # move time to see all forces on the origin
        time = time - time[0]
        axes[0].plot(time, data[:, 0])
        axes[1].plot(time, data[:, 1])




def draw6DPatternPlane(fig, idx, matchedArray, startIdxList, endIdxList, testdata ):
    seqList = matchedArray[idx]
    # x-z 평면에서 패턴 보기

    axes = fig.subplots(nrows = 1, ncols = 2)

    _xlim = [min(min(testdata[:, 0]), min(testdata[:, 3])), max(max(testdata[:, 0]), max(testdata[:, 3]))]
    _ylim = [min(min(testdata[:, 2]), min(testdata[:, 5])), max(max(testdata[:, 2]), max(testdata[:, 5]))]

    axes[0].grid()
    axes[0].set_xlabel('Z Position (mm)')
    axes[0].set_ylabel('X Position (mm)')
    axes[0].set_xlim(_xlim)
    axes[0].set_ylim(_ylim)
    axes[0].invert_yaxis()

    axes[1].grid()
    axes[1].set_xlabel('Z Position (mm)')
    axes[1].set_ylabel('X Position (mm)')
    axes[1].set_xlim(_xlim)
    axes[1].set_ylim(_ylim)
    axes[1].invert_yaxis()

    avgDepth = 0
    for i in range(len(seqList)):
        # 가장 많이 일치한 시퀀스의 Idx로 부터 그 시퀀스의 time 데이터에서 시작 위치 찾음
        x = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 0])
        z = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 2])
        _y = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 1])
        depth = max(_y) - min(_y)
        avgDepth += depth
        axes[0].plot(z, x)

        x = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 3])
        z = getSequencefromIdx(seqList[i], startIdxList, endIdxList, testdata[:, 5])
        axes[1].plot(z, x)

    avgDepth /= len(seqList)
    avgDepth = round(avgDepth, 2)

    plt.suptitle("Pattern Index: " + str(idx) + ", The number of repetition: " + str(len(seqList)))

# 비교 분석용
# 데이터를 전부 통으로 받는다
def drawPatternPlaneV2Limited(fig, idx, matchedArray, startIdxList, endIdxList, testdata, _xlim, _ylim):
    seqList = matchedArray[idx]
    # x-z 평면에서 패턴 보기
    ax = fig.gca()
    ax.grid()

    ax.set_xlabel('Z Position (mm)')
    ax.set_ylabel('X Position (mm)')

    ax.set_xlim(_xlim)
    ax.set_ylim(_ylim)

    ax.invert_yaxis()

    for i in range(len(seqList)):
        # 가장 많이 일치한 시퀀스의 Idx로 부터 그 시퀀스의 time 데이터에서 시작 위치 찾음
        dataIdx = seqList[i][0]
        seqIdx = seqList[i][1]
        startIdx = startIdxList[dataIdx][seqIdx]
        endIdx = endIdxList[dataIdx][seqIdx]
        x = testdata[dataIdx][startIdx:endIdx, 0]
        z = testdata[dataIdx][startIdx:endIdx, 2]

        if dataIdx == 0:
            ax.plot(z, x, color='red')
        if dataIdx == 1:
            ax.plot(z, x, color='blue')




    ax.set_title("Pattern Index: " + str(idx))

# save sequence indexes
def savePatternForUnity(data, filename):
    f = open(filename, 'w')

    line = "%d\n" % len(data)
    f.write(line)

    for i in range(len(data)):
        line = "%d " % len(data[i])
        f.write(line)
        for j in range(len(data[i])):
            line = "%d " % data[i][j]
            f.write(line)
        line = "\n"
        f.write(line)
    f.close()

# save sequence indexes with start and end index of sequences
def savePatternV2ForUnity(matchedIndex, startIdx, endIdx, filename):
    f = open(filename, 'w')

    # save start and end index
    line = "%d\n" % len(startIdx)
    f.write(line)

    for i in range(len(startIdx)):
        line = "%d," % startIdx[i]
        f.write(line)
        line = "%d\n" % endIdx[i]
        f.write(line)


    # save sequence indexes
    line = "%d\n" % len(matchedIndex)
    f.write(line)

    for i in range(len(matchedIndex)):
        line = "%d," % len(matchedIndex[i])
        f.write(line)
        for j in range(len(matchedIndex[i])):
            line = "%d," % matchedIndex[i][j]
            f.write(line)
        line = "\n"
        f.write(line)
    f.close()

def drawFilledSquare(ax, x, y, width, height, facecolorRGB, edgecolorRGB, _alpha):
    facecolorHex = convertRGBtoHEX(facecolorRGB)
    edgecolorHex = convertRGBtoHEX(edgecolorRGB)
    ax.add_patch(
        patches.Rectangle(
            (x, y),  # (x, y)
            width, height,  # width, height
            edgecolor= edgecolorHex,
            facecolor= facecolorHex,
            fill=True,
            alpha = _alpha
        ))


def convertRGBtoHEX(rgb):
    hexColor ="#"

    for i in range(3):
        if len(str(hex(rgb[i]))[2:]) < 2:
            hexColor += "0" + str(hex(rgb[i]))[2:]
        else:
            hexColor += str(hex(rgb[i]))[2:]
    return hexColor

def averageForce(x, y, f, x_range, y_range):
    force_list = listWithInRange(x,y,x_range, y_range)
    avgF = 0
    if len(force_list) > 0:
        for i in force_list:
            avgF += f[i]
        avgF /= len(force_list)
    else:
        avgF = 0

    return avgF

def maxForce(x, y, f, x_range, y_range):
    force_list = listWithInRange(x,y,x_range, y_range)
    max_F = -100000000000
    if len(force_list)>0:
        for i in force_list:
            if f[i] > max_F:
                max_F = f[i]
    else:
        max_F = 0


    return max_F

def listWithInRange(x,y,x_range, y_range):
    idx_list = []
    for i in range(len(x)):
        if x_range[0] <= x[i] and x_range[1] > x[i]:
            idx_list.append(i)

    return_list = []
    for i in idx_list:
        if y_range[0] <= y[i] and y_range[1] > y[i]:
            return_list.append(i)

    return return_list