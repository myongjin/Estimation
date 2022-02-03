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

def addSequenceList(listA, listB):
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

def removeDuplicatedPattern(matchedArray, duprate):
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
                        listA = addSequenceList(listA, listB)
            # 2번 이상 반복된 것만 찾기
            if len(listA) > 3:
                # print(listA)
                new_matchedArray.append(listA)

    return new_matchedArray

def drawPattern(fig, i, matchedArray,startIdxList, endIdxList, testdata):
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

# Load data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
dataSet = pd.read_csv('../../Data/Palpation_one_finger_Train 14-04-2021 13-14.csv').to_numpy()
print(dataSet.shape)

# 데이터 생성 우선 y축 데이터로 테스트해보자
time = dataSet[:,0]
xyzdata = dataSet[:,1:4]
forcedata = dataSet[:,8]



print(xyzdata.shape)

# 데이터 세워서 합치기
forcedata = forcedata.reshape(-1,1)
testdata = np.hstack((xyzdata,forcedata))
print([min(testdata[:, 0]), max(testdata[:, 0])])
print([min(testdata[:, 2]), max(testdata[:, 2])])

print(testdata.shape)



#앞 뒤 잘라내기
cutStart = 100
cutEnd = -100
testdata = testdata[cutStart:cutEnd,:]
time = time[cutStart:cutEnd]


# 전체 데이터 길이
total_length = testdata.shape[0]
print(total_length)

# 변수 설정
# 균일 간격 생성, 균일 길이, 하지만 겹치게 생성
do_analysis = False
drawGeneratedSequence = False
drawSequencesInTime = False
drawBestMatch = True
l_min = 10
l_max = 22
R = 150
file_name = "PatternAnalysis4D_R" + str(R) +"_Lminmax" + str(l_min) + str(l_max)
fig_Idx = 0
pattern_filter = 3
dup_rate = 0.5
# 뭘 그릴지 정함
bestIdx = 0
drawfrom = 0
drawUptoIdx = 0



# 시퀀스 생성
startIdx = 0
sequences=[]
startIdxList = []
endIdxList = []
total_nbSequence = 0

# 시퀀스 생성
if do_analysis:
    for _startIdx in range(0, l_min, 2):
        for l_Sequence in range(l_min, l_max + 2, 2):
            print("Start Idx: ", _startIdx, " Length: ", l_Sequence)
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
                startIdx += round(l_Sequence*(3/4))


            # 생성된 시퀀스 그리기
            if drawGeneratedSequence:
                for j in range(testdata.shape[1]):
                    plt.figure(fig_Idx)
                    fig_Idx += 1
                    plt.plot(time, testdata[:,j])
                    plt.title('Start Index: ' + str(_startIdx) + ' Sequence Length: ' + str(l_Sequence))
                    plt.xlabel('Time (s)')
                    if j == 0:
                        plt.ylabel('Y Position (mm)')
                    if j == 1:
                        plt.ylabel('Force (N)')

                    for i in range(len(tempStartIdx)):
                        # 가장 많이 일치한 시퀀스의 Idx로 부터 그 시퀀스의 time 데이터에서 시작 위치 찾음
                        startIdx = tempStartIdx[i]
                        endIdx = tempEndIdx[i]
                        drawSquareColor(time[startIdx:endIdx], testdata[startIdx:endIdx, j], 0.1, 'r')
    print("Total sequences: ", total_nbSequence)
plt.show()




# 서치 하기, 저장할 값, 매치 Index
matchedArray = []
# 각 i에 대해
if do_analysis:
    for i in range(total_nbSequence):
        # 자기 자신 추가
        matchedIdx = [i]
        match = 0
        # 다른 모든 시퀀스와 비교
        for j in range(total_nbSequence):
            if i != j:
                # 새로 추가하려는 j와 기존 그룹과 겹치는게 없는지 부터 확인
                overlapping = False
                for k in matchedIdx:
                    if checkOverlapping([startIdxList[k], endIdxList[k]],[startIdxList[j], endIdxList[j]],l_min/4):
                        overlapping = True
                        break

                if not overlapping:
                    # 만약 둘 사이 거리가 일정 이하라면
                    # print(computeDTWVector(sequences[i], sequences[j]))
                    # 둘 사이 거리 비교할때는 원점으로 각 시퀀스를 옮겨서 비교함
                    if computeDTWVector(sequences[i], sequences[j]) < R:
                        matchedIdx.append(j)

        print(matchedIdx)
        if len(matchedIdx) > pattern_filter:
            matchedArray.append(matchedIdx)

    # 각 행기준 매치되는 다른 시퀀스를 찾음
    matchedArray = np.array(matchedArray, dtype=object)

    # 결과 저장 부분
    # 나중에 결과만 그리는 코드 사용시
    np.savez("Data/" + file_name, matchedArray=matchedArray, startIdxList=startIdxList, endIdxList=endIdxList)


# 분석 안하고 불러올 경우
if not do_analysis:
    loadedData = np.load("Data/" + file_name+ ".npz", allow_pickle=True)
    matchedArray = loadedData["matchedArray"]
    startIdxList = loadedData['startIdxList']
    endIdxList = loadedData['endIdxList']




# 행렬 반복순으로 정렬하기
matchedArray = sortInDecendingOrder(matchedArray)
NbPattern = 0
for list in matchedArray:
    if len(list) > pattern_filter:
        NbPattern += 1
print("Before filtering total patterns: ", NbPattern)


# 필터링

runFlag = True
numPost = 0
pre = len(matchedArray)
while runFlag:
    matchedArray = removeDuplicatedPattern(matchedArray, dup_rate)
    print("Total patterns (", numPost, "): ", len(matchedArray))
    numPost += 1
    if len(matchedArray) < pre:
        pre = len(matchedArray)
    else:
        runFlag = False

matchedArray = sortInDecendingOrder(matchedArray)



# 제일 잘맞는 시퀀스들을 그림
if drawBestMatch:

    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.set_title("The number of repetition: " + str(len(matchedArray[bestIdx])))
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')


    for i in range(min(len(matchedArray[bestIdx]),20)):
        x = getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 0])
        y = getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 1])
        z = getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 2])
        draw3DLineOnFig(fig, x, z, y)

# x-z 평면에서 패턴 보기
plt.figure()
plt.title("The number of repetition: " + str(len(matchedArray[bestIdx])))
plt.grid()


plt.xlabel('X Position (mm)')
plt.ylabel('Z Position (mm)')
plt.xlim([min(testdata[:, 0]), max(testdata[:, 0])])
plt.ylim([min(testdata[:, 2]), max(testdata[:, 2])])

for i in range(len(matchedArray[bestIdx])):
    # 가장 많이 일치한 시퀀스의 Idx로 부터 그 시퀀스의 time 데이터에서 시작 위치 찾음
    x = getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 0])
    z = getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 2])
    plt.plot(x,z)

fig = plt.figure()
drawPattern(fig, bestIdx, matchedArray,startIdxList, endIdxList, testdata)

# 예로 하나만 그리기
if drawUptoIdx > 0:
    for i in range(drawUptoIdx):
        fig = plt.figure()
        drawPattern(fig, i, matchedArray,startIdxList, endIdxList, testdata)



plt.show()