import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import PattenAnalysisTool as pat



# Load data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
loadname = 'Tanja two fingers'

dataSet = pd.read_csv('../../Data/' + loadname + '.csv').to_numpy()
print(dataSet.shape)

# 데이터 생성 우선 y축 데이터로 테스트해보자
time = dataSet[:,0]
xyzdata = dataSet[:,1:4]
print(xyzdata.shape)

# 데이터 세워서 합치기
testdata = xyzdata
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
l_interval = 2
R = 150
file_name = loadname + "PatternAnalysis4D_R" + str(R) +"_Lminmax" + str(l_min) + str(l_max)
fig_Idx = 0
pattern_filter = 2
dup_rate = 0.5
# 뭘 그릴지 정함
bestIdx = 0
drawfrom = 0
drawUptoIdx = 5



# 시퀀스 생성
startIdx = 0
sequences=[]
startIdxList = []
endIdxList = []
total_nbSequence = 0

# 시퀀스 생성
if do_analysis:
    for _startIdx in range(0, l_min, l_interval):
        for l_Sequence in range(l_min, l_max + l_interval, l_interval):
            print("Start Idx: ", _startIdx, " Length: ", l_Sequence)
            startIdx = _startIdx
            tempStartIdx = []
            tempEndIdx = []
            while startIdx+l_Sequence<total_length:
                startIdxList.append(startIdx)
                endIdxList.append(startIdx+l_Sequence)

                tempStartIdx.append(startIdx)
                tempEndIdx.append(startIdx + l_Sequence)

                sequences.append(pat.generateSequence(startIdx,l_Sequence,testdata))
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
                        pat.drawSquareColor(time[startIdx:endIdx], testdata[startIdx:endIdx, j], 0.1, 'r')
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
                    if pat.checkOverlapping([startIdxList[k], endIdxList[k]],[startIdxList[j], endIdxList[j]],l_min/4):
                        overlapping = True
                        break

                if not overlapping:
                    # 만약 둘 사이 거리가 일정 이하라면
                    # print(computeDTWVector(sequences[i], sequences[j]))
                    # 둘 사이 거리 비교할때는 원점으로 각 시퀀스를 옮겨서 비교함
                    if pat.computeDTWVector(sequences[i], sequences[j]) < R:
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
matchedArray = pat.sortInDecendingOrder(matchedArray)
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
    matchedArray = pat.removeDuplicatedPattern(matchedArray, dup_rate, startIdxList, endIdxList)
    print("Total patterns (", numPost, "): ", len(matchedArray))
    numPost += 1
    if len(matchedArray) < pre:
        pre = len(matchedArray)
    else:
        runFlag = False

matchedArray = pat.sortInDecendingOrder(matchedArray)



# 제일 잘맞는 시퀀스들을 그림
if drawBestMatch:

    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.set_title("The number of repetition: " + str(len(matchedArray[bestIdx])))
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')


    for i in range(min(len(matchedArray[bestIdx]),20)):
        x = pat.getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 0])
        y = pat.getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 1])
        z = pat.getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 2])
        pat.draw3DLineOnFig(fig, x, z, y)


# 예로 하나만 그리기
if drawUptoIdx > 0:
    for i in range(len(matchedArray)):
        fig = plt.figure()
        # pat.drawPattern3D(fig, i, matchedArray,startIdxList, endIdxList, testdata)
        # pat.drawPatternPlane(fig, i,matchedArray,startIdxList, endIdxList, testdata)
        pat.drawPatternPlaneLimited(fig, i,matchedArray,startIdxList, endIdxList, testdata,[-170, 170], [-100, 100])

plt.show()