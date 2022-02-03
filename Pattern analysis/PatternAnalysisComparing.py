import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import PattenAnalysisTool as pat

# Load data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
loadname1 = 'Tanja two fingers'
loadname2 = 'Jenny_two 05-07-2021 14-37'

loadNames = [loadname1, loadname2]

# 앞 뒤 잘라내기
cutStart = 100
cutEnd = -100
testDatas = []
# for name in loadNames:
data = pd.read_csv('../../Data/' + loadNames[0] + '.csv').to_numpy()
testDatas.append(data[cutStart:700, 1:4])

data = pd.read_csv('../../Data/' + loadNames[1] + '.csv').to_numpy()
testDatas.append(data[cutStart:cutEnd, 1:4])


# 변수 설정
# 균일 간격 생성, 균일 길이, 하지만 겹치게 생성
do_analysis = True
drawGeneratedSequence = False
drawSequencesInTime = False
drawBestMatch = True
l_min = 50
l_max = 60
l_interval = 5
R = 2000
file_name = loadname1 + "Tanja_Jennifer_R" + str(R) + "_Lminmax" + str(l_min) + str(l_max)
fig_Idx = 0
pattern_filter = 1
dup_rate = 0.4
# 뭘 그릴지 정함
bestIdx = 0
drawfrom = 0
drawUptoIdx = 5

# 시퀀스 생성
startIdx = 0
sequences = []
startIdxList = []
endIdxList = []
total_nbSequence = 0

# 시퀀스 생성
if do_analysis:
    for dataIdx in range(len(testDatas)):
        data = testDatas[dataIdx]
        print('data Idx: {0}, Length: {1}'.format(dataIdx, len(testDatas[dataIdx])))
        subSequences = []
        subStartList = []
        subEndList = []
        for _startIdx in range(0, l_min, l_interval):
            for l_Sequence in range(l_min, l_max + l_interval, l_interval):
                print("Start Idx: ", _startIdx, " Length: ", l_Sequence)
                startIdx = _startIdx
                tempStartIdx = []
                tempEndIdx = []
                while startIdx + l_Sequence < len(data):
                    # 몇번째 데이터에서 온 시퀀스인지도 저장
                    subStartList.append(startIdx)
                    subEndList.append(startIdx + l_Sequence)

                    tempStartIdx.append(startIdx)
                    tempEndIdx.append(startIdx + l_Sequence)

                    # 원점으로 옮겨놓은 시퀀스를 추가함
                    subSequences.append(pat.generateSequence(startIdx, l_Sequence, data))
                    total_nbSequence += 1
                    startIdx += round(l_Sequence * (3 / 4))

        startIdxList.append(subStartList)
        endIdxList.append(subEndList)
        sequences.append(subSequences)

    print("Total sequences: ", total_nbSequence)

# 서치 하기, 저장할 값, 매치 Index
matchedArray = []
# 각 i에 대해
if do_analysis:
    # 어차피 타 데이터와 비교니까 i =0을 기준으로 다른거와 비교하는 걸로 하자
    i = 0
    seqsA = sequences[i]
    # A와 B안의 시퀀스 비교



    # A안의 각 시퀀스에 대해
    for ii in range(len(seqsA)):
        # 우선 자기 자신 추가
        matchedIdx = []
        matchedIdx.append([i, ii])

        # 각 시퀀스 B에 대해
        for j in range(i + 1, len(sequences)):
            seqsB = sequences[j]

            smallestR = 1000000000000
            bestMatchedIdx = []
            matchedFlag = False
            # 기준 A와 가장 잘 맞는 시퀀스 찾음
            for jj in range(len(seqsB)):
                value = pat.computeDTWVector(seqsA[ii], seqsB[jj])
                # 기준보다 작으면서 작은 애들중 가장 작은애를 찾음
                if value < R and value < smallestR:
                    bestMatchedIdx = [j, jj]
                    matchedFlag = True
            # 만약 찾았다면 추가함
            if matchedFlag:
                matchedIdx.append(bestMatchedIdx)

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
    loadedData = np.load("Data/" + file_name + ".npz", allow_pickle=True)
    matchedArray = loadedData["matchedArray"]
    startIdxList = loadedData['startIdxList']
    endIdxList = loadedData['endIdxList']

# matchedArray[pattern Idx][data Idx, seq Idx]
# 행렬 반복순으로 정렬하기
# matchedArray = pat.sortInDecendingOrder(matchedArray)
print("Before filtering total patterns: ", len(matchedArray))
print(matchedArray)


# 제일 잘맞는 시퀀스들을 그림
if drawBestMatch:

    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.set_title("The number of repetition: " + str(len(matchedArray[bestIdx])))
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    for i in range(min(len(matchedArray[bestIdx]), 20)):
        dataIdx = matchedArray[bestIdx][i][0]
        seqIdx = matchedArray[bestIdx][i][1]
        x = pat.getSequencefromIdx(seqIdx, startIdxList[dataIdx], endIdxList[dataIdx], testDatas[dataIdx][:, 0])
        y = pat.getSequencefromIdx(seqIdx, startIdxList[dataIdx], endIdxList[dataIdx], testDatas[dataIdx][:, 1])
        z = pat.getSequencefromIdx(seqIdx, startIdxList[dataIdx], endIdxList[dataIdx], testDatas[dataIdx][:, 2])
        pat.draw3DLineOnFig(fig, x, z, y)



# 평면에 그리기
if drawUptoIdx > 0:
    print('The number of patterns')
    print(len(matchedArray))
    for i in range(0, len(matchedArray), 5):
        fig = plt.figure()
        pat.drawPatternPlaneV2Limited(fig, i, matchedArray, startIdxList, endIdxList, testDatas,[-170, 170], [-100, 100])

plt.show()
