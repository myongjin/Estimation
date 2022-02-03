import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import PattenAnalysisTool as pat


# the most recent version of pattern analysis program
# the input is 6dof position data
# draw and save data

# updated on 02/02/2022
# developed by Myeongjin Kim (myeongjin.kim@imperial.ac.uk, proengkim@gmail.com)

# Load data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
# loadname = 'ID014_filtered_MAF_pos_ori3'
loadname = 'ID020_filtered_MAF_pos3'
dataSet = pd.read_csv('../../Data/study/' + loadname + '.csv').to_numpy()

print(dataSet.shape)

#앞 뒤 잘라내기
cutStart = 0
cutEnd = -1
dataSet = dataSet[cutStart:cutEnd, :]

# generate several dataset
time = dataSet[:,0]
firstPos = dataSet[:,1:4]
secondPos = dataSet[:,9:12]

twoPos = np.concatenate([firstPos, secondPos], axis =1)
print(twoPos.shape)

firstForce = dataSet[:, 8]
secondForce = dataSet[:, 16]
print(firstForce.shape)
print(secondForce.shape)

twoForces = np.column_stack([firstForce, secondForce])
print(twoForces.shape)


# 데이터 세워서 합치기
testdata = twoPos
print([min(testdata[:, 0]), max(testdata[:, 0])])
print([min(testdata[:, 2]), max(testdata[:, 2])])
print(testdata.shape)





# check data length
total_length = testdata.shape[0]
print(total_length)

#################################### set variables #############################################
# uniform gap, length, overlapping

# set flags
do_analysis = False
drawGeneratedSequence = False
drawSequencesInTime = False
drawBestMatch = False
drawPatternResult = True
drawForcePattern = True
saveDataForUnity = True
saveFig = True

# set length
l_min = 18
l_max = 26
l_interval = 4

# set threshold
R = 400

# set file name
file_name = loadname + "PatternAnalysis6D_R" + str(R) +"_Lminmax" + str(l_min) + str(l_max)

# variables
fig_Idx = 0
bestIdx = 0
drawfrom = 0


# if the number of repetition is more than the below number, then the repeated sequence is considered as a pattern
pattern_filter = 2
# if two patterns share more than the below dup_rate, then the patterns are merged
dup_rate = 0.8

# sequence variables
startIdx = 0
sequences=[]
startIdxList = []
endIdxList = []
total_nbSequence = 0

# Sequence generation
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

############################################################################################################


# Search pattern
matchedArray = []
# 각 i에 대해
if do_analysis:
    for i in range(total_nbSequence):
        # put itself to a list of repeated sequences
        matchedIdx = [i]
        match = 0
        # compare i sequence with other
        for j in range(total_nbSequence):
            if i != j:
                # check if sequence j is already in the list or not
                overlapping = False
                for k in matchedIdx:
                    if pat.checkOverlapping([startIdxList[k], endIdxList[k]],[startIdxList[j], endIdxList[j]],l_min/4):
                        overlapping = True
                        break

                # if sequence j is not in the list
                if not overlapping:
                    # check the distance between sequence i and j and if the D is less than R, add it to the list
                    if pat.computeDTWVector(sequences[i], sequences[j]) < R:
                        matchedIdx.append(j)

        # print result
        print(matchedIdx)

        # if the number of repeated sequence is higher than threshold, add it to a pattern
        if len(matchedIdx) > pattern_filter:
            matchedArray.append(matchedIdx)

    # convert the result to nparray
    matchedArray = np.array(matchedArray, dtype=object)

    # save the result to load and draw result later
    np.savez("Data/" + file_name, matchedArray=matchedArray, startIdxList=startIdxList, endIdxList=endIdxList)


# load analysed data
if not do_analysis:
    loadedData = np.load("Data/" + file_name+ ".npz", allow_pickle=True)
    matchedArray = loadedData["matchedArray"]
    startIdxList = loadedData['startIdxList']
    endIdxList = loadedData['endIdxList']



# rearrange a pattern list in order of the number of repetition
matchedArray = pat.sortInDecendingOrder(matchedArray)
NbPattern = 0
for list in matchedArray:
    if len(list) > pattern_filter:
        NbPattern += 1
print("Before filtering total patterns: ", NbPattern)


# filtering
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

# save data for unity
if saveDataForUnity:
    pat.savePatternV2ForUnity(matchedArray, startIdxList, endIdxList, loadname + '_pattern.txt')

# draw patterns on 3D
if drawBestMatch and len(matchedArray) > 0:

    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.set_title("Left hand, The number of repetition: " + str(len(matchedArray[bestIdx])))
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    # if the number of sequences in a pattern over 20, then draw up to 20 only
    for i in range(min(len(matchedArray[bestIdx]),20)):
        x = pat.getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 0])
        y = pat.getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 1])
        z = pat.getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 2])
        pat.draw3DLineOnFig(fig, x, z, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("Right hand,The number of repetition: " + str(len(matchedArray[bestIdx])))
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    for i in range(min(len(matchedArray[bestIdx]),20)):
        x = pat.getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 3])
        y = pat.getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 4])
        z = pat.getSequencefromIdx(matchedArray[bestIdx][i], startIdxList, endIdxList, testdata[:, 5])
        pat.draw3DLineOnFig(fig, x, z, y)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca(projection='2d')
    ax.set_title("Right hand,The number of repetition: " + str(len(matchedArray[bestIdx])))
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    # draw two fingertip forces of pattern
    for i in range(min(len(matchedArray[bestIdx]),20)):
        check =0

# draw patterns on 2D
if drawPatternResult:
    for i in range(len(matchedArray)):
        fig = plt.figure(figsize=(10, 5))
        # pat.drawPattern3D(fig, i, matchedArray,startIdxList, endIdxList, testdata)
        # pat.drawPatternPlane(fig, i,matchedArray,startIdxList, endIdxList, testdata)
        pat.draw6DPatternPlaneLimited(fig, i, matchedArray,startIdxList, endIdxList, testdata,[-200, 200], [-100, 100])
        # pat.draw6DPatternPlane(fig, i, matchedArray, startIdxList, endIdxList, testdata)
        if saveFig:
            plt.savefig('Figures/' + file_name + "_" + str(i) + "_dup" + str(dup_rate) + ".png")

# draw forces
if drawForcePattern:

    for i in range(len(matchedArray)):
        fig = plt.figure(figsize=(10,5))

        pat.draw2ForcePatternInTime(fig, matchedArray[i], startIdxList, endIdxList, time, twoForces)

        if saveFig:
            plt.savefig('Figures/' + file_name + "_" + str(i) + "_dup" + str(dup_rate) + "_Force.png")

# plt.show()