import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import PattenAnalysisTool as pat
import multiprocessing
from multiprocessing import Pool, Process
import os

# this version is made to find patterns from all sequences from all participants
# the most recent version of pattern analysis program
# the input is 6 dof position data
# it draws and saves data

# updated on 02/02/2022
# developed by Myeongjin Kim (myeongjin.kim@imperial.ac.uk, proengkim@gmail.com)

# Load data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
# loadname = 'ID011_filtered_MAF_pos_ori10'

if __name__ == "__main__":


    #################################### set variables #############################################
    # uniform gap, length, overlapping

    # set flags
    do_analysis = True
    load_filteredData = False
    analyseAllData = True
    drawGeneratedSequence = False
    drawSequencesInTime = False
    drawBestMatch = False
    drawPatternResult = True
    drawForcePattern = False
    saveDataForUnity = False
    saveFig = True



    # set length
    l_min = 20
    l_max = 50
    l_interval = 5

    # set threshold
    R = 400

    # if the number of repetition is more than the below number, then the repeated sequence is considered as a pattern
    pattern_filter = 2
    # if two patterns share more than the below dup_rate, then the patterns are merged
    dup_rate = 0.7

    if analyseAllData:
        #first trials
        #idList = ['002', '003', '005', '006', '007', '008', '009', '010','011', '014', '015', '016', '018', '019', '020', '024']

        #second trials
        idList = ['013_2', '014_2', '015_2', '017_2', '018_2', '020_2', '024_2', '016_2', '019_2']
    else:
        idList = ['002']

    # sequence variables
    allTestData = []
    startIdx = 0
    sequences = []  # contains array of sequences
    startIdxList = []  # contains start idx list of sequences
    endIdxList = []  # contains end idx list of sequences
    seqIdList = []  # contains id of data where each sequence comes from
    total_nbSequence = 0

    # set file name to save data
    file_name = "All_PatternAnalysis6D_Right_R" + str(R) + "_Lminmax" + str(l_min) + str(l_max)

    # variables
    fig_Idx = 0
    bestIdx = 0
    drawfrom = 0

    ########################################### Generate sequences ##########################################

    for i in range(0, len(idList)):
        loadname = 'ID' + idList[i] + '_filtered_MAF_pos3'
        print(loadname)

        ################load data and filter#####################################################################################
        dataSet = pd.read_csv('../../Data/study/' + loadname + '.csv').to_numpy()

        print(dataSet.shape)

        # cut off the beginning and end part
        cutStart = 0
        cutEnd = -1
        dataSet = dataSet[cutStart:cutEnd, :]

        # generate several dataset
        time = dataSet[:, 0]
        firstPos = dataSet[:, 1:4]
        secondPos = dataSet[:, 9:12]

        twoPos = np.concatenate([firstPos, secondPos], axis=1)
        print("Shape of twoPos:", twoPos.shape)

        firstForce = dataSet[:, 8]
        secondForce = dataSet[:, 16]
        print("Shape of First force:", firstForce.shape)
        print("Shape of Second force:", secondForce.shape)

        twoForces = np.column_stack([firstForce, secondForce])
        print("Shape of two forces:", twoForces.shape)


        ########################################### Set test data ###########################################
        testdata = secondPos
        print("Shape of testdata:", testdata.shape)

        # Check the range of motion
        print("Min Max of first pos, x:", [min(testdata[:, 0]), max(testdata[:, 0])])
        print("Min Max of first pos, z:", [min(testdata[:, 2]), max(testdata[:, 2])])
        allTestData.append(testdata)

        # check data length
        total_length = testdata.shape[0]
        print("Total length of testdata:", total_length)
        ################load data and filter#####################################################################################

        ########################################## Sequence generation ##########################################
        if True:
            # generate sequences
            # temp variable is used for each ID data
            tempStartIdx = []
            tempEndIdx = []

            for _startIdx in range(0, l_min, l_interval):
                for l_Sequence in range(l_min, l_max + l_interval, l_interval):
                    print("Start Idx: ", _startIdx, " Length: ", l_Sequence)
                    startIdx = _startIdx
                    endIdx = startIdx + l_Sequence

                    while endIdx < total_length:
                        # check if new one is not overlapped with existing sequences
                        overlapping = False

                        # use tempList because the duplication check only needs to be done in the same id data
                        for k in range(0, len(tempStartIdx)):
                            if pat.checkOverlapping([tempStartIdx[k], tempEndIdx[k]], [startIdx, endIdx],
                                                    l_min):
                                overlapping = True
                                # print("overlapped")
                                break

                        if not overlapping:
                            startIdxList.append(startIdx)
                            endIdxList.append(endIdx)
                            seqIdList.append(i)

                            tempStartIdx.append(startIdx)
                            tempEndIdx.append(endIdx)

                            sequences.append(pat.generateSequence(startIdx, l_Sequence, testdata))
                            total_nbSequence += 1
                        startIdx += round(l_Sequence * (4 / 5))
                        endIdx = startIdx + l_Sequence
            print("Total sequences: ", total_nbSequence)

            # Draw generated sequences
            if drawGeneratedSequence:
                # draw only x axis for test
                j = 0
                plt.figure(fig_Idx)
                fig_Idx += 1
                plt.plot(time, testdata[:, j])

                plt.xlabel('Time (s)')
                plt.ylabel('X Position (mm)')

                # draw boxes that contains a sequence
                for i in range(len(tempStartIdx)):
                    startIdx = tempStartIdx[i]
                    endIdx = tempEndIdx[i]
                    pat.drawSquareColor(time[startIdx:endIdx], testdata[startIdx:endIdx, j], 0.1, 'r')
                plt.show()

        ############################################################################################################
    print("Sequence generation is done")
    #########################################################################################################

    ########################################### Search pattern ##########################################
    matchedArray = []
    # parallel programming
    processes = []

    def multiProcessingTest(i):
        print('pid of parent', os.getppid())
        print('i value %d : pid %d' % (i, os.getpid()))

        matchedArray.append(i)
        return i * i
    def findMatchedSequencesWith(i):
        # put itself to a list of repeated sequences
        matchedIdx = [i]
        match = 0
        # compare i sequence with other
        for j in range(total_nbSequence):
            # check if sequence j is already in the list or not
            if i != j:
                # check the distance between sequence i and j and if the D is less than R, add it to the list
                if pat.computeDTWVector(sequences[i], sequences[j]) < R:
                    matchedIdx.append(j)

        return matchedIdx
        # print result
        print(matchedIdx)
        # if the number of repeated sequence is higher than threshold, add it to a pattern
        if len(matchedIdx) > pattern_filter:
            matchedArray.append(matchedIdx)


    # for each i
    if do_analysis:

        print("Start pattern searching")
        #Single process
        for i in range(total_nbSequence):
            # put itself to a list of repeated sequences
            matchedIdx = [i]
            match = 0
            # compare i sequence with other
            for j in range(total_nbSequence):
                # check if sequence j is already in the list or not
                if i != j:
                    # check the distance between sequence i and j and if the D is less than R, add it to the list
                    if pat.computeDTWVector(sequences[i], sequences[j]) < R:
                        matchedIdx.append(j)

            # print result
            print(matchedIdx)
            # if the number of repeated sequence is higher than threshold, add it to a pattern
            if len(matchedIdx) > pattern_filter:
                matchedArray.append(matchedIdx)

        # Parallel computing using Pool
        # pool = Pool(processes=4)
        # print(pool.map(multiProcessingTest, range(total_nbSequence)))
        # pool.close()
        # pool.join()
        # print("All process is done")

        # Parallel computing using Process
        # for i in range(total_nbSequence):
        #     p = Process(target=multiProcessingTest(i))
        #     processes.append(p)
        #     p.start()
        #
        # for p in processes:
        #     p.join()
        # print(matchedArray)


        # convert the result to nparray
        matchedArray = np.array(matchedArray, dtype=object)

        # save the result to load and draw result later
        np.savez("Data/" + file_name, matchedArray=matchedArray, startIdxList=startIdxList, endIdxList=endIdxList, seqIdList=seqIdList)
        ######################################################################################################

    ########################################### load analysed data ##########################################
    if not do_analysis and not load_filteredData:
        loadedData = np.load("Data/" + file_name + ".npz", allow_pickle=True)
        matchedArray = loadedData["matchedArray"]
        startIdxList = loadedData['startIdxList']
        endIdxList = loadedData['endIdxList']
        # seqIdList = loadedData['seqIdList']
    ########################################################################################################

    ####################### filtering and save ##############################################
    if not load_filteredData:

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
        ########################################################################################


        # save the filtered result to load and draw result later
        np.savez("Data/" + file_name + "_filtered", matchedArray=matchedArray, startIdxList=startIdxList,
                 endIdxList=endIdxList, seqIdList=seqIdList)

        ######################################################################################################

    ########################################### load analysed data ##########################################
    if not do_analysis and load_filteredData:
        loadedData = np.load("Data/" + file_name  + "_filtered.npz", allow_pickle=True)
        matchedArray = loadedData["matchedArray"]
        startIdxList = loadedData['startIdxList']
        endIdxList = loadedData['endIdxList']
        seqIdList = loadedData['seqIdList']
    ########################################################################################################


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
        for i in range(min(len(matchedArray[bestIdx]), 20)):
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

        for i in range(min(len(matchedArray[bestIdx]), 20)):
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

        plt.show()

    # draw patterns on 2D
    if drawPatternResult:
        for i in range(len(matchedArray)):
            fig = plt.figure(figsize=(7, 5))

            pat.draw3DPatternPlaneLimitedforAllParticipant(fig, i, matchedArray,
                                                           startIdxList, endIdxList, seqIdList,
                                                           idList, allTestData,
                                                           [-200, 200], [-100, 100])
            # pat.draw6DPatternPlane(fig, i, matchedArray, startIdxList, endIdxList, testdata)
            if saveFig:
                plt.savefig('Figures/' + file_name + "_" + str(i) + "_dup" + str(dup_rate) + ".png")

    # draw forces
    if drawForcePattern:
        for i in range(len(matchedArray)):
            fig = plt.figure(figsize=(11, 5))
            pat.draw2ForcePatternInTime(fig, matchedArray[i], startIdxList, endIdxList, time, twoForces)

            if saveFig:
                plt.savefig('Figures/' + file_name + "_" + str(i) + "_dup" + str(dup_rate) + "_Force.png")
