import sys
import os

# 상위 폴더 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import PattenAnalysisTool as pat
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # set flags
    load_filteredData = False
    analyseAllData = False
    drawGeneratedSequence = True

    # set length
    l_min = 20
    l_max = 50
    l_interval = 5


    if analyseAllData:
        idList = ['013_2', '014_2', '015_2', '017_2', '018_2', '020_2', '024_2', '016_2', '019_2']
        # set file name to save data
        file_name = "AllSequences"
    else:
        idList = ['002']
        # set file name to save data
        file_name = "ID002_Sequence"

    # sequence variables
    allTestData = []
    startIdx = 0
    sequences = []  # contains array of sequences
    startIdxList = []  # contains start idx list of sequences
    endIdxList = []  # contains end idx list of sequences
    seqIdList = []  # contains id of data where each sequence comes from
    total_nbSequence = 0



    # variables
    fig_Idx = 0
    bestIdx = 0
    drawfrom = 0

    for i in range(0, len(idList)):
        loadname = 'ID' + idList[i] + '_filtered_MAF_pos3'
        print(loadname)

        dataSet = pd.read_csv('../../../Data/study/' + loadname + '.csv').to_numpy()
        print(dataSet.shape)

        # cut off the beginning and end part
        cutStart = 0
        cutEnd = -1
        dataSet = dataSet[cutStart:cutEnd, :]

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

        testdata = twoPos
        print("Shape of testdata:", testdata.shape)

        print("Min Max of first pos, x:", [min(testdata[:, 0]), max(testdata[:, 0])])
        print("Min Max of first pos, z:", [min(testdata[:, 2]), max(testdata[:, 2])])

        total_length = testdata.shape[0]
        print("Total length of testdata:", total_length)

        if True:
            for _startIdx in range(0, total_length - l_min, l_interval):
                for l_Sequence in range(l_min, l_max + l_interval, l_interval):
                    print("Start Idx: ", _startIdx, " Length: ", l_Sequence)
                    startIdx = _startIdx
                    endIdx = startIdx + l_Sequence

                    while endIdx < total_length:
                        overlapping = False

                        for k in range(0, len(startIdxList)):
                            if pat.checkOverlapping([startIdxList[k], endIdxList[k]], [startIdx, endIdx], l_min):
                                overlapping = True
                                break

                        if not overlapping:
                            startIdxList.append(startIdx)
                            endIdxList.append(endIdx)
                            seqIdList.append(i)

                            sequences.append(pat.generateSequence(startIdx, l_Sequence, testdata))

                            total_nbSequence += 1
                        startIdx += round(l_Sequence * (4 / 5))
                        endIdx = startIdx + l_Sequence
            print("Total sequences: ", total_nbSequence)

            if drawGeneratedSequence:
                j = 0
                plt.figure(fig_Idx)
                fig_Idx += 1
                plt.plot(time, testdata[:, j])

                plt.xlabel('Time (s)')
                plt.ylabel('X Position (mm)')

                for i in range(len(startIdxList)):
                    if seqIdList[i] == seqIdList[-1]:
                        startIdx = startIdxList[i]
                        endIdx = endIdxList[i]
                        pat.drawSquareColor(time[startIdx:endIdx], testdata[startIdx:endIdx, j], 0.1, 'r')
                plt.show()

    print("Sequence generation is done")

    # Save sequences and related information
    sequence_data = {
        'sequences': sequences,
        'startIdxList': startIdxList,
        'endIdxList': endIdxList,
        'seqIdList': seqIdList,
        'idList': idList,
        'l_min': l_min,
        'l_max': l_max,
        'l_interval': l_interval,
        'total_nbSequence': total_nbSequence
    }
    pat.save_sequences(sequence_data, file_name + '.pkl')

