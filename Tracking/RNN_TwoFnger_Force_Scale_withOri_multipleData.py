import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import winsound

# Build data set
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2288)[17~]
def build_dataset(time_series, seq_length):
    trainData = []
    testData = []
    for i in range(0, len(time_series) - seq_length):
        x = np.zeros([seq_length, 2288+6+8])

        x[:, 0:7] = time_series[i:i + seq_length, 1:8]
        x[:, 7:14] = time_series[i:i + seq_length, 9:16]
        x[:, 14:] = time_series[i:i + seq_length, 17:17 + 2288]

        #print(x[:, 16:].shape)
        #print(time_series[i:i + seq_length, 17:17 + 2288].shape)
        y = np.zeros(2)
        y[0] = time_series[i + seq_length, 8]
        y[1] = time_series[i + seq_length, 16]

        trainData.append(x)
        testData.append(y)

    return np.array(trainData), np.array(testData)


def build_datasetWithPos(time_series, seq_length):
    posData = []
    input = []
    output = []
    for i in range(0, len(time_series) - seq_length):
        x = np.zeros([seq_length, 2288+6+8])
        x[:, 0:7] = time_series[i:i + seq_length, 1:8]
        x[:, 7:14] = time_series[i:i + seq_length, 9:16]
        x[:, 14:] = time_series[i:i + seq_length, 17:17 + 2288]

        y = np.zeros(2)
        y[0] = time_series[i + seq_length, 8]
        y[1] = time_series[i + seq_length, 16]

        pos = np.zeros(6)
        pos[0:3] = time_series[i + seq_length, 1:4]
        pos[3:6] = time_series[i + seq_length, 9:12]

        input.append(x)
        output.append(y)
        posData.append(pos)

    return np.array(input), np.array(output), np.array(posData)

# 하나의 데이터를 train과 test로 나누는것
def build_datasetV2(time_series, seq_length, train_ratio):
    trainX = []
    trainY = []
    testX = []
    testY = []
    for i in range(0, len(time_series) - seq_length):
        x = np.zeros([seq_length, 2288+6+8])
        x[0:7] = time_series[i + seq_length, 1:4]
        x[7:14] = time_series[i + seq_length, 9:12]
        x[16:] = time_series[i:i + seq_length, 17:17 + 2288]

        y = np.zeros(2)
        y[0] = time_series[i + seq_length, 8]
        y[1] = time_series[i + seq_length, 16]

        if i < (len(time_series) - seq_length)*train_ratio:
            trainX.append(x)
            trainY.append(y)
        else:
            testX.append(x)
            testY.append(y)

    return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)

def filter_data(data, pressure, threshold):
    return_data = []
    for i in range(len(pressure[:, 0])):
        sumValue = sum(pressure[i, :])
        print(sumValue)
        if sumValue > threshold:
            return_data.append(data[i, :])
    return np.array(return_data)

def filter_time(time, pressure, threshold):
    return_data = []
    for i in range(len(pressure[:, 0])):
        sumValue = sum(pressure[i, :])
        print(sumValue)
        if sumValue > threshold:
            return_data.append(time[i])
    return np.array(return_data)

def compute1DRMS(dataA, dataB):
    rms = 0
    for i in range(len(dataA)):
        rms += (dataA[i]-dataB[i])**2

    return (rms/len(dataA))**(1/2)

def compute3DRMS(dataA, dataB):
    rms = 0
    for i in range(len(dataA)):
        rms += np.linalg.norm(dataA[i,:]-dataB[i,:])**2

    return (rms/len(dataA))**(1/2)


def manualNormalization1D(min, max, data):
    for i in range(len(data)):
        data[i] = (data[i] - min) / (max - min)
    return data

def cutOffNegative(data):
    for i in range(len(data)):
        if data[i]<0:
            data[i] = 0
    return data

def demanualNormalization1D(min, max, data):
    for i in range(len(data)):
        data[i] = data[i] * (max - min) + min
    return data

def manualNormalization2D(min, max, data):
    for i in range(len(data[:,0])):
        for j in range(len(data[0,:])):
            data[i,j] = (data[i,j] - min)/(max - min)
    return data

def make_noise():
  duration = 1000  # milliseconds
  freq = 440  # Hz
  winsound.Beep(freq, duration)

def manualFullDataNormalization(manualNorm, manualForceNorm, dataSet):
    #pressure normalization
    # dataSet[:, 17:] = manualNormalization2D(0, 2.698211E-05 * 255, dataSet[:, 17:])
    print("check min max of pressure")
    print(minMax2DArray(dataSet[:, 17:]))


    #position normalization
    print("check min max of position")
    for i in range(3):
        dataSet[:, 1 + i] = manualNormalization1D(manualNorm[i][0], manualNorm[i][1], dataSet[:, 1 + i])
        print(min(dataSet[:, 1 + i]), max(dataSet[:, 1 + i]))

        dataSet[:, 9 + i] = manualNormalization1D(manualNorm[i][0], manualNorm[i][1], dataSet[:, 9 + i])
        print(min(dataSet[:, 9 + i]), max(dataSet[:, 9 + i]))


    #force normalization
    print("check min max of force")
    dataSet[:, 8] = manualNormalization1D(manualForceNorm[0], manualForceNorm[1], dataSet[:, 8])
    print(min(dataSet[:, 8]), max(dataSet[:, 8]))
    dataSet[:, 16] = manualNormalization1D(manualForceNorm[0], manualForceNorm[1], dataSet[:, 16])
    print(min(dataSet[:, 16]), max(dataSet[:, 16]))

    return dataSet

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
# train Parameters

# 데이터에 따라 단위가 다름 2021년 4월 이후 데이터는 mm 이전은 cm
train_file_list = []

dataSet = 5

if dataSet == 1:
    train_file_name = '../Data/study/ID002_filtered.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID003_filtered.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID005_filtered.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID007_filtered.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID008_filtered.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID009_filtered.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID011_filtered.csv'
    train_file_list.append(train_file_name)

if dataSet == 2:
    train_file_name = '../Data/Traindata16082021.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/Traindata2 16-08-2021 15-37.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/train 01-09-2021 15-47_filtered.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/train 01-09-2021 14-16_filtered.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/Train_smallforce 16-09-2021 12-37.csv'
    train_file_list.append(train_file_name)

if dataSet == 3:
    train_file_name = '../Data/study/ID002_filtered_MAF10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID003_filtered_MAF10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID005_filtered_MAF10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID007_filtered_MAF10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID008_filtered_MAF10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID009_filtered_MAF10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID011_filtered_MAF10.csv'
    train_file_list.append(train_file_name)

if dataSet == 4:
    train_file_name = '../Data/study/ID002_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID003_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID005_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID007_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID008_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID009_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    #train_file_name = '../Data/study/ID006_filtered_MAF_pos_ori10.csv'
    #train_file_list.append(train_file_name)

if dataSet == 5:
    train_file_name = '../Data/study/ID002_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID003_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID005_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID006_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID007_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID008_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID009_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID010_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)
    train_file_name = '../Data/study/ID011_filtered_MAF_pos_ori10.csv'
    train_file_list.append(train_file_name)

    #train_file_name = '../Data/study/ID006_filtered_MAF_pos_ori10.csv'
    #train_file_list.append(train_file_name)

# 이건 잠시 쓰지 말자
# train_file_name = '../Data/train 01-09-2021 15-51_filtered.csv'



test_file_list=[]
# test_file_name = '../Data/study/ID006_filtered.csv'
test_file_name = '../Data/study/ID010_filtered_MAF_pos_ori10.csv'

# test_file_name = '../Data/BreastDontMoveForce.csv'
# test_file_name = '../Data/BreastMoveForce.csv'
# test_file_name = '../Data/Jennifer_pos_force.csv'
# test_file_name = '../Data/test 01-09-2021 15-53_filtered.csv'
# test_file_name = '../Data/study/ID002_filtered_LPF.csv'
# test_file_name = '../Data/test_smallforce 16-09-2021 12-38.csv'

# 원래 훈련 쓰던건데 테스트로 쓰는중
# test_file_name = '../Data/train 01-09-2021 15-51_filtered.csv'

# model_name = 'Palpation_pos_RNN_twofinger_palpation_25042021_2.h5'


# 여러 데이터로 훈련, 내 데이터로 훈련한건 엄청 잘됨, 압력을 정규화 하지 않았을때 잘됨
# load_model_name = 'Models/Twohands_withOri_force_2LSTM_seq10_23092021.h5'

# 여러 데이터로 훈련, 내 데이터로 압력을 정규화 했을때, 안됨
# load_model_name = 'Models/Twohands_withOri_force_2LSTM_seq10_23092021_2.h5'

# 여러 데이터로, 압력 정규화 안하고 의사들 데이터로, 힘을 키웠을때, 그나마 조금 되지만 아직 한참 멀었음
load_model_name = 'Models/Twohands_withOri_force_2LSTM_seq10_23092021_3.h5'

# 여러 데이터로, 압력 안쓰고 의사들 데이터로, 힘을 키웠을때, 그나마 조금 되지만 아직 한참 멀었음
# load_model_name = 'Models/Twohands_withOri_force_2LSTM_seq10_24092021_3.h5'

save_model_name = load_model_name



seq_length = 10
data_dim = 2288+6+8
test_size = 1000
output_dim = 2
learning_rate = 0.01
EPOCH = 500
PATIENCE = 20

load_trained_model = True
training = True
draw_TrainResult = True
draw_TestResult = True


# 모델 정의
if load_trained_model:
    tf.model = keras.models.load_model(load_model_name)
else:
    tf.model = tf.keras.Sequential()
    tf.model.add(tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=(seq_length, data_dim)))
    tf.model.add(tf.keras.layers.LSTM(units=50))
    tf.model.add(tf.keras.layers.Dense(units=output_dim))
    tf.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     metrics=['accuracy'])
tf.model.summary()


############ 테스트 데이터 하나 불러오기 ############
testSet = pd.read_csv(test_file_name).to_numpy()
test_pressure = testSet[seq_length-1:-1, 17:17 + 2288]
ttime = testSet[seq_length-1:-1, 0]

##manual scale, 힘 키우기
manualNorm = [[-100, 100],[10, 150],[-200, 200]]
manualForceNorm = [0, 0.1]
testSet = manualFullDataNormalization(manualNorm, manualForceNorm, testSet)





# testX, testY, testPos = build_datasetWithPos(testSet, seq_length)
testX, testY = build_dataset(testSet, seq_length)
print("Test X shape: ", testX.shape,", Test Y shape: ", testY.shape)

# 여러개의 훈련 데이터를 불러오고 훈련 시키기
if training:
    for trainNum in range(len(train_file_list)):
        print("Training file index: ", trainNum)
        train_file_name = train_file_list[trainNum]

        # time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
        dataSet = pd.read_csv(train_file_name).to_numpy()
        time = dataSet[seq_length - 1:-1, 0]

        ##manual scale
        dataSet = manualFullDataNormalization(manualNorm, manualForceNorm, dataSet)

        trainX, trainY = build_dataset(dataSet, seq_length)
        print("Train X shape: ", trainX.shape, ", Train Y shape: ", trainY.shape)


        # Set check point
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_model_name,             # file명을 지정합니다
                                     monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                                     verbose=1,            # 로그를 출력합니다
                                     save_best_only=True,  # 가장 best 값만 저장합니다
                                     mode='auto'           # auto는 알아서 best를 찾습니다. min/max
                                    )

        # Set early stopping
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience= PATIENCE)



        history = tf.model.fit(trainX,
                                trainY,
                                epochs=EPOCH,
                                validation_data=(testX, testY), # 여기서는 scale 아웃풋을 안쓰기 때문에
                                callbacks=[checkpoint, earlystopping])


tf.model = keras.models.load_model(save_model_name)

if draw_TrainResult:
    for trainNum in range(len(train_file_list)):
        print("Training file index: ", trainNum)
        train_file_name = train_file_list[trainNum]

        # time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
        dataSet = pd.read_csv(train_file_name).to_numpy()
        time = dataSet[seq_length - 1:-1, 0]

        #manual scale
        dataSet = manualFullDataNormalization(manualNorm, manualForceNorm, dataSet)
        trainX, trainY = build_dataset(dataSet, seq_length)

        predicted = tf.model.predict(trainX)

        plt.figure(figsize=(9,8))
        plt.suptitle('Train')
        for i in range(1, 3, 1):
            plt.subplot(2, 1, i)
            data = trainY[:test_size, i - 1]
            data = demanualNormalization1D(manualForceNorm[0], manualForceNorm[1], data)
            y = predicted[:test_size, i - 1]
            y = demanualNormalization1D(manualForceNorm[0], manualForceNorm[1], y)

            plt.ylabel('Force (N)')
            plt.plot(time[:test_size], data, 'b-', label='Reference')
            plt.plot(time[:test_size], y, 'g-', linewidth=2, label='Predicted')
            plt.xlabel('Time (s)')
            plt.grid()
            plt.legend()

if draw_TestResult:
    predicted = tf.model.predict(testX)

    plt.figure(figsize=(9,8))
    plt.suptitle('Test_position and orientation input')
    for i in range(1, 3, 1):
        plt.subplot(2, 1, i)
        data = testY[:test_size, (i - 1)]
        data = demanualNormalization1D(manualForceNorm[0], manualForceNorm[1], data)
        y = predicted[:test_size, (i - 1)]
        y = demanualNormalization1D(manualForceNorm[0], manualForceNorm[1],y)
        plt.ylabel('Force (N)')
        plt.plot(ttime[:test_size], data, 'b-', label='Reference')
        plt.plot(ttime[:test_size], y, 'g-', linewidth=2, label='Predicted')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()



    # Print accuracy
    print('Test Result')
    print('Left Finger: RMS Force (N)')

    print('%.3f' % compute1DRMS(testY[:, 0], predicted[:, 0]))
    # print(compute3DRMS(testY[:test_size, 0:3],predicted[:test_size, 0:3]))

    print('Left Finger: Max Force (N)')
    print('%.3f' % np.max(np.abs(testY[:, 0]-predicted[:, 0])))

    print('Right Finger: RMS Force (N)')
    print('%.3f' % compute1DRMS(testY[:, 1], predicted[:, 1]))

    print('Right Finger: Max Force (N)')
    print('%.3f' % np.max(np.abs(testY[:, 1]-predicted[:, 1])))

make_noise()
plt.show()
