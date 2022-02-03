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
        x = np.zeros([seq_length, 6+8])

        x[:, 0:7] = time_series[i:i + seq_length, 1:8]
        x[:, 7:14] = time_series[i:i + seq_length, 9:16]

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
        x = np.zeros([seq_length, 6+8])
        x[:, 0:7] = time_series[i:i + seq_length, 1:8]
        x[:, 7:14] = time_series[i:i + seq_length, 9:16]

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

def demanualNormalization1D(min, max, data):
    for i in range(len(data)):
        data[i] = data[i] * (max - min) + min
    return data

def manualNormalization2D(min, max, data):
    for i in range(len(data[0,:])):
        for j in range(len(data[:,0])):
            data[i,j] = (data[i,j] - min)/(max - min)
    return data

def make_noise():
  duration = 1000  # milliseconds
  freq = 440  # Hz
  winsound.Beep(freq, duration)

# train Parameters

# 데이터에 따라 단위가 다름 2021년 4월 이후 데이터는 mm 이전은 cm

# train_file_name = '../Data/Traindata16082021.csv'
# train_file_name = '../Data/Traindata2 16-08-2021 15-37.csv'
# train_file_name = '../Data/train 01-09-2021 15-47_filtered.csv'
train_file_name = '../Data/train 01-09-2021 15-51_filtered.csv'




test_file_name = '../Data/BreastDontMoveForce.csv'
# test_file_name = '../Data/BreastMoveForce.csv'
# test_file_name = '../Data/Jennifer_pos_force.csv'
# test_file_name = '../Data/test 01-09-2021 15-53_filtered.csv'


# model_name = 'Palpation_pos_RNN_twofinger_palpation_25042021_2.h5'
# load_model_name = 'Models/Twohands_withOri_force_1hidden_seq10_27082021.h5'
load_model_name = 'Models/Twohands_OnlyPosOri_force_3LSTM_seq17_01092021.h5'
save_model_name = load_model_name

seq_length = 17
data_dim = 6+8
test_size = 1000
output_dim = 2
learning_rate = 0.01
EPOCH = 500
PATIENCE = 10

load_trained_model = True
training = True
draw_TrainResult = True
draw_TestResult = True

# 처음 모델 만드는경우 이 수를 늘려서 충분히 훈련 시킴
if not load_trained_model:
    PATIENCE *= 10



# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
dataSet = pd.read_csv(train_file_name).to_numpy()
testSet = pd.read_csv(test_file_name).to_numpy()


#데이터 자르기
cutStart = 50
cutEnd = -50
dataSet = dataSet[cutStart:cutEnd, :]
testSet = testSet[cutStart:cutEnd, :]

test_pressure = testSet[seq_length-1:-1, 17:17 + 2288]
time = dataSet[seq_length-1:-1, 0]
ttime = testSet[seq_length-1:-1, 0]


# scale data
if False:
    pressureScaler = MinMaxScaler()
    posScaler = MinMaxScaler()

    twoPos = np.concatenate([dataSet[:, 1:4], dataSet[:, 9:12]], axis=1)
    print("two pos shape: ",twoPos.shape)
    posScaler.fit(twoPos)
    twoPos = posScaler.transform(twoPos)
    dataSet[:, 1:4] = twoPos[:, 0:3]
    dataSet[:, 9:12] = twoPos[:, 3:6]

    twoPos = np.concatenate([testSet[:, 1:4], testSet[:, 9:12]], axis=1)
    twoPos = posScaler.transform(twoPos)
    testSet[:, 1:4] = twoPos[:, 0:3]
    testSet[:, 9:12] = twoPos[:, 3:6]

##manual scale
manualNorm = [[-100, 100],[10, 150],[-200, 200]]

for i in range(3):
    dataSet[:,1 + i] = manualNormalization1D(manualNorm[i][0],manualNorm[i][1],dataSet[:,1 + i])
    print(min(dataSet[:,1 + i]), max(dataSet[:,1 + i]))

    dataSet[:, 9 + i] = manualNormalization1D(manualNorm[i][0], manualNorm[i][1], dataSet[:, 9 + i])
    print(min(dataSet[:, 9 + i]), max(dataSet[:, 9 + i]))

dataSet[:,8] = manualNormalization1D(0,8,dataSet[:,8])
print(min(dataSet[:,8]), max(dataSet[:,8]))
dataSet[:,16] = manualNormalization1D(0,8,dataSet[:,16])
print(min(dataSet[:,16]), max(dataSet[:,16]))

for i in range(3):
    testSet[:,1 + i] = manualNormalization1D(manualNorm[i][0],manualNorm[i][1],testSet[:,1 + i])
    print(min(testSet[:,1 + i]), max(testSet[:,1 + i]))

    testSet[:, 9 + i] = manualNormalization1D(manualNorm[i][0], manualNorm[i][1], testSet[:, 9 + i])
    print(min(testSet[:, 9 + i]), max(testSet[:, 9 + i]))

testSet[:,8] = manualNormalization1D(0,8,testSet[:,8])
print(min(testSet[:,8]), max(testSet[:,8]))
testSet[:,16] = manualNormalization1D(0,8,testSet[:,16])
print(min(testSet[:,16]), max(testSet[:,16]))


trainX, trainY = build_dataset(dataSet, seq_length)
testX, testY, testPos = build_datasetWithPos(testSet, seq_length)

print("Train X shape: ", trainX.shape,", Train Y shape: ", trainY.shape)
print("Test X shape: ", testX.shape,", Test Y shape: ", testY.shape)



if load_trained_model:
    tf.model = keras.models.load_model(load_model_name)
else:
    tf.model = tf.keras.Sequential()
    tf.model.add(tf.keras.layers.LSTM(units=200, return_sequences=True, input_shape=(seq_length, data_dim)))
    tf.model.add(tf.keras.layers.LSTM(units=100, return_sequences=True))
    tf.model.add(tf.keras.layers.LSTM(units=50))
    tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='linear'))
    tf.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     metrics=['accuracy'])
tf.model.summary()

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


if training:
    history = tf.model.fit(trainX,
                           trainY,
                           epochs=EPOCH,
                           validation_data=(testX, testY), # 여기서는 scale 아웃풋을 안쓰기 때문에
                           callbacks=[checkpoint, earlystopping])


tf.model = keras.models.load_model(save_model_name)

if draw_TrainResult:
    predicted = tf.model.predict(trainX)

    plt.figure()
    plt.suptitle('Train')
    for i in range(1, 3, 1):
        plt.subplot(2, 1, i)
        data = trainY[:test_size, i - 1]
        data = demanualNormalization1D(0, 8, data)
        y = predicted[:test_size, i - 1]
        y = demanualNormalization1D(0, 8, y)

        plt.ylabel('Force (N)')
        plt.plot(data, 'b-', label='Reference')
        plt.plot(y, 'g-', linewidth=2, label='Predicted')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()

if draw_TestResult:
    predicted = tf.model.predict(testX)

    plt.figure(figsize=(9,8))
    plt.suptitle('Test_position and orientation input')
    for i in range(1, 3, 1):
        plt.subplot(2, 1, i)
        data = testY[:test_size, i - 1]
        data = demanualNormalization1D(0, 8, data)
        y = predicted[:test_size, i - 1]
        y = demanualNormalization1D(0,8,y)
        plt.ylabel('Force (N)')
        plt.plot(ttime[:test_size], data, 'b-', label='Reference')
        plt.plot(ttime[:test_size], y, 'g-', linewidth=2, label='Predicted')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()

    fig = plt.figure()
    pltIndex = [1, 3, 5, 2, 4, 6]
    plt.suptitle('Left')
    for i in range(1, 4, 1):
        plt.subplot(3, 1, i)
        data = testPos[:test_size, i - 1]

        if i == 1:
            plt.ylabel('X axis (mm)')
        if i == 2:
            plt.ylabel('Y axis (mm)')
        if i == 3:
            plt.ylabel('Z axis (mm)')
        plt.plot(ttime[:test_size], data, 'b-')
        plt.xlabel('Time (s)')
        plt.grid()

    fig = plt.figure()
    plt.suptitle('Right')
    for i in range(1, 4, 1):
        plt.subplot(3, 1, i)
        data = testPos[:test_size, i + 2]

        if i == 1:
            plt.ylabel('X axis (mm)')
        if i == 2:
            plt.ylabel('Y axis (mm)')
        if i == 3:
            plt.ylabel('Z axis (mm)')
        plt.plot(ttime[:test_size], data, 'b-')
        plt.xlabel('Time (s)')
        plt.grid()

    # Print accuracy
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
