import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Build data set
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2288)[17~]
def build_dataset(time_series, seq_length):
    trainData = []
    testData = []
    for i in range(0, len(time_series) - seq_length):
        x = np.zeros([seq_length, 2288+6])

        x[:, 0:3] = time_series[i:i + seq_length, 1:4]
        x[:, 3:6] = time_series[i:i + seq_length, 9:12]
        x[:, 6:] = time_series[i:i + seq_length, 17:17 + 2288]

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
        x = np.zeros([seq_length, 2288+6])
        x[:, 0:3] = time_series[i:i + seq_length, 1:4]
        x[:, 3:6] = time_series[i:i + seq_length, 9:12]
        x[:, 6:] = time_series[i:i + seq_length, 17:17 + 2288]

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
        x = np.zeros(2288 + 6)
        x[0:3] = time_series[i + seq_length, 1:4]
        x[3:6] = time_series[i + seq_length, 9:12]
        x[6:] = time_series[i:i + seq_length, 17:]

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
# train Parameters

# 데이터에 따라 단위가 다름 2021년 4월 이후 데이터는 mm 이전은 cm
# train_file_name = '../Data/Palpation_twoFinger 17-05-2021 13-58.csv'
train_file_name = '../Data/traindata.csv'

# 붙여서 한거
# test_file_name = '../Data/Palpation_twoFinger_test 17-05-2021 14-00.csv'
# 안붙여서 한거
test_file_name = '../Data/BreastMoveForce.csv'
# test_file_name = '../Data/Tanja two fingers.csv'
# test_file_name = '../Data/Tanja all fingers.csv'


# model_name = 'Palpation_pos_RNN_twofinger_palpation_25042021_2.h5'
load_model_name = 'Models/Twohands_force_1hidden_seq5_16082021.h5'
save_model_name = load_model_name

seq_length = 5
data_dim = 2294
test_size = 200
output_dim = 2
learning_rate = 0.01
EPOCH = 500
PATIENCE = 20

load_trained_model = False
training = True
draw_TrainResult = False
draw_TestResult = True

# 처음 모델 만드는경우 이 수를 늘려서 충분히 훈련 시킴
if not load_trained_model:
    PATIENCE *= 10



# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
dataSet = pd.read_csv(train_file_name).to_numpy()
testSet = pd.read_csv(test_file_name).to_numpy()


#데이터 자르기
cutStart = 100
cutEnd = -100
dataSet = dataSet[cutStart:cutEnd, :]
testSet = testSet[cutStart:cutEnd, :]

test_pressure = testSet[seq_length-1:-1, 17:]
time = dataSet[seq_length-1:-1, 0]
ttime = testSet[seq_length-1:-1, 0]

# data
# trainX, trainY = build_dataset(dataSet, seq_length)
# testX, testY = build_dataset(testSet, seq_length)

trainX, trainY = build_dataset(dataSet, seq_length)
testX, testY, testPos = build_datasetWithPos(testSet, seq_length)

print("Train X shape: ", trainX.shape,", Train Y shape: ", trainY.shape)
print("Test X shape: ", testX.shape,", Test Y shape: ", testY.shape)


# scale data
scaler = MinMaxScaler()
scaler.fit(trainY)

# scale trainY and testY according to the range of trainY
trainY = scaler.transform(trainY)
scaled_testY = scaler.transform(testY)
# print(trainY)


if load_trained_model:
    tf.model = keras.models.load_model(load_model_name)
else:
    tf.model = tf.keras.Sequential()
    tf.model.add(tf.keras.layers.LSTM(units=1000, input_shape=(seq_length, data_dim)))
    tf.model.add(tf.keras.layers.Dense(units=100, activation='relu'))
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
                           validation_data=(testX, scaled_testY), # scale된 아웃풋이 나오기 때문에 이렇게 함
                           callbacks=[checkpoint, earlystopping])


tf.model = keras.models.load_model(save_model_name)

if draw_TrainResult:
    predicted = tf.model.predict(trainX)
    predicted = scaler.inverse_transform(predicted)
    trainY = scaler.inverse_transform(trainY)

    plt.figure()
    plt.suptitle('Train')
    for i in range(1, 3, 1):
        plt.subplot(2, 1, i)
        data = trainY[:test_size, i - 1]
        y = predicted[:test_size, i - 1]

        plt.ylabel('Force (N)')
        plt.plot(data, 'b-', label='Reference')
        plt.plot(y, 'g-', linewidth=2, label='Predicted')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()

if draw_TestResult:
    predicted = tf.model.predict(testX)
    predicted = scaler.inverse_transform(predicted)

    plt.figure()
    plt.suptitle('Test')
    for i in range(1, 3, 1):
        plt.subplot(2, 1, i)
        data = testY[:test_size, i - 1]
        y = predicted[:test_size, i - 1]

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
    print('Index Finger: RMS Force (N)')
    print(compute1DRMS(testY[:, 0], predicted[:, 0]))
    # print(compute3DRMS(testY[:test_size, 0:3],predicted[:test_size, 0:3]))

    print('Index Finger: Max Force (N)')
    print(np.max(np.abs(testY[:, 0]-predicted[:, 0])))

    print('Middle Finger: RMS Force (N)')
    print(compute1DRMS(testY[:, 1], predicted[:, 1]))

    print('Middle Finger: Max Force (N)')
    print(np.max(np.abs(testY[:, 1]-predicted[:, 1])))


plt.show()
