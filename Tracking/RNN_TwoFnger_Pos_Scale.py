import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Build data set
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        x = time_series[i:i + seq_length, 17:]
        y = np.zeros(6)
        y[0:3] = time_series[i + seq_length, 1:4]
        y[3:6] = time_series[i + seq_length, 9:12]

        dataX.append(x)
        dataY.append(y)

    return np.array(dataX), np.array(dataY)

def build_datasetV2(time_series, seq_length, train_ratio):
    trainX = []
    trainY = []
    testX = []
    testY = []
    for i in range(0, len(time_series) - seq_length):
        x = time_series[i:i + seq_length, 17:]
        y = np.zeros(6)
        y[0:3] = time_series[i + seq_length, 1:4]
        y[3:6] = time_series[i + seq_length, 9:12]

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
train_file_name = '../Data/two_hands_move.csv'

# 붙여서 한거
# test_file_name = '../Data/Palpation_twoFinger_test 17-05-2021 14-00.csv'
# 안붙여서 한거
test_file_name = '../Data/two_hands_move_test.csv'
# test_file_name = '../Data/Tanja two fingers.csv'
# test_file_name = '../Data/Tanja all fingers.csv'


# model_name = 'Palpation_pos_RNN_twofinger_palpation_25042021_2.h5'
load_model_name = 'Models/Twohands_move_fullmodel_1hidden_10082021.h5'
save_model_name = load_model_name

seq_length = 10
data_dim = 2288
test_size = 200
output_dim = 6
learning_rate = 0.01
EPOCH = 500
PATIENCE = 20

load_trained_model = True
training = True
draw_TrainResult = False
draw_TestResult = True
use_Contact_Filter = False
threshold = 3.25

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
testX, testY = build_dataset(testSet, seq_length)

print("Train X shape: ", trainX.shape,", Train Y shape: ", trainY.shape)
print("Test X shape: ", testX.shape,", Test Y shape: ", testY.shape)


# scale data
scaler = MinMaxScaler()
scaler.fit(trainY)
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
                           validation_data=(testX, scaled_testY),
                           callbacks=[checkpoint, earlystopping])


tf.model = keras.models.load_model(save_model_name)

if draw_TrainResult:
    predicted = tf.model.predict(trainX)
    predicted = scaler.inverse_transform(predicted)
    trainY = scaler.inverse_transform(trainY)

    plt.figure(1)
    plt.suptitle('Training: Index finger')
    for i in range(1,4,1):
        plt.subplot(3,1,i)
        data = trainY[:, i-1]
        y = predicted[:,i-1]

        if i==1:
            plt.ylabel('X axis (mm)')
        if i==2:
            plt.ylabel('Y axis (mm)')
        if i==3:
            plt.ylabel('Z axis (mm)')
        plt.plot( data, 'b-', label='Reference')
        plt.plot( y, 'g-', linewidth=2, label='Predicted')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()

    plt.figure(2)
    plt.suptitle('Training: Middle finger')
    for i in range(1, 4, 1):
        plt.subplot(3, 1, i)
        data = trainY[:, i + 2]
        y = predicted[:, i + 2]
        if i==1:
            plt.ylabel('X axis (mm)')
        if i==2:
            plt.ylabel('Y axis (mm)')
        if i==3:
            plt.ylabel('Z axis (mm)')
        plt.plot( data, 'b-', label='Reference')
        plt.plot( y, 'g-', linewidth=2, label='Predicted')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()


if draw_TestResult:
    predicted = tf.model.predict(testX)
    predicted = scaler.inverse_transform(predicted)

    # filter
    if use_Contact_Filter:

        print('before', predicted.shape)

        predicted = filter_data(predicted, test_pressure, threshold)
        testY = filter_data(testY, test_pressure, threshold)

        time = filter_time(time, test_pressure, threshold)
        print('after', predicted.shape)

    plt.figure(3)
    plt.suptitle('Test: Index finger')
    for i in range(1, 4, 1):
        plt.subplot(3, 1, i)
        data = testY[:test_size, i - 1]
        y = predicted[:test_size, i - 1]
        if i==1:
            plt.ylabel('X axis (mm)')
        if i==2:
            plt.ylabel('Y axis (mm)')
        if i==3:
            plt.ylabel('Z axis (mm)')

        plt.plot(data, 'b-', label='Reference')
        plt.plot(y, 'g-', linewidth=2, label='Predicted')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()


    plt.figure(4)
    plt.suptitle('Test: Middle finger')
    for i in range(1, 4, 1):
        plt.subplot(3, 1, i)
        data = testY[:test_size, i + 2]
        y = predicted[:test_size, i + 2]
        if i==1:
            plt.ylabel('X axis (mm)')
        if i==2:
            plt.ylabel('Y axis (mm)')
        if i==3:
            plt.ylabel('Z axis (mm)')
        plt.plot( data, 'b-', label='Reference')
        plt.plot( y, 'g-', linewidth=2, label='Predicted')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()

    # Print accuracy

    print('Index Finger: RMS Pos (mm)')
    posNormVector = np.linalg.norm(testY[:test_size, 0:3] - predicted[:test_size, 0:3], axis=1)
    print(np.sqrt(np.mean((posNormVector) ** 2)))
    # print(compute3DRMS(testY[:test_size, 0:3],predicted[:test_size, 0:3]))

    print('Index Finger: Max Pos (mm)')
    print(np.max(posNormVector))

    for i in range(3):
        print('Index Finger: RMS Pos (mm) - ', i)
        print(compute1DRMS(testY[:test_size, i], predicted[:test_size, i]))

        print('Index Finger: Max Pos (mm) - ', i)
        print(np.max(np.abs(testY[:test_size, i] - predicted[:test_size, i])))

    print('Middle Finger: RMS Pos (mm)')
    posNormVector = np.linalg.norm(testY[:test_size, 3:6] - predicted[:test_size, 3:6], axis=1)
    print(np.sqrt(np.mean((posNormVector) ** 2)))
    # print(compute3DRMS(testY[:test_size, 3:6], predicted[:test_size, 3:6]))

    print('Middle Finger: Max Pos (mm)')
    print(np.max(posNormVector))

    for i in range(3):
        print('Middle Finger: RMS Pos (mm) - ', i)
        print(compute1DRMS(testY[:test_size, i+3], predicted[:test_size, i+3]))

        print('Middle Finger: Max Pos (mm) - ', i)
        print(np.max(np.abs(testY[:test_size, i+3] - predicted[:test_size, i+3])))




plt.show()
