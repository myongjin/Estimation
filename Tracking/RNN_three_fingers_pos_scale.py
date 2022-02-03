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
        x = time_series[i:i + seq_length, 25:]
        y = np.zeros(9)
        y[0:3] = time_series[i + seq_length, 1:4]
        y[3:6] = time_series[i + seq_length, 9:12]
        y[6:9] = time_series[i + seq_length, 17:20]
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


# train Parameters

# 데이터에 따라 단위가 다름 2021년 4월 이후 데이터는 mm 이전은 cm
train_file_name = '../Data/Palpation_threeFinger 19-05-2021 14-05.csv'
test_file_name = '../Data/Palpation_threeFinger 19-05-2021 14-07.csv'
# model_name = 'Palpation_pos_RNN_twofinger_palpation_25042021_2.h5'
model_name = 'Palpation_pos_RNN_ThreeFinger_palpation_20052021.h5'

test_length = 200
seq_length = 10
data_dim = 2288
output_dim = 9
learning_rate = 0.01
iterations = 5
EPOCH = 100
PATIENCE = 20


load_trained_model = True
training = True
draw_TrainResult = False
draw_TestResult = True



# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
dataSet = pd.read_csv(train_file_name).to_numpy()
testSet = pd.read_csv(test_file_name).to_numpy()
time = dataSet[seq_length-1:-1, 0]
ttime = testSet[seq_length-1:-1, 0]

# data
trainX, trainY = build_dataset(dataSet, seq_length)
testX, testY = build_dataset(testSet, seq_length)

# trainX,trainY,testX,testY = build_datasetV2(dataSet,seq_length,0.7)

print("Train X shape: ", trainX.shape,", Train Y shape: ", trainY.shape)
print("Test X shape: ", testX.shape,", Test Y shape: ", testY.shape)


# scale data
scaler = MinMaxScaler()
scaler.fit(trainY)
trainY = scaler.transform(trainY)
scaled_testY = scaler.transform(testY)
# print(trainY)


if load_trained_model:
    tf.model = keras.models.load_model(model_name)
else:
    tf.model = tf.keras.Sequential()
    tf.model.add(tf.keras.layers.LSTM(units=500, input_shape=(seq_length, data_dim)))
    tf.model.add(tf.keras.layers.Dense(units=100, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=50, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='linear'))
    tf.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                     metrics=['accuracy'])
tf.model.summary()

# Set check point
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name,             # file명을 지정합니다
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
                           validation_data=[testX, scaled_testY],
                           callbacks=[checkpoint, earlystopping])


tf.model = keras.models.load_model(model_name)

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

    plt.figure(3)
    plt.suptitle('Training: Pinky finger')
    for i in range(1, 4, 1):
        plt.subplot(3, 1, i)
        data = trainY[:, i + 5]
        y = predicted[:, i + 5]
        if i == 1:
            plt.ylabel('X axis (mm)')
        if i == 2:
            plt.ylabel('Y axis (mm)')
        if i == 3:
            plt.ylabel('Z axis (mm)')
        plt.plot(data, 'b-', label='Reference')
        plt.plot(y, 'g-', linewidth=2, label='Predicted')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()


if draw_TestResult:
    predicted = tf.model.predict(testX)
    predicted = scaler.inverse_transform(predicted)
    predicted = predicted[:test_length,:]
    testY = testY[:test_length,:]

    plt.figure(4)
    plt.suptitle('Test: Index finger')
    for i in range(1, 4, 1):
        plt.subplot(3, 1, i)
        data = testY[:, i - 1]
        y = predicted[:, i - 1]
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


    plt.figure(5)
    plt.suptitle('Test: Middle finger')
    for i in range(1, 4, 1):
        plt.subplot(3, 1, i)
        data = testY[:, i + 2]
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

    plt.figure(6)
    plt.suptitle('Test: Pinky finger')
    for i in range(1, 4, 1):
        plt.subplot(3, 1, i)
        data = testY[:, i + 5]
        y = predicted[:, i + 5]
        if i == 1:
            plt.ylabel('X axis (mm)')
        if i == 2:
            plt.ylabel('Y axis (mm)')
        if i == 3:
            plt.ylabel('Z axis (mm)')
        plt.plot(data, 'b-', label='Reference')
        plt.plot(y, 'g-', linewidth=2, label='Predicted')
        plt.xlabel('Time (s)')
        plt.grid()
        plt.legend()

    # Print accuracy

    print('Index Finger: RMS Pos (mm)')
    posNormVector = np.linalg.norm(testY[:, 0:3] - predicted[:, 0:3], axis=1)
    print(np.sqrt(np.mean((posNormVector) ** 2)))

    print('Index Finger: Max Pos (mm)')
    print(np.max(posNormVector))


    print('Middle Finger: RMS Pos (mm)')
    posNormVector = np.linalg.norm(testY[:, 3:6] - predicted[:, 3:6], axis=1)
    print(np.sqrt(np.mean((posNormVector) ** 2)))

    print('Middle Finger: Max Pos (mm)')
    print(np.max(posNormVector))

    print('Pinky Finger: RMS Pos (mm)')
    posNormVector = np.linalg.norm(testY[:, 6:9] - predicted[:, 6:9], axis=1)
    print(np.sqrt(np.mean((posNormVector) ** 2)))

    print('Pinky Finger: Max Pos (mm)')
    print(np.max(posNormVector))


plt.show()
