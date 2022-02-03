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
        y = np.zeros(4)
        y[0:3] = time_series[i + seq_length, 1:4]
        y[3] = time_series[i + seq_length, 8]
        # print(x, "->", y)
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)


# Parameters
classifier_file_name = 'Classifier_EarlyStopping_2layers_0.8_22042021.h5'
seq_length = 10
startIdx = 30
endIdx = -1

left = 20
right = 50
use_filter = False
draw_touch = False
threshold = 0.7
load_model_name = 'Palpation_pos_force_RNN_onefinger_32042021.h5'
test_data_name = '../Data/Palpation_one_finger_Test 14-04-2021 13-17.csv'
train_data_name = '../Data/Palpation_one_finger_Train 14-04-2021 13-14.csv'
# 'Palpation_pos_force_RNN_onefinger_22042021.h5'

# Load data for tracking
# time, pos(3), ori(3), force, pos(3), ori(3), force, pressure(2258)
dataSet = pd.read_csv(test_data_name).to_numpy()
dataSet = dataSet[startIdx:, :]
print(dataSet.shape)

time = dataSet[seq_length:endIdx, 0]
test_Y = dataSet[seq_length:endIdx, 1:9]
pressure = dataSet[seq_length:endIdx, 17:]

print("pressure data shape: ", pressure.shape)
print("pos and ori shape: ", test_Y.shape)

testX, testY = build_dataset(dataSet[:endIdx], seq_length)
print("Shape of testX: ", testX.shape)
print("Shape of testY: ", testY.shape)

# Load classifier
classifier = keras.models.load_model(classifier_file_name)
classifier.summary()

# Load tracking model from H5 file
new_model = keras.models.load_model(load_model_name)
new_model.summary()

# predict tracking
predicted = new_model.predict(testX)

# predict touch or not
touchOrNot = classifier.predict(pressure)

# convert result
filtered_time = []
filtered_prediction = []
filtered_y = []
for i in range(0, touchOrNot.shape[0], 1):
    if touchOrNot[i] > threshold:
        touchOrNot[i] = 1
        filtered_time.append(time[i])
        filtered_prediction.append(predicted[i,:])
        filtered_y.append(test_Y[i,:])
    else:
        touchOrNot[i] = 0


filtered_prediction = np.array(filtered_prediction, dtype=np.float32)
filtered_y = np.array(filtered_y, dtype=np.float32)
filtered_time = np.array(filtered_time, dtype=np.float32)

if use_filter:
    print('Before: ' , predicted.shape)
    predicted = filtered_prediction
    print('After: ' , predicted.shape)
    test_Y=filtered_y
    time=filtered_time

# scale up
dataSet2 = pd.read_csv(train_data_name).to_numpy()
trainX, trainY = build_dataset(dataSet2, seq_length)
scaler = MinMaxScaler()
scaler.fit(trainY)
predicted = scaler.inverse_transform(predicted)

# build augmented data
augmented_predicted = np.zeros((predicted.shape[0],8))
augmented_predicted[:, 0:3] = predicted[:, 0:3]
augmented_predicted[:, 7] = predicted[:, 3]

# check
print(predicted)

# save
np.savetxt('predictData.txt', augmented_predicted)


# draw
predictedPos = augmented_predicted[:, 0:3]
predictedOri = augmented_predicted[:, 3:7]
predictedForce = augmented_predicted[:, 7]

plt.figure(1)
plt.subplot(221)
plt.plot(time,test_Y[:,0],label='Test')
plt.plot(time,predictedPos[:,0],label='Predicted')
if use_filter==False and draw_touch:
    plt.plot(time, touchOrNot * 20)
plt.grid(True)
plt.xlim(left,right)
plt.xlabel('Time (s)')
plt.ylabel('X (mm)')
plt.legend(loc='upper right')

plt.subplot(222)
plt.plot(time,test_Y[:, 1])
plt.plot(time,predictedPos[:, 1])
if use_filter==False and draw_touch:
    plt.plot(time, touchOrNot * 20)
plt.grid(True)
plt.xlim(left,right)
plt.xlabel('Time (s)')
plt.ylabel('Y (mm)')

plt.subplot(223)
plt.plot(time,test_Y[:, 2])
plt.plot(time,predictedPos[:, 2])
if use_filter==False and draw_touch:
    plt.plot(time, touchOrNot * 20)
plt.grid(True)
plt.xlim(left,right)
plt.xlabel('Time (s)')
plt.ylabel('Z (mm)')

plt.subplot(224)
plt.plot(time,test_Y[:, 7])
plt.plot(time,predictedForce)
if use_filter==False and draw_touch:
    plt.plot(time, touchOrNot)
plt.grid(True)
plt.xlim(left,right)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')


# plt.figure(2)
# plt.plot(touchOrNot)

# plt.figure(2)
# plt.grid(True)
# plt.plot(posAndOri[:, 1], abs(posAndOri[:, 0]-predictedPos[:, 0]),label='X error (cm)', marker='o', markersize=5, linestyle='')
# plt.plot(posAndOri[:, 1], abs(posAndOri[:, 2]-predictedPos[:, 2]),label='Z error (cm)', marker='o', markersize=5, linestyle='')
# plt.legend(loc='upper left')
# plt.xlabel('Y (cm)')
# plt.ylabel('Absolute Error (cm)')

print('RMS Pos (mm)')
posNormVector = np.linalg.norm(test_Y[:, 0:3]-predictedPos, axis=1)
print(np.sqrt(np.mean((posNormVector)**2)))

print('RMS Force (N)')
print(np.sqrt(np.mean((test_Y[:, 7]-predictedForce)**2)))

print('Max Pos (mm)')
print(np.max(posNormVector))

print('Max Force (N)')
print(np.max(np.abs(test_Y[:, 7]-predictedForce)))

plt.show()