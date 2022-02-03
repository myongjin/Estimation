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
seq_length = 10
EndIdx = -1
test_length = 200

# Load data
# time, pos(3), ori(3), force, pos(3), ori(3), force, pressure(2258)
dataSet = pd.read_csv('../Data/PalpationOneFinger_Moved 10-02-2021 11-26.csv').to_numpy()
print(dataSet.shape)

time = dataSet[seq_length:EndIdx, 0]
posAndOri = dataSet[seq_length:EndIdx, 1:9]
pressure = dataSet[seq_length:50, 17:]

print("pressure data shape: ", pressure.shape)
print("pos and ori shape: ", posAndOri.shape)

testX, testY = build_dataset(dataSet[:EndIdx], seq_length)
print("Shape of testX: ", testX.shape)
print("Shape of testY: ", testY.shape)


# Load model from H5 file
new_model = keras.models.load_model('Palpation_pos_force_RNN_onefinger_palpation_11022021.h5')
new_model.summary()

# predict
predicted = new_model.predict(testX)


# scale up
dataSet2 = pd.read_csv('../Data/PalpationOneFinger_Train 10-02-2021 11-09.csv').to_numpy()
trainX, trainY = build_dataset(dataSet2, seq_length)
scaler = MinMaxScaler()
scaler.fit(trainY)
predicted = scaler.inverse_transform(predicted)

#adjust length
time = time[:test_length]
predicted = predicted[:test_length, :]
posAndOri = posAndOri[:test_length, :]

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
plt.plot(time, posAndOri[:,0] * 10,label='Test')
plt.plot(time, predictedPos[:,0] * 10,label='Predicted')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('X (mm)')
plt.legend(loc='upper right')

plt.subplot(222)
plt.plot(time, posAndOri[:, 1] * 10,time,predictedPos[:, 1] * 10)
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Y (mm)')

plt.subplot(223)
plt.plot(time, posAndOri[:, 2] * 10,time,predictedPos[:, 2] * 10)
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Z (mm)')

plt.subplot(224)
plt.plot(time, posAndOri[:, 7],time,predictedForce)
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')


# plt.figure(2)
# plt.grid(True)
# plt.plot(posAndOri[:, 1], abs(posAndOri[:, 0]-predictedPos[:, 0]),label='X error (cm)', marker='o', markersize=5, linestyle='')
# plt.plot(posAndOri[:, 1], abs(posAndOri[:, 2]-predictedPos[:, 2]),label='Z error (cm)', marker='o', markersize=5, linestyle='')
# plt.legend(loc='upper left')
# plt.xlabel('Y (cm)')
# plt.ylabel('Absolute Error (cm)')

print('RMS Pos (mm)')
posNormVector = np.linalg.norm(posAndOri[:, 0:3]-predictedPos, axis=1)
print(np.sqrt(np.mean((posNormVector)**2))*10)

print('RMS Force')
print(np.sqrt(np.mean((posAndOri[:, 7]-predictedForce)**2)))

print('Max Pos (mm)')
print(np.max(posNormVector)*10)

print('Max Force (N)')
print(np.max(np.abs(posAndOri[:, 7]-predictedForce)))

plt.show()