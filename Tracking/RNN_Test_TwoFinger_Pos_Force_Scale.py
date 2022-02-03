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
        y = np.zeros(8)
        y[0:3] = time_series[i + seq_length, 1:4]
        y[3] = time_series[i + seq_length, 8]
        y[4:7] = time_series[i + seq_length, 9:12]
        y[7] = time_series[i + seq_length, 16]
        dataX.append(x)
        dataY.append(y)

    return np.array(dataX), np.array(dataY)


# Parameters
seq_length = 10
EndIdx = -100

# Load data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
dataSet = pd.read_csv('../Data/PalpationTwoFinger_Train 10-02-2021 11-16.csv').to_numpy()
print(dataSet.shape)

time = dataSet[seq_length:EndIdx, 0]
posAndOri = dataSet[seq_length:EndIdx, 1:9]
posAndOri2 = dataSet[seq_length:EndIdx, 9:17]
pressure = dataSet[seq_length:EndIdx, 17:]

print("pressure data shape: ", pressure.shape)
print("pos and ori shape: ", posAndOri.shape)

testX, testY = build_dataset(dataSet[:EndIdx], seq_length)
print("Shape of testX: ", testX.shape)
print("Shape of testY: ", testY.shape)


# Load model from H5 file
new_model = keras.models.load_model('Palpation_pos_force_RNN_twofinger_palpation_25042021.h5')
new_model.summary()


# predict
predicted = new_model.predict(testX)


# scale up
dataSet2 = pd.read_csv('../Data/PalpationTwoFinger_Test 10-02-2021 11-22.csv').to_numpy()
trainX, trainY = build_dataset(dataSet2, seq_length)
scaler = MinMaxScaler()
scaler.fit(trainY)
predicted = scaler.inverse_transform(predicted)

# build augmented data for Unity
augmented_predicted = np.zeros((predicted.shape[0],8))
augmented_predicted[:, 0:3] = predicted[:, 0:3]
augmented_predicted[:, 7] = predicted[:, 3]

augmented_predicted2 = np.zeros((predicted.shape[0],8))
augmented_predicted2[:, 0:3] = predicted[:, 4:7]
augmented_predicted2[:, 7] = predicted[:, 7]

# check
print(predicted)

# save

np.savetxt('predictData.txt', augmented_predicted)
np.savetxt('predictData2.txt', augmented_predicted2)

# draw
predictedPos = augmented_predicted[:, 0:3]
predictedOri = augmented_predicted[:, 3:7]
predictedForce = augmented_predicted[:, 7]

predictedPos2 = augmented_predicted2[:, 0:3]
predictedOri2 = augmented_predicted2[:, 3:7]
predictedForce2 = augmented_predicted2[:, 7]

plt.figure(1)
plt.subplot(221)
plt.plot(time, posAndOri[:,0],label='Test')
plt.plot(time, predictedPos[:,0],label='Predicted')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('X (cm)')
plt.legend(loc='upper right')

plt.subplot(222)
plt.plot(time, posAndOri[:, 1],time,predictedPos[:, 1])
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Y (cm)')

plt.subplot(223)
plt.plot(time, posAndOri[:, 2],time,predictedPos[:, 2])
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Z (cm)')

plt.subplot(224)
plt.plot(time, posAndOri[:, 7],time,predictedForce)
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')

plt.figure(2)
plt.subplot(221)
plt.plot(time, posAndOri2[:,0],label='Test')
plt.plot(time, predictedPos2[:,0],label='Predicted')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('X (cm)')
plt.legend(loc='upper right')

plt.subplot(222)
plt.plot(time, posAndOri2[:, 1],time,predictedPos2[:, 1])
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Y (cm)')

plt.subplot(223)
plt.plot(time, posAndOri2[:, 2],time,predictedPos2[:, 2])
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Z (cm)')

plt.subplot(224)
plt.plot(time, posAndOri2[:, 7],time,predictedForce2)
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')



print('RMS Pos')
posNormVector = np.linalg.norm(posAndOri[:, 0:3]-predictedPos, axis=1)
print(np.sqrt(np.mean((posNormVector)**2))*10)

print('RMS Orientation')
print(np.sqrt(np.mean((posAndOri[:, 3:7]-predictedOri)**2)))

print('RMS Force')
print(np.sqrt(np.mean((posAndOri[:, 7]-predictedForce)**2)))

print('Max Pos')
print(np.max(posNormVector)*10)

print('Max Orientation')
print(np.max(np.abs(posAndOri[:, 3:7]-predictedOri)))

print('Max Force')
print(np.max(np.abs(posAndOri[:, 7]-predictedForce)))

plt.show()