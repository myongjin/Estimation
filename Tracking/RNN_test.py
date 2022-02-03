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
        y = time_series[i +
                        seq_length, 1:9]  # Next close price
        # print(x, "->", y)
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)


# train Parameters
seq_length = 20


# time, pos(3), ori(3), force, pos(3), ori(3), force, pressure(2258)
dataSet = pd.read_csv('../Data/PalpationOneFinger_Test 04-02-2021 13-35.csv').to_numpy()
print(dataSet.shape)

time = dataSet[seq_length:,0]
posAndOri = dataSet[seq_length:, 1:9]
pressure = dataSet[seq_length:, 17:]

print("pressure data shape: ", pressure.shape)
print("pos and ori shape: ", posAndOri.shape)

testX, testY = build_dataset(dataSet, seq_length)
print("Shape of testX: ", testX.shape)
print("Shape of testY: ", testY.shape)


# Load model from H5 file
new_model = keras.models.load_model('Palpation_pos_force_RNN_onefinger_palpation.h5')
new_model.summary()
new_model.save('my_model.pb')

# predict
predicted = new_model.predict(testX)

# check
print(predicted)

# save
np.savetxt('predictData.txt', predicted)


# draw
predictedPos = predicted[:, 0:3]
predictedOri = predicted[:, 3:7]
predictedForce = predicted[:, 7]

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
plt.plot(time, posAndOri[:,3],label='Test')
plt.plot(time, predictedOri[:,0],label='Predicted')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('x')
plt.legend(loc='upper right')

plt.subplot(222)
plt.plot(time, posAndOri[:, 4],time,predictedOri[:, 1])
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('y')

plt.subplot(223)
plt.plot(time, posAndOri[:, 5],time,predictedOri[:, 2])
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('z')

plt.subplot(224)
plt.plot(time, posAndOri[:, 6],time,predictedOri[:, 3])
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('w')

plt.figure(3)
plt.grid(True)
plt.plot(posAndOri[:, 1], abs(posAndOri[:, 0]-predictedPos[:, 0]),label='X error (cm)', marker='o', markersize=5, linestyle='')
plt.plot(posAndOri[:, 1], abs(posAndOri[:, 2]-predictedPos[:, 2]),label='Z error (cm)', marker='o', markersize=5, linestyle='')
plt.legend(loc='upper left')
plt.xlabel('Y (cm)')
plt.ylabel('Absolute Error (cm)')

print('RMS Pos')

print(np.sqrt(np.mean((posAndOri[:, 0:3]-predictedPos)**2)))

print('RMS Orientation')
print(np.sqrt(np.mean((posAndOri[:, 3:7]-predictedOri)**2)))


print('RMS Force')
print(np.sqrt(np.mean((posAndOri[:, 7]-predictedForce)**2)))

plt.show()