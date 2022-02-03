import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# train Parameters
load_trained_model = False
model_name = 'Palpation_pos_force_RNN_twofinger_palpation_25042021'
seq_length = 10
data_dim = 2288
output_dim = 8
learning_rate = 0.01
iterations = 1
EPOCHS = 50



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



# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
dataSet = pd.read_csv('../Data/PalpationTwoFinger_Train 10-02-2021 11-16_manualFilter.csv').to_numpy()
print(dataSet.shape)

time = dataSet[seq_length-1:-1, 0]


# data
trainX, trainY = build_dataset(dataSet, seq_length)
print("pos and ori data shape:", trainY.shape)
print("pressure data shape:", trainX.shape)

# scale data
scaler = MinMaxScaler()
scaler.fit(trainY)
trainY = scaler.transform(trainY)
# print(trainY)


if load_trained_model:
    tf.model = keras.models.load_model(model_name + '.h5')
    tf.model.summary()
else:
    tf.model = tf.keras.Sequential()
    tf.model.add(tf.keras.layers.LSTM(units=500, input_shape=(seq_length, data_dim)))
    tf.model.add(tf.keras.layers.Dense(units=100, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=50, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='linear'))
    tf.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                     metrics=['accuracy'])
    tf.model.summary()


for idx in range(0, iterations, 1):
    print("iteration: ", idx)
    history = tf.model.fit(trainX, trainY, epochs=EPOCHS)
    tf.model.save(model_name + '.h5')


predicted = tf.model.predict(trainX)

print(time.shape)

plt.figure(1)
threshold = 10
for i in range(1,4,1):
    plt.subplot(3,1,i)
    data = trainY[:, i-1]
    y = predicted[:,i-1]

    plt.plot(time, data, 'b-', label='Test')
    plt.plot(time, y, 'g-', linewidth=2, label='Predicted')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

plt.figure(2)
for i in range(1, 4, 1):
    plt.subplot(3, 1, i)
    data = trainY[:, i + 3]
    y = predicted[:, i + 3]

    plt.plot(time, data, 'b-', label='Test')
    plt.plot(time, y, 'g-', linewidth=2, label='Predicted')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()


# plt.plot(history.history['accuracy'])
plt.show()
