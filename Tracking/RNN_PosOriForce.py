import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# train Parameters
seq_length = 20
data_dim = 2288
output_dim = 8
learning_rate = 0.01
iterations = 10



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



# time, pos(3), ori(4), force, pos(3), ori(4), force, pressure(2258)
dataSet = pd.read_csv('../Data/PalpationOneFinger_Train 04-02-2021 13-31.csv').to_numpy()
print(dataSet.shape)




# data
trainX, trainY = build_dataset(dataSet, seq_length)

# scale data
scaler = MinMaxScaler()
scaler.fit(trainY)
trainY = scaler.transform(trainY)
print(trainY)


print("pos and ori data shape:", trainY.shape)
print("pressure data shape:", trainX.shape)




tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units=1000, input_shape=(seq_length, data_dim)))
tf.model.add(tf.keras.layers.Dense(units=500, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=250, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=125, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=50, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=25, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='linear'))
tf.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
tf.model.summary()

# tf.model.load_weights('./checkpoints/LSTM_Dense')


for idx in range(0, iterations, 1):
    history = tf.model.fit(trainX, trainY, epochs=100)
    tf.model.save_weights('./checkpoints/LSTM_Dense')
    tf.model.save('Palpation_pos_force_RNN_onefinger_palpation.h5')


plt.plot(history.history['accuracy'])
plt.show()
