import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# train Parameters
seq_length = 10
data_dim = 2288
output_dim = 4
learning_rate = 0.01
iterations = 5
load_model_flag = True
save_model_name = 'Palpation_pos_force_RNN_onefinger_32042021.h5'
load_model_name = save_model_name



# Build data set
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        x = time_series[i:i + seq_length, 17:]
        y = np.zeros(4)
        y[0:3] = time_series[i + seq_length, 1:4]
        y[3] = time_series[i + seq_length, 8]  # Next close price
        # print(x, "->", y)
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)



# time[0], pos(3)[1~3], ori(4), force, pos(3), ori(4), force, pressure(2258)
dataSet = pd.read_csv('../Data/Palpation_one_finger_Train 14-04-2021 13-14.csv').to_numpy()
print(dataSet.shape)




# data
trainX, trainY = build_dataset(dataSet, seq_length)
print("pos and ori data shape:", trainY.shape)
print("pressure data shape:", trainX.shape)

# scale data
scaler = MinMaxScaler()
scaler.fit(trainY)
trainY = scaler.transform(trainY)
print(trainY)


if load_model_flag:
    tf.model = keras.models.load_model(load_model_name)
else:
    tf.model = tf.keras.Sequential()
    tf.model.add(tf.keras.layers.LSTM(units=500, input_shape=(seq_length, data_dim)))
    tf.model.add(tf.keras.layers.Dense(units=100, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=50, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='linear'))
    tf.model.compile(loss='mean_squared_error',
                     optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
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
                                 patience= 10)

for idx in range(0, iterations, 1):
    print("iteration: ", idx)
    history = tf.model.fit(trainX,
                           trainY, epochs=10)
    #tf.model.save(save_model_name)


# plt.plot(history.history['accuracy'])
plt.show()
