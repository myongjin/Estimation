import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime
import os
from matplotlib import pyplot as plt


# time, pos(3), ori(4), force, pos(3), ori(4), force, pressure(2258)
dataSet = pd.read_csv('../Data/PalpationOneFinger_Train 04-02-2021 13-31.csv').to_numpy()
print(dataSet.shape)


pos = dataSet[:, 1:4]
posAndOri = dataSet[:, 1:9]
pressure = dataSet[:, 17:]

# scale down
# posAndOri[:, 0:2] /= np.max(posAndOri)

print("pos and ori data shape:", posAndOri.shape)
print("pressure data shape:", pressure.shape)

# generate model
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1500, input_dim=2288, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=800, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=400, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=200, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=100, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=50, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=25, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=16, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=8, activation='linear'))
tf.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
tf.model.summary()

# tf.model.load_weights('./checkpoints/my_checkpoint')


# for tensorboard
log_dir = os.path.join(".", "logs", "BreastTissueTumour_logs_0.01", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

for idx in range(0, 3, 1):
    history = tf.model.fit(pressure, posAndOri, epochs=100, callbacks=[tensorboard_callback])
    tf.model.save_weights('./checkpoints/my_checkpoint')
    tf.model.save('model_BreastModel_10layers_8output_onefinger_palpation.h5')



plt.plot(history.history['accuracy'])
plt.show()