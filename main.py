# Lab 4 Multi-variable linear regression
import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime
import os



xy = np.loadtxt('../Data/TestData 23-11-2020 14-18.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 6:]
y_data = xy[:, 0:6]

# Make sure the shape and data are OK
print(x_data, "\nx_data shape:", x_data.shape)
print(y_data, "\ny_data shape:", y_data.shape)

# generate model
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=400, input_dim=529, activation='linear'))
tf.model.add(tf.keras.layers.Dense(units=300, activation='linear'))
tf.model.add(tf.keras.layers.Dense(units=100, activation='linear'))
tf.model.add(tf.keras.layers.Dense(units=50, activation='linear'))
tf.model.add(tf.keras.layers.Dense(units=20, activation='linear'))
tf.model.add(tf.keras.layers.Dense(units=6, activation='linear'))
tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5), metrics=['accuracy'])
tf.model.summary()

# tf.model.load_weights('./checkpoints/my_checkpoint')




# for tensorboard
log_dir = os.path.join(".", "logs", "breast_logs_0.01", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = tf.model.fit(x_data, y_data, epochs=200, callbacks=[tensorboard_callback])

tf.model.save_weights('./checkpoints/my_checkpoint')
tf.model.save('my_model_4layers.h5')