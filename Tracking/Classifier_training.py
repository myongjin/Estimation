import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime
import os
from matplotlib import pyplot as plt

# train Parameters
data_dim = 2288
output_dim = 1
learning_rate = 0.01
nbEpochs = 100
iterations = 0
forceMin = 0.1
load_trained_model = True
model_name = 'Classifier_15022021'



# Load train data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)
dataSet = pd.read_csv('../Data/PalpationOneFinger_Train 10-02-2021 11-09.csv').to_numpy()
print(dataSet.shape)

force=np.array(dataSet[:,8])
pressure = dataSet[:, 17:]

print("pressure data shape:", pressure.shape)

# generate classification
touchFlag = []
touchedForce = []
for f in force:
    if f < forceMin:
        touchFlag.append(0)
    else:
        touchFlag.append(1)
        touchedForce.append(f)
train_x = pressure
train_y = np.array(touchFlag, dtype=np.float32)

#plt.plot(touchFlag)
#plt.plot(force)
#plt.plot(touchedForce)

if load_trained_model:
    tf.model = keras.models.load_model(model_name + '.h5')
    tf.model.summary()
else:
    tf.model = tf.keras.Sequential()
    tf.model.add(tf.keras.layers.Dense(units=500, input_dim=2288, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=100, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=50, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='softmax'))
    tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.1), metrics=['accuracy'])
    tf.model.summary()


    # tf.model.load_weights('./checkpoints/Classifier_15022021')


for idx in range(0, iterations, 1):
    print("iteration: ", idx)
    history = tf.model.fit(train_x, train_y, epochs= nbEpochs)
    tf.model.save_weights('./checkpoints/' + model_name)
    tf.model.save(model_name + '.h5')



# Load test data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)
testSet = pd.read_csv('../Data/PalpationOneFinger_Test 10-02-2021 11-14.csv').to_numpy()
ttime = testSet[:, 0]
tforce=np.array(testSet[:,8])
tpressure = testSet[:, 17:]

print(ttime.shape)
print(tpressure.shape)


# generate classification
touchFlag = []
touchedForce = []
for f in tforce:
    if f < forceMin:
        touchFlag.append(0)
    else:
        touchFlag.append(1)
        touchedForce.append(f)

test_y = np.array(touchFlag, dtype=np.float32)

# Load model from H5 file
new_model = keras.models.load_model(model_name + '.h5')
new_model.summary()

# predict
predicted = new_model.predict(tpressure)

# plot
plt.plot(ttime, tforce)
plt.plot(ttime, test_y,label='Test', marker='o', markersize=5, linestyle='')
plt.plot(ttime, predicted,label='Predicted', marker='o', markersize=5, linestyle='')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Touched or not')
plt.legend(loc='upper right')
plt.show()
