import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# train Parameters
data_dim = 2288
output_dim = 1
learning_rate = 0.01
nbEpochs = 500
iterations = 0
forceMin = 0.8
load_trained_model = True
save_model_name = 'Classifier_Sigmoid_3layers_0.8_15042021'
load_model_name = save_model_name

train_file_name = 'Palpation_one_finger_even_Train 14-04-2021 13-10'
test_file_name = 'Palpation_one_finger_Test 14-04-2021 13-17'



# 'Classifier_Sigmoid_3layers'
# 'Classifier_Softmax_3layers'

# Load train data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)
dataSet = pd.read_csv('../Data/' + train_file_name +'.csv').to_numpy()
print(dataSet.shape)

time = np.array(dataSet[:,0])
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

print('Train X shape: ', train_x.shape)
print('Train Y shape: ', train_y.shape)

#plt.plot(touchFlag)
#plt.plot(force)
#plt.plot(touchedForce)




if load_trained_model:
    tf.model = keras.models.load_model(load_model_name + '.h5')
    tf.model.summary()
else:
    # modify here
    # design your network model
    tf.model = tf.keras.Sequential()
    tf.model.add(tf.keras.layers.Dense(units=500, input_dim=2288, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=100, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=50, activation='relu'))
    tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='sigmoid'))
    tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
    tf.model.summary()


for idx in range(0, iterations, 1):
    print("iteration: ", idx)
    history = tf.model.fit(train_x, train_y, epochs=nbEpochs)
    # tf.model.save_weights('./checkpoints/' + model_name)
    tf.model.save(save_model_name + '.h5')



# Load test data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)
testSet = pd.read_csv('../Data/' + test_file_name +'.csv').to_numpy()
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
new_model = keras.models.load_model(save_model_name + '.h5')
new_model.summary()


# plot train data
# predict
predicted = new_model.predict(train_x)
# convert result
for i in range(0, predicted.shape[0], 1):
    if predicted[i] > 0.5:
        predicted[i] = 1
    else:
        predicted[i] = 0

plt.figure(1)
plt.plot(time, force)
plt.plot(time, train_y,label='Test', marker='o', markersize=5, linestyle='')
plt.plot(time, predicted,label='Predicted', marker='o', markersize=5, linestyle='')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Touched or not')
plt.legend(loc='upper right')



# predict
predicted = new_model.predict(tpressure)

# convert result
for i in range(0, predicted.shape[0], 1):
    if predicted[i] > 0.5:
        predicted[i] = 1
    else:
        predicted[i] = 0

# print accuracy
count = predicted.shape[0]
correct = 0
correct_true = 0
for i in range(0, count, 1):
    if predicted[i] == test_y[i]:
        correct+=1
    if predicted[i]==0:
        if test_y[i]==0:
            correct_true +=1

print(correct_true)
print('Accuracy (%): ', correct/count*100)



# plot test result
plt.figure(2)
plt.plot(ttime, tforce)
plt.plot(ttime, test_y,label='Test', marker='o', markersize=5, linestyle='')
plt.plot(ttime, predicted,label='Predicted', marker='o', markersize=5, linestyle='')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Touched or not')
plt.legend(loc='upper right')
plt.show()




