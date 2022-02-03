import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
from functools import partial
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# train Parameters
data_dim = 2288
output_dim = 1
learning_rate = 0.01
nbEpochs = 50
iterations = 2
forceMin = 0.8

load_trained_model = True
save_model_name = 'Classifier_CustomCost_3layers_0.8_22042021'
load_model_name = save_model_name
# train_file_name = 'Palpation_one_finger_even_Train 14-04-2021 13-10'
train_file_name = 'Palpation_one_finger_Train 14-04-2021 13-14'
train_file_name2 = 'Palpation_one_finger_even_Train 14-04-2021 13-10'
test_file_name = 'Palpation_one_finger_Test 14-04-2021 13-17'

use_custom_cost = False
weight = 1
def custom_loss(y_actual,y_pred):
    custom_loss = -tf.math.reduce_mean(y_actual * tf.math.log(y_pred) + (1 - y_actual) * tf.math.log(1 - y_pred) * weight)
    return custom_loss


# 'Classifier_Sigmoid_3layers'
# 'Classifier_Softmax_3layers'

# Load train data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)
dataSet = pd.read_csv('../Data/' + train_file_name +'.csv').to_numpy()
print(dataSet.shape)
dataSet2 = pd.read_csv('../Data/' + train_file_name2 +'.csv').to_numpy()
print(dataSet2.shape)





time = np.array(dataSet[:,0])
force = np.array(dataSet[:,8])
pressure = np.array(dataSet[:,17:])

time = np.concatenate([np.array(dataSet[:,0]), np.array(dataSet2[:,0])], axis=0)
force = np.concatenate([np.array(dataSet[:,8]), np.array(dataSet2[:,8])], axis=0)
pressure = np.concatenate([np.array(dataSet[:,17:]), np.array(dataSet2[:,17:])], axis=0)


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
    if use_custom_cost:
        tf.model = keras.models.load_model(load_model_name + '.h5', custom_objects={'custom_loss': custom_loss})
    else:
        tf.model = keras.models.load_model(load_model_name + '.h5')
    tf.model.summary()

else:
    # modify here
    # design your network model
    tf.model = tf.keras.Sequential()
    tf.model.add(tf.keras.layers.Dense(units=500, input_dim=2288, activation = 'relu'))
    tf.model.add(tf.keras.layers.Dense(units=50, activation = 'sigmoid'))
    tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='sigmoid'))
    if use_custom_cost:
        tf.model.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    else:
        tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
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
tpos = testSet[:,1:4]
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

# Load model from H5 file for prediction
new_model = keras.models.load_model(save_model_name + '.h5', custom_objects={'custom_loss': custom_loss})
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
plt.plot(force)
plt.plot(train_y,label='Test', marker='o', markersize=5, linestyle='')
plt.plot(predicted,label='Predicted', marker='o', markersize=5, linestyle='')
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
true_count = 0
false_count = 0
correct_true = 0
correct_false = 0
for i in range(0, count, 1):
    if predicted[i] == test_y[i]:
        correct+=1

    if test_y[i] == 1:
        true_count += 1
        if predicted[i] == 1:
            correct_true += 1
    if test_y[i] == 0:
        false_count += 1
        if predicted[i] == 0:
            correct_false += 1

print(correct_true)
print('Accuracy (%): ', correct/count*100)
print('True ratio (%): ', true_count/count*100)
print('True accuracy (%): ', correct_true/true_count*100)
print('False accuracy (%): ', correct_false/false_count*100)



# plot test result
plt.figure(2)

# plt.subplot(221)
# plt.plot(ttime, tpos[:,0])
# plt.grid(True)
# plt.xlabel('Time (s)')
# plt.ylabel('X (mm)')

# plt.subplot(222)
# plt.plot(ttime, tpos[:,1])
# plt.grid(True)
# plt.xlabel('Time (s)')
# plt.ylabel('Y (mm)')

# plt.subplot(223)
# plt.plot(ttime, tpos[:,2])
# plt.grid(True)
# plt.xlabel('Time (s)')
# plt.ylabel('Z (mm)')


# plt.subplot(224)
plt.plot(ttime, tforce)
plt.plot(ttime, test_y,label='Test', marker='o', markersize=5, linestyle='')
plt.plot(ttime, predicted,label='Predicted', marker='o', markersize=5, linestyle='')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Touched or not')
plt.legend(loc='upper right')
plt.show()




