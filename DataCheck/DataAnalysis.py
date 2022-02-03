# data analysis

import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Build data set
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        x = time_series[i:i + seq_length, 17:]
        y = np.zeros(6)
        y[0:3] = time_series[i + seq_length, 1:4]

        y[3:6] = time_series[i + seq_length, 9:12]

        dataX.append(x)
        dataY.append(y)

    return np.array(dataX), np.array(dataY)

# train Parameters
load_trained_model = False
# 데이터에 따라 단위가 다름 2021년 4월 이후 데이터는 mm 이전은 cm
train_file_name = '../../Data/PalpationTwoFinger_Train 10-02-2021 11-16.csv'
test_file_name = '../../Data/PalpationTwoFinger_Test 10-02-2021 11-22.csv'
# model_name = 'Palpation_pos_RNN_twofinger_palpation_25042021_2.h5'
model_name = 'Palpation_pos_RNN_twofinger_palpation_28042021.h5'
seq_length =10
data_dim = 2288
output_dim = 6
learning_rate = 0.01
iterations = 0
EPOCH = 100
PATIENCE = 20
training = True
draw_TrainResult = False
draw_TestResult = True
axis=1



# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
dataSet = pd.read_csv(train_file_name).to_numpy()
testSet = pd.read_csv(test_file_name).to_numpy()
time = dataSet[:, 0]
ttime = testSet[:, 0]

pos1=dataSet[:,1:4]
pos2=dataSet[:,9:12]

tpos1=testSet[:,1:4]
tpos2=testSet[:,9:12]

# data
trainX, trainY = build_dataset(dataSet, seq_length)
print("Train X shape: ", trainX.shape,", Train Y shape: ", trainY.shape)

testX, testY = build_dataset(testSet, seq_length)
print("Test X shape: ", testX.shape,", Test Y shape: ", testY.shape)

plt.figure(1)
plt.plot(trainY[:,axis])
plt.plot(testY[:,axis])
plt.xlabel('Steps')
plt.ylabel('Position (mm)')

# scale data
scaler = MinMaxScaler()
scaler.fit(trainY)
trainY = scaler.transform(trainY)
testY = scaler.transform(testY)

plt.figure(2)
plt.plot(trainY[:,axis])
plt.plot(testY[:,axis])
plt.xlabel('Steps')
plt.ylabel('Normalised Position')
plt.show()