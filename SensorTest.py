from tensorflow import keras
import numpy as np
import pandas as pd
import datetime
import os
from matplotlib import pyplot as plt

# time, pos(3), ori(3), force, pos(3), ori(3), force, pressure(2258)
dataSet = pd.read_csv('../Data/SensorTesting_200 27-01-2021 10-47.csv').to_numpy()
print(dataSet.shape)

time = dataSet[:,0]
posAndOri = dataSet[:, 1:9]
pressure = dataSet[:, 17:]

print(np.argmax(pressure[100,:]))

for idx in range(0,11):
    plt.plot(time, pressure[:,1055+idx])

plt.show()