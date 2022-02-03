import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
import pandas as pd
import math
from matplotlib import pyplot as plt
import random

# 힘 분포를 그리기 위한 그래프

# Load data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
dataSet = pd.read_csv('../../Data/Palpation_one_finger_Train 14-04-2021 13-14.csv').to_numpy()
print(dataSet.shape)

dataSet = dataSet[100:-100]

# 데이터 생성 우선 y축 데이터로 테스트해보자
time = dataSet[:,0]

forcedata = dataSet[:,8]
xdata = dataSet[:,1]
ydata = dataSet[:,2]
zdata = dataSet[:,3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xdata, zdata, forcedata, marker='o', s=15, cmap='Greens')

plt.show()






