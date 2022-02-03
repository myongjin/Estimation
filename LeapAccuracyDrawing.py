import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9~11], ori(4)[12~15], force[16],pos(3)[17~19]
# index 24-03-2021 14-56
# middle 24-03-2021 14-25
# ring 24-03-2021 14-33
# pinky 24-03-2021 14-45

name = 'index 24-03-2021 14-56'
path = '../Data/'
dataSet = pd.read_csv(path + name + '.csv').to_numpy()
print(dataSet.shape)
xRangeMin = 0
xRangeMax = 3000

t = dataSet[xRangeMin:xRangeMax, 0]

plt.figure(1)
for i in range(1,4,1):
    plt.subplot(1,3,i)
    data = dataSet[xRangeMin:xRangeMax, i]
    data2 = dataSet[xRangeMin:xRangeMax, i + 16]
    plt.plot(t, data, label='TrakStar')
    plt.plot(t, data2, label='LeapMotion')
    plt.xlabel('Time [sec]')
    if i==1:
        plt.ylabel('X [mm]')
        print('RMS X')

    elif i==2:
        plt.ylabel('Y [mm]')
        print('RMS Y')
    else:
        plt.ylabel('Z [mm]')
        print('RMS Z')

    print(np.sqrt(np.mean((data - data2) ** 2)))
    plt.grid()
    plt.xlim(0,20)
    plt.legend()



# draw a graph
plt.subplots_adjust(hspace=0.35)
plt.show()
