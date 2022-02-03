import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


file_name1 = 'Train_smallforce 16-09-2021 12-37'
file_name2 = 'Jennifer_pos_force'


# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
load_name = '../../Data/' + file_name1 + '.csv'
dataSet1 = pd.read_csv(load_name).to_numpy()
load_name = '../../Data/' + file_name2 + '.csv'
dataSet2 = pd.read_csv(load_name).to_numpy()

dataSet1 = dataSet1[20:-40]
dataSet2 = dataSet2[20:-40]

d1x1 = dataSet1[:,1]
d1x2 = dataSet1[:,9]
d1x = np.vstack((d1x1, d1x2))

d1y1 = dataSet1[:,2]
d1y2 = dataSet1[:,10]
d1y = np.vstack((d1y1, d1y2))

d1z1 = dataSet1[:,3]
d1z2 = dataSet1[:,11]
d1z = np.vstack((d1z1, d1z2))

d1f1 = dataSet1[:,8]
d1f2 = dataSet1[:,16]
d1f = np.vstack((d1f1, d1f2))

d2x1 = dataSet2[:,1]
d2x2 = dataSet2[:,9]
d2x = np.vstack((d2x1, d2x2))

d2y1 = dataSet2[:,2]
d2y2 = dataSet2[:,10]
d2y = np.vstack((d2y1, d2y2))

d2z1 = dataSet2[:,3]
d2z2 = dataSet2[:,11]
d2z = np.vstack((d2z1, d2z2))

d2f1 = dataSet2[:,8]
d2f2 = dataSet2[:,16]
print(min(d2f1), min(d2f2))

#d2f1 += 2.5
#d2f2 += 2.5
print(min(d2f1), min(d2f2))
d2f = np.vstack((d2f1, d2f2))



# x-y
plt.figure()
plt.subplot(2, 2, 1)
plt.scatter(d1x, d1y, color = 'red', label = 'Training data', s=1)
plt.scatter(d2x, d2y, color = 'blue', label = 'Test data', s=1)
plt.title('X-Y plan')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend()

# x-z
plt.subplot(2, 2, 2)
plt.scatter(d1z, d1x, color = 'red', label = 'Training data', s=1)
plt.scatter(d2z, d2x, color = 'blue', label = 'Test data', s=1)
plt.title('X-Z plan')
plt.xlabel('Z (mm)')
plt.ylabel('X (mm)')
plt.legend()

# y-z
plt.subplot(2, 2, 3)
plt.scatter(d1z, d1y, color = 'red', label = 'Training data', s=1)
plt.scatter(d2z, d2y, color = 'blue', label = 'Test data', s=1)
plt.title('Z-Y plan')
plt.xlabel('Z (mm)')
plt.ylabel('Y (mm)')
plt.legend()

# y-z
plt.subplot(2, 2, 4)
plt.scatter(d1y, d1f, color = 'red', label = 'Training data', s=1)
plt.scatter(d2y, d2f, color = 'blue', label = 'Test data', s=1)
plt.title('Y-F')
plt.xlabel('Y (mm)')
plt.ylabel('F (N)')
plt.legend()

plt.show()

