import pandas as pd
from matplotlib import pyplot as plt


# time, pos(3), ori(4), force, pos(3), ori(4), force, pressure(2258)
name = 'Palpation_twoFinger 17-05-2021 13-58'
path = '../../Data/'
dataSet = pd.read_csv(path + name + '.csv').to_numpy()
print(dataSet.shape)
t = dataSet[:, 0]


plt.figure(1)
for i in range(1,4,1):
    plt.subplot(2,2,i)
    data = dataSet[:, i]
    plt.plot(t, data, 'b-')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

data=dataSet[:,8]
plt.subplot(2, 2, 4)
plt.plot(t, data, 'b-')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.figure(2)
for i in range(1, 4, 1):
    plt.subplot(2, 2, i)
    data = dataSet[:, 8 + i]

    plt.plot(t, data, 'b-')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

data=dataSet[:,16]
plt.subplot(2, 2, 4)
plt.plot(t, data, 'b-')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()




# draw a graph
plt.subplots_adjust(hspace=0.35)
plt.show()
