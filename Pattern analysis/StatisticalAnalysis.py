import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import PattenAnalysisTool as pat
import matplotlib.patches as patches

# this is for statiscal analysis of time and force

forceAnalysis = True
timeAnalysis = False
colorbar = False
saveFlag = False

# Load data
# time[0], pos(3)[1~3], ori(4)[4~7], force[8], pos(3)[9-11], ori(4)[12-15], force[16], pressure(2258)[17~]
# loadname = 'ID014_filtered_MAF_pos_ori3'
loadNameList =["002", "003", "005", "006", "007", "008", "009", "010",
               "011", "014", "015", "016", "018", "019"]


loadname = 'ID005_filtered_MAF_pos3'

dataSet = pd.read_csv('../../Data/study/' + loadname + '.csv').to_numpy()
print("data length: ")
print(len(dataSet[:, 1]))
dataSet = dataSet[0:-1, :]

print(len(dataSet[:, 1]))

z_data1 = dataSet[:, 3]
x_data1 = dataSet[:, 1]
f_data1 = dataSet[:, 8]

z_data2 = dataSet[:, 11]
x_data2 = dataSet[:, 9]
f_data2 = dataSet[:, 16]


imageSelection = 2

if imageSelection ==1:

    img=plt.imread("full_breast_v1.jpg")
    z_range = [-185, 160] # x
    x_range = [-170, 210] # y
elif imageSelection==2:

    img=plt.imread("Breast top view trans.png")
    z_range = [-190, 175] # x
    x_range = [-150, 170] # y



width = z_range[1] - z_range[0]
height = x_range[1] - x_range[0]


nbX = 20
nbY = 15
dW = width/nbX
dH = height/nbY
print(dW, dH)




print("Max force: ")
print(max(f_data1))

# Force analysis
if forceAnalysis:
    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize=(15, 5))
    fig.suptitle('Maximum force')

    ax[0].imshow(img, extent=[z_range[0], z_range[1], x_range[0], x_range[1]])
    ax[1].imshow(img, extent=[z_range[0], z_range[1], x_range[0], x_range[1]])

    ax[0].set_title("Left index finger")
    ax[0].set_xlabel('Z Position (mm)')
    ax[0].set_ylabel('X Position (mm)')

    ax[1].set_title("Right index finger")
    ax[1].set_xlabel('Z Position (mm)')
    ax[1].set_ylabel('X Position (mm)')

    for i in range(nbX):
        for j in range(nbY):

            X = z_range[0] + i * dW
            Y = x_range[0] + j * dH
            _x_range = [X, X + dW]
            _y_range = [Y, Y + dH]
            _edge = [0, 0, 0]

            avg_F1 = pat.maxForce(z_data1, x_data1, f_data1, _x_range, _y_range)
            if avg_F1<0:
                avg_F1 = 0
            ratio = avg_F1/4

            _face = [int(255*ratio), 0, 0]
            if _face[0]>255:
                _face[0]=255
            pat.drawFilledSquare(ax[0], X, Y, dW*0.9, dH*0.9, _face, _edge, 0.5)

            avg_F2 = pat.maxForce(z_data2, x_data2, f_data2, _x_range, _y_range)
            if avg_F2 < 0:
                avg_F2 = 0

            ratio = avg_F2 / 4
            _face = [int(255 * ratio), 0, 0]
            if _face[0]>255:
                _face[0]=255
            pat.drawFilledSquare(ax[1], X, Y, dW * 0.9, dH * 0.9, _face, _edge, 0.5)

    if saveFlag:
        plt.savefig('Figures/' + loadname + "_max.png")

    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    fig2.suptitle('Average force')

    ax2[0].imshow(img, extent=[z_range[0], z_range[1], x_range[0], x_range[1]])
    ax2[1].imshow(img, extent=[z_range[0], z_range[1], x_range[0], x_range[1]])

    ax2[0].set_title("Left index finger")
    ax2[0].set_xlabel('Z Position (mm)')
    ax2[0].set_ylabel('X Position (mm)')

    ax2[1].set_title("Right index finger")
    ax2[1].set_xlabel('Z Position (mm)')
    ax2[1].set_ylabel('X Position (mm)')

    for i in range(nbX):
        for j in range(nbY):
            X = z_range[0] + i * dW
            Y = x_range[0] + j * dH
            _x_range = [X, X + dW]
            _y_range = [Y, Y + dH]
            _edge = [0, 0, 0]

            avg_F1 = pat.averageForce(z_data1, x_data1, f_data1, _x_range, _y_range)
            if avg_F1 < 0:
                avg_F1 = 0
            ratio = avg_F1 / 4

            _face = [int(255 * ratio), 0, 0]
            if _face[0] > 255:
                _face[0] = 255
            pat.drawFilledSquare(ax2[0], X, Y, dW * 0.9, dH * 0.9, _face, _edge, 0.5)

            avg_F2 = pat.averageForce(z_data2, x_data2, f_data2, _x_range, _y_range)
            if avg_F2 < 0:
                avg_F2 = 0

            ratio = avg_F2 / 4
            _face = [int(255 * ratio), 0, 0]
            if _face[0] > 255:
                _face[0] = 255
            pat.drawFilledSquare(ax2[1], X, Y, dW * 0.9, dH * 0.9, _face, _edge, 0.5)
    if saveFlag:
        plt.savefig('Figures/' + loadname + "_avg.png")


# Time analysis
if timeAnalysis:
    timePerUnit = 0.085 # second
    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize=(15, 5))
    fig.suptitle('Time spent')

    ax[0].imshow(img, extent=[z_range[0], z_range[1], x_range[0], x_range[1]])
    ax[1].imshow(img, extent=[z_range[0], z_range[1], x_range[0], x_range[1]])

    ax[0].set_title("Left index finger")
    ax[0].set_xlabel('Z Position (mm)')
    ax[0].set_ylabel('X Position (mm)')

    ax[1].set_title("Right index finger")
    ax[1].set_xlabel('Z Position (mm)')
    ax[1].set_ylabel('X Position (mm)')
    max_time = -10000000
    max_data = 120
    for i in range(nbX):
        for j in range(nbY):

            X = z_range[0] + i * dW
            Y = x_range[0] + j * dH
            _x_range = [X, X + dW]
            _y_range = [Y, Y + dH]
            _edge = [0, 0, 0]

            list = pat.listWithInRange(z_data1, x_data1,_x_range, _y_range)
            time = len(list) * timePerUnit
            ratio = len(list) / max_data

            if len(list)>max_time:
                max_time=len(list)


            _face = [int(255*ratio), 0, 0]
            if _face[0]>255:
                _face[0]=255
            pat.drawFilledSquare(ax[0], X, Y, dW*0.9, dH*0.9, _face, _edge, 0.5)

            list = pat.listWithInRange(z_data2, x_data2, _x_range, _y_range)
            time = len(list) * timePerUnit
            ratio = len(list) / max_data

            if len(list) > max_time:
                max_time = len(list)

            _face = [int(255 * ratio), 0, 0]
            if _face[0] > 255:
                _face[0] = 255
            pat.drawFilledSquare(ax[1], X, Y, dW * 0.9, dH * 0.9, _face, _edge, 0.5)

    print(max_time)

    if saveFlag:
        plt.savefig('Figures/' + loadname + "_time.png")


if colorbar:
    fig3, ax3 = plt.subplots(figsize=(3, 5))
    max_range = 40
    max_value = 10

    plt.xlim([0,1])
    plt.ylim([0,max_value])
    # plt.ylabel("Force (N)")
    plt.ylabel("Time (s)")

    for i in range(max_range):

        print((i+1)/max_range)
        _face = [int(255 * (i+1)/max_range), 0, 0]
        _edge = [0, 0, 0]
        pat.drawFilledSquare(ax3, 0, (i)/max_range * max_value, 1, max_value/max_range, _face, _edge, 0.5)

if False:
    for i in range(nbX):
        for j in range(nbY):
            X= z_range[0] + i*dW
            Y= x_range[0] + j*dH


            _x_range = [X, X+dW]
            _y_range = [Y, Y+dH]

            avg_F = pat.maxForce(z_data1, x_data1, f_data1, _x_range, _y_range)
            print([X, Y, avg_F])
            # plt.scatter(X+0.5*dW, Y+0.5*dH, s=200, c=avg_F, alpha=1, cmap='Reds')


plt.show()

