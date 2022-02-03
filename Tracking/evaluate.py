# Lab 4 Multi-variable linear regression
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.tools import freeze_graph



# evaluate data set
xy = np.loadtxt('../Data/evaluation.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 6:]
y_data = xy[:, 0:6]

# Make sure the shape and data are OK
print(x_data, "\nx_data shape:", x_data.shape)
print(y_data, "\ny_data shape:", y_data.shape)

# Load model from H5 file
new_model = keras.models.load_model('my_model_4layers.h5')
new_model.summary()
new_model.save('my_model.pb')

# Evaluation
results = new_model.evaluate(x_data,  y_data)
print("test loss, test acc:", results)