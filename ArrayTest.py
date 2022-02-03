import numpy as np


a = np.array([[1,2,3],[4,5,6]])
b = np.array([[4,5,6],[7,8,9]])
print(a.shape)
print(b.shape)
print(a[0, :]+b[0, :])

print(np.linalg.norm(a,axis= 1))