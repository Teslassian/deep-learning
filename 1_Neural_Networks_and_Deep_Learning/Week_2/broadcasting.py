import numpy as np

A = np.array([[56, 0, 4.4, 68],
              [1.2, 104, 52, 8],
              [1.8, 135, 99, 0.9]])
print(A)
print('\n')

cal = A.sum(axis=0)
print(cal)
print('\n')

percentage = A/cal.reshape(1,4)*100
print(percentage)
