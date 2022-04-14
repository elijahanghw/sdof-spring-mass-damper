import numpy as np

# Spring mass damper parameters
m = 10
k = 20
c = 5

# Set up statespace matrices
A = np.array([[0, 1],
              [-k/m, -c/m]])

B = np.array([0, 1/m])

C = np.array([1, 0])

D = np.array([0, 0])