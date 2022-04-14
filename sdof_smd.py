import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Spring mass damper parameters
m = 10
k = 20
c = -5

# Set up statespace matrices
A = np.array([[0, 1],
              [-k/m, -c/m]])

B = np.array([[0], 
              [1/m]])

C = np.array([1, 0])

D = 0

# Set up state-space
sys1 = signal.StateSpace(A,B,C,D)
t1, y1 = signal.step(sys1)

plt.plot(t1, y1)
plt.show()