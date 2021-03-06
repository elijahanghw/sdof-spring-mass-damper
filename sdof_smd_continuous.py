import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Spring mass damper parameters
m = 1      # Mass
k = 20      # Stiffness
c = 5      # Damping
u = 10      # Input (force)

# Simulation parameters
tstart = 0
tstop = 10
dt = 0.01
t = np.arange(tstart, tstop+1, dt)
Ft = u*np.ones_like(t)              # Custom step input

# Set up state-space matrices
A = np.array([[0, 1],
              [-k/m, -c/m]])
B = np.array([[0],
              [1/m]])
C = np.array([1, 0])
D = 0

# Set up state-space
sys1 = signal.StateSpace(A,B,C,D)

# Response to input
t1, y1, x1 = signal.lsim(sys1, Ft, t)

# Plot response
plt.plot(t1, y1)
plt.xlabel("t")
plt.ylabel("y")
plt.show()