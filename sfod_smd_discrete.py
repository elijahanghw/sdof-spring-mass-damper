import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Spring mass damper parameters
m = 1      # Mass
k = 20      # Stiffness
c = 5       # Damping
u = 10      # Input (force)

# Simulation parameters
x_0 = np.array([[0], 
                [0]])
tstart = 0
tstop = 60
dt = 0.01
t = np.arange(tstart, tstop+1, dt)
U_k = u*np.ones_like(t)              # Custom step input

# Set up state-space matrices
A = np.array([[0, 1],
              [-k/m, -c/m]])
B = np.array([[0], 
              [1/m]])
C = np.array([1, 0])
D = 0

# Discrete state-space (Bilinear transform)
I = np.eye(2)
A_k = np.matmul((I + 0.5*dt*A), inv((I - 0.5*dt*A)))
B_k = np.matmul(inv(A), (A_k - I))
B_k = np.matmul(B_k, B) 
C_k = C
D_k = D

# Time stepping solution
x_old = x_0
y = np.zeros(len(t))

for k in range(len(t)):
    y[k] = np.matmul(C_k, x_old) + D_k*U_k[k]
    x_new = np.matmul(A_k, x_old) + B_k*U_k[k]
    x_old = x_new

# Plot results
plt.plot(t, y)
plt.xlabel("t")
plt.ylabel("y")
plt.show()