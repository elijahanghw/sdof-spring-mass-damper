import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.linalg import eig

# Spring mass parameters
M = 10
K = 10
v = 1

# Simulation parameters
x_0 = np.array([[0], 
                [0]])
tstart = 0
tstop = 10
dt = 0.01
t = np.arange(tstart, tstop+1, dt)
U_k = v*t              # Prescribed motion

# Set up state-space matrices
A = np.array([[0, 1],
             [-K/M, 0]])


B = np.array([[0],
              [K/M]])

C = np.array([[1, 0],
              [0, 0]])

D = np.array([[0],
             [1]])

# Continuous to Discrete (RK4 numerical integration)
I = np.eye(2)
A_2 = np.matmul(A, A)
A_3 = np.matmul(A, A_2)
A_4 = np.matmul(A, A_3)
A_k = I + dt*A + dt**2/2*A_2 + dt**3/6*A_3 + dt**4/24*A_4
B_k = np.matmul((dt*I + dt**2/2*A + dt**3/6*A_2 + dt**4/24*A_3), B)
C_k = C
D_k = D

# Time stepping solution
x_old = x_0
y = np.zeros((len(t),2))
for k in range(len(t)):
    y[k,0] = (np.matmul(C_k, x_old) + D_k*U_k[k])[0]
    y[k,1] = (np.matmul(C_k, x_old) + D_k*U_k[k])[1]
    x_new = np.matmul(A_k, x_old) + B_k*U_k[k]
    x_old = x_new

# Plot results
plt.plot(t, y[:,0])
plt.plot(t, y[:,1])
plt.xlabel("t")
plt.ylabel("y")
plt.legend(["Mass", "Anchor"])
plt.show()