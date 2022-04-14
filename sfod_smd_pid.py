import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Spring mass damper parameters
m = 1       # Mass
k = 20      # Stiffness
c = 10       # Damping

# Simulation parameters
x_0 = np.array([[0], 
                [0]])
tstart = 0
tstop = 1.2
dt = 0.01
t = np.arange(tstart, tstop+1, dt)
ref = 1             # Reference point

# Set up state-space matrices
A = np.array([[0, 1],
              [-k/m, -c/m]])
B = np.array([[0], 
              [1/m]])
C = np.array([1, 0])
D = 0

# Continuous to Discrete (RK4 numerical integration)
I = np.eye(2)
A_2 = np.matmul(A, A)
A_3 = np.matmul(A, A_2)
A_4 = np.matmul(A, A_3)
A_k = I + dt*A + dt**2/2*A_2 + dt**3/6*A_3 + dt**4/24*A_4
B_k = np.matmul((dt*I + dt**2/2*A + dt**3/6*A_2 + dt**4/24*A_3), B)
C_k = C
D_k = D

# PID Gains
K_P = 300
K_I = 500
K_D = 5

# Time stepping solution
x_old = x_0
y = np.zeros(len(t))
error = []
err_old = 0
U_k = 0
for k in range(len(t)):
    y[k] = np.matmul(C_k, x_old) + D_k*U_k
    err_new = ref - y[k]
    error.append(err_new)
    d_err = (err_new - err_old)/dt
    I_err = sum(error)*dt
    U_k = K_P*err_new + K_D*d_err + K_I*I_err
    x_new = np.matmul(A_k, x_old) + B_k*U_k
    x_old = x_new
    err_old = err_new

# Plot results
plt.plot(t, y)
plt.plot(t, ref*np.ones_like(t), color='r', linestyle="--")
plt.xlabel("t")
plt.ylabel("y")
plt.legend(["Response", "Reference"])
plt.show()