import numpy as np
from scipy.integrate import odeint
import time 

#===================================== FUNCTIONS =================================================#

# ODE System
def f(y, t):
    dydt = [
        -1.71*y[0] + 0.43*y[1] + 8.32*y[2] + 0.0007,
        1.71*y[0] - 8.75*y[1],
        -10.03*y[2] + 0.43*y[3] + 0.035*y[4],
        8.32*y[1] + 1.71*y[2] - 1.12*y[3],
        -1.745*y[4] + 0.43*y[5] + 0.43*y[6],
        -280*y[5]*y[7] + 0.69*y[3] + 1.71*y[4] - 0.43*y[5] + 0.69*y[6],
        280*y[5]*y[7] - 1.81*y[6],
        -280*y[5]*y[7] + 1.81*y[6]
    ]
    return dydt

# MESCD
def mescd(yapprox, yfinal):
    value =  -np.log10(np.max(np.abs(yapprox[-1]-yfinal)) / (atol/rtol)+np.max(np.abs(yfinal)))
    return value

# SCD
def scd(yapprox, yfinal):
    if np.max(np.abs(yfinal)) > 1e-12:
        value = -np.log10(np.max(np.abs(yapprox[-1]-yfinal)) / np.max(np.abs(yfinal)))
        return value
    elif np.max(np.abs(yfinal)):
        value = -np.log10(np.max(np.abs(yapprox[-1]-yfinal)))
        return value
    
#=================================================================================================#


# Parameters
y0 = [1, 0, 0, 0, 0, 0, 0, 0.0057] 
t0 = 0
tmax = 321.8122 
yfinal = [0.7371312573325668e-3, 
          0.1442485726316185e-3,
          0.5888729740967575e-4,
          0.1175651343283149e-2,
          0.2386356198831331e-2,
          0.6238968252742796e-2,
          0.2849998395185769e-2,
          0.2850001604814231e-2]

h = 1#e-7
atol = 1e-7
rtol = 1e-4  
dt = int((tmax - t0) / h) + 1
t = np.linspace(t0, tmax, dt)  
cpu_times = np.zeros(5)
runs = [0, 1, 2, 3, 4]

# Begin solving
for n in runs:
    tic = time.clock()
    yapprox, stats = odeint(f, y0, t, rtol=rtol, atol=atol, full_output=True)
    toc = time.clock()
    cpu_times[n] = (toc-tic)

cputime = np.median(cpu_times)


# Print statistics
print("|    Solver    |    mescd    |    scd    |    steps    |    feval    |    nJac    |    CPU    |")
print("|    odeint    |  {:.7f} |  {:.7f}   |      {}     |      {}     |     {}     |  {:.7f}   |".format(
    round(mescd(yapprox, yfinal),7), round(scd(yapprox, yfinal),7), stats['nst'][-1],
    stats['nfe'][-1], stats['nje'][-1], round(cputime,7)))
