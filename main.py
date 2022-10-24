
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# constants
g = 9.81           # gravity [m/(s*s)]
C = 0.0025         # drag coeff.
Gam = 0.10         # mixing efficiency coeff.
a = 0.009          # slope [m/m]
N = 0.0242**2      # buoyancy freq. squared
dt = 60            # time step [s]

# observations, load data
dat = pq.read_table('data.parquet')
Bobs = np.array(dat.column(0))
uobs = np.array(dat.column(1))
hobs = np.array(dat.column(2))
uh = np.array(dat.column(3))

# n-steps
tf = len(Bobs)

# ICs
def IC(param_obs):
    global tf
    self = np.zeros(tf)
    self[0] = np.mean(param_obs[0:9])
    return self
B = IC(Bobs)       # buoyancy defect
u = IC(uobs)       # layer-averged velocity
h = IC(hobs)       # mixed layer thickness

# entrainment closure
G1 = 2*Gam*C
G2 = C**(1/2)/(Gam*0.4)
def entrain(B,u,h,uh):
    global dt, a, N, C, G1, G2
    Fs = 1/12 * uh*N*a*h
    Ri = (1/2 * N*h*h - B) / (u*u)
    Lmo = C**(3/2)*u**3 / (0.4*Fs)
    ht = G1/Ri * (1-G2*h/Lmo) * abs(u) *dt
    return ht

# update u
def up_u(B,u,h,uh,n):
    global dt, a, C
    part1 = (C*u[n]**2+B[n+1]*a)
    part2 = h[n]*(u[n]-uh[n])
    u1 = (dt*part1+part2)/h[n+1] + uh[n+1]
    return u1

# solve
for n in range(tf-1):
    B[n+1] = B[n] - N*a*h[n]*(u[n]-uh[n])*dt
    h[n+1] = h[n]
    u[n+1] = up_u(B,u,h,uh,n)

    # entrainment thickening
    h1 = entrain(B[n],u[n],h[n],uh[n])
    h[n+1] = h1 + h[n]
    u[n+1] = up_u(B,u,h,uh,n)

    h2 = entrain(B[n+1],u[n+1],h[n+1],uh[n+1])
    h[n+1] = (h1+h2)/2 + h[n]
    u[n+1] = up_u(B,u,h,uh,n)

# plot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
def plot_results(y1,y2,title,ax):
    axs[ax].plot(y1,'--',linewidth=5,color='gray')
    axs[ax].plot(y2,'k')
    axs[ax].set(xlabel='minutes')
    axs[ax].set_title(title)
    axs[ax].legend(('model','observed'))
plot_results(h,hobs,'h(m)',0)
plot_results(-u,-uobs,'u (m$s^{-1}$)',1)
plot_results(-B,-Bobs,'B ($m^2$$s^{-2}$)',2)
plt.savefig('results.png')
print('results.png saved in current directory')
plt.show()

# RMSE
MSE = np.square(np.subtract(hobs,h)).mean()
RMSE = MSE**(0.5)
print("herr=",RMSE)

MSE = np.square(np.subtract(uobs,u)).mean()
RMSE = MSE**(0.5)
print("uerr=", RMSE)

MSE = np.square(np.subtract(Bobs,B)).mean()
RMSE = MSE**(0.5)
print("Berr=", RMSE)