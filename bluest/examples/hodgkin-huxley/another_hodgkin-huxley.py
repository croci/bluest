import numpy as np
import matplotlib.pyplot as plt

T = 25 # ms
dt = 0.01
N = int(np.ceil(T/dt))
dt = T/float(N)

gna = 1.2
gk = 0.36
gl = 0.003

vna = 55.17
vk = -72.14
vl = -49.42
Cm = 0.01

I = 0.1 #0.005*gna*vna
print(I)

alphan = lambda v : 0.01*(v+50)/(1-np.exp(-5-v/10))
alpham = lambda v : 0.1*(35+v)/(1-np.exp(-3.5-v/10))
alphah = lambda v : 0.07*np.exp(-(v+60)/20)
betan  = lambda v : np.exp(-(v+60)/80)/8
betam  = lambda v : 4*np.exp(-(v+60)/18)
betah  = lambda v : 1/(1 + np.exp(-3-v/10))
ninf   = lambda v : alphan(v)/(alphan(v) + betan(v))
minf   = lambda v : alpham(v)/(alpham(v) + betam(v))
hinf   = lambda v : alphah(v)/(alphah(v) + betah(v))

veq = (vk*gk*ninf(0)**4 + gna*vna*minf(0)**3*hinf(0) + gl*vl)/(gk*ninf(0)**4 + gna*minf(0)**3*hinf(0) + gl)

t = np.linspace(0,T,N+1)
V = np.zeros((N+1,)); V[0] = -60
n = np.zeros((N+1,)); n[0] = ninf(V[0])
m = np.zeros((N+1,)); m[0] = minf(V[0])
h = np.zeros((N+1,)); h[0] = hinf(V[0])

print(veq, n[0], m[0], h[0])

for i in range(N):
    Dn = alphan(V[i])*(1-n[i]) - betan(V[i])*n[i]
    Dm = alpham(V[i])*(1-m[i]) - betam(V[i])*m[i]
    Dh = alphah(V[i])*(1-h[i]) - betah(V[i])*h[i]
    n[i+1] = n[i] + dt*Dn
    m[i+1] = m[i] + dt*Dm
    h[i+1] = h[i] + dt*Dh

    Ina = gna*m[i]**3*h[i]*(V[i] - vna)
    Ik  = gk*n[i]**4*(V[i] - vk)
    Il  = gl*(V[i] - vl)

    DV = (1/Cm)*(I - (Ina + Ik + Il))
    V[i+1] = V[i] + dt*DV

plt.plot(t, V)
print(n.min(), n.max(), m.min(), m.max(), h.min(), h.max())
plt.show()
