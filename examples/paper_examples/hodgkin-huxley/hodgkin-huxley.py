import numpy as np
import matplotlib.pyplot as plt

T = 50 # ms
dt = 0.001
N = int(np.ceil(T/dt))
t = np.linspace(0,T,N+1)
dt = T/N

gna = 120
gk = 36
gl = 0.3

vna = 56
vk = -77
vl = -60

Cm = 1
I = 0.005*gna*vna

def alphan(v):
    y = np.exp(1-v/10)
    return 0.1*np.log(y)/(y-1)
def alpham(v):
    y = np.exp(2.5-v/10)
    return np.log(y)/(y-1)
alphah = lambda v : 0.07*np.exp(-v/20)
#alphan = lambda v : 0.01*(10-v)/(np.exp(1-v/10)-1)
#alpham = lambda v : 0.1*(25-v)/(np.exp(2.5-v/10)-1)
betan  = lambda v : np.exp(-v/80)/8
betam  = lambda v : 4*np.exp(-v/18)
betah  = lambda v : 1/(1 + np.exp(3-v/10))
ninf   = lambda v : alphan(v)/(alphan(v) + betan(v))
minf   = lambda v : alpham(v)/(alpham(v) + betam(v))
hinf   = lambda v : alphah(v)/(alphah(v) + betah(v))

veq = (vk*gk*ninf(0)**4 + gna*vna*minf(0)**3*hinf(0) + gl*vl)/(gk*ninf(0)**4 + gna*minf(0)**3*hinf(0) + gl)

# H-H variables
V = np.zeros((N+1,)); V[0] = veq
n = np.zeros((N+1,)); n[0] = ninf(V[0]-veq)
m = np.zeros((N+1,)); m[0] = minf(V[0]-veq)
h = np.zeros((N+1,)); h[0] = hinf(V[0]-veq)

# F-N variables
Vfn = np.zeros((N+1,)); Vfn[0] = veq
nfn = np.zeros((N+1,)); nfn[0] = ninf(Vfn[0]-veq)
hbar = ninf(Vfn[0]-veq) + hinf(Vfn[0]-veq)

for i in range(N):
    # Hodgkin-Huxley
    Dn = alphan(V[i]-veq)*(1-n[i]) - betan(V[i]-veq)*n[i]
    Dm = alpham(V[i]-veq)*(1-m[i]) - betam(V[i]-veq)*m[i]
    Dh = alphah(V[i]-veq)*(1-h[i]) - betah(V[i]-veq)*h[i]
    n[i+1] = n[i] + dt*Dn
    m[i+1] = m[i] + dt*Dm
    h[i+1] = h[i] + dt*Dh

    Ina = gna*m[i]**3*h[i]*(V[i] - vna)
    Ik  = gk*n[i]**4*(V[i] - vk)
    Il  = gl*(V[i] - vl)

    DV = (1/Cm)*(I - (Ina + Ik + Il))
    V[i+1] = V[i] + dt*DV

    # Fitzhugh-Nagumo
    Dnfn = alphan(Vfn[i]-veq)*(1-nfn[i]) - betan(Vfn[i]-veq)*nfn[i]
    nfn[i+1] = nfn[i] + dt*Dnfn

    Ina_fn = gna*minf(Vfn[i]-veq)**3*(hbar - nfn[i])*(Vfn[i] - vna)
    Ik_fn  = gk*nfn[i]**4*(Vfn[i] - vk)
    Il_fn  = gl*(Vfn[i] - vl)

    DVfn = (1/3/Cm)*(I - (Ina_fn + Ik_fn + Il_fn))
    Vfn[i+1] = Vfn[i] + dt*DVfn

plt.plot(t, V-veq, t, Vfn-veq)
plt.show()
