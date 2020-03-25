import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft,rfftfreq
from ETDRK4 import ETDRK4

N = 256; L = 32*np.pi; dx = L/N
x = np.arange(N)*dx
u = np.cos(x/16)*(1 + np.sin(x/16))
p = 2*np.pi*rfftfreq(N,dx)
A = p**2 - p**4 # Kuramoto-Sivashinsky

def F(t,y): return -0.5j*p*rfft(irfft(y)**2)

T = 150
y = ETDRK4(A, F, [0,T], rfft(u), 1024, 8)
t = np.linspace(0, T, len(y))
plt.contourf(x, t, irfft(y), cmap='seismic')
plt.xlabel('x', fontsize=14)
plt.ylabel('t', fontsize=14)
plt.colorbar()
plt.tight_layout()
plt.savefig('fig5.eps')
plt.show()
