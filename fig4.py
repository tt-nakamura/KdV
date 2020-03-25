import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft,rfftfreq
from ETDRK4 import ETDRK4

N = 256; L = 2*np.pi; dx = L/N
x = np.arange(N)*dx
a,b = 25,16
u = 3*(a/np.cosh(a/2*(x-1)))**2
u+= 3*(b/np.cosh(b/2*(x-2)))**2

p = 2*np.pi*rfftfreq(N,dx)
A = 1j*p**3 # KdV
def F(t,y): return -0.5j*p*rfft(irfft(y)**2)

T = 6e-3
y = ETDRK4(A, F, [0,T], rfft(u), 1024, 8)
t = np.linspace(0, T, len(y))
plt.contourf(x, t, irfft(y), cmap='jet')
plt.xlabel('x', fontsize=14)
plt.ylabel('t', fontsize=14)
plt.colorbar()
plt.tight_layout()
plt.savefig('fig4.eps')
plt.show()
