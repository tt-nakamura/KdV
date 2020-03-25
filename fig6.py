import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft,rfftfreq
from ETDRK4 import ETDRK4
from IFRK4 import IFRK4
from soliton import two_solitons
from time import process_time

N = 512
L = 2*np.pi
dx = L/N
x = np.arange(N)*dx
alpha = [25,16]
s = [1,2]
u = two_solitons(0,x,alpha,s)
v = rfft(u)
p = 2*np.pi*rfftfreq(N,dx)
A = 1j*p**3

def F(t,y): return -0.5j*p*rfft(irfft(y)**2)

T = 1e-3
u = two_solitons(T,x,alpha,s)
u0 = np.max(u)

n_step = np.logspace(7,14,8,base=2)
e1,e2,t1,t2 = [],[],[],[]

for n in n_step:
    ta = process_time()
    y1 = IFRK4(A, F, [0,T], v, n, n)
    tb = process_time()
    y2 = ETDRK4(A, F, [0,T], v, n, n)
    tc = process_time()
    e1.append(np.max(np.abs(irfft(y1[-1]) - u))/u0)
    e2.append(np.max(np.abs(irfft(y2[-1]) - u))/u0)
    t1.append(tb-ta)
    t2.append(tc-tb)

h = 1/n_step

plt.subplots_adjust(left=0.12, right=0.98,
                    bottom=0.12, top=0.96, wspace=0)

plt.subplot(1,2,1)
plt.loglog(h, e1, '*-', label='IFRK4')
plt.loglog(h, e2, '+--', label='ETDRK4')
plt.ylabel('relative error at $t = 0.001$', fontsize=14)
plt.xlabel('relative time-step', fontsize=14)
plt.xticks([1e-4,1e-3])
plt.legend()

plt.subplot(1,2,2)
plt.loglog(t1, e1, '*-', label='IFRK4')
plt.loglog(t2, e2, '+--', label='ETDRK4')
plt.yticks([])
plt.xlabel('computer time  / sec', fontsize=14)
plt.legend()

plt.savefig('fig6.eps')
plt.show()
