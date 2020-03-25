import numpy as np

def ETDRK4(A, F, t_span, y0, n, n_step):
    """ solve diff eq dy/dt = A*y + F(t,y)
      by Exponential Time Difference and Runge Kutta 4th order
    input:
      A = coeff of linear term A*y (same shape as y)
      F = Non-linear function F(t,y)
      t_span = [t0,t1], integrate from t=t0 to t=t1
      y0 = initial value of y at t=t0
      n = number of integration steps in t_span
      n_step = number of steps between two outputs
    return:
      y in shape (m+1, len(y0)) where m = [n/n_step]
       y[i,j] = j th component of y at t=t_i (i=0...m)
       where t_i = t0 + i*n_step*(t1-t0)/n
    reference:
      Kassam and Trefethen, SIAM J Sci Comput 26 (2005) 1214
    """
    t0,t1 = t_span
    h = (t1-t0)/n
    t = np.linspace(t0,t1,n,endpoint=False)
    E = np.exp(h*A/2)
    E2 = E**2

    M = 64
    r = np.exp(np.pi*1j*(2*np.arange(M) + 1)/M)
    z = h*np.expand_dims(A,-1) + r
    Q  = h*np.mean((np.exp(z/2)-1)/z, axis=-1)
    f1 = h*np.mean((-4-z+np.exp(z)*(4-3*z+z**2))/z**3, axis=-1)
    f2 = h*np.mean(2*(2+z+np.exp(z)*(-2+z))/z**3, axis=-1)
    f3 = h*np.mean((-4-3*z-z**2+np.exp(z)*(4-z))/z**3, axis=-1)
    if np.all(np.isreal(A)):
        Q,f1,f2,f3 = np.real([Q,f1,f2,f3])

    y,Y = y0,[y0]
    for i,t in enumerate(t):
        a = F(t,y);  y1 = E*y + Q*a
        b = F(t+h/2, y1)
        c = F(t+h/2, E*y + Q*b)
        d = F(t+h, E*y1 + Q*(2*c-a))
        y = E2*y + a*f1 + (b+c)*f2 + d*f3
        if (i+1)%n_step==0: Y.append(y) 

    return np.array(Y)
