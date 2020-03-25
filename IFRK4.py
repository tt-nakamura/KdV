import numpy as np

def IFRK4(A, F, t_span, y0, n, n_step):
    """ solve diff eq dy/dt = A*y + F(t,y)
      by Integrating Factor and Runge Kutta 4th order
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
      L.N.Trefethen "Spectral Methods in MATLAB" p112
    """
    t0,t1 = t_span
    h = (t1-t0)/n
    t = np.linspace(t0,t1,n,endpoint=False)
    E = np.exp(h*A/2)
    E2 = E**2

    y,Y = y0,[y0]
    for i,t in enumerate(t):
        a = h*F(t,y)
        b = h*F(t+h/2, E*(y + a/2))
        c = h*F(t+h/2, E*y + b/2)
        d = h*F(t+h, E2*y + E*c)
        y = E2*y + (E2*a + 2*E*(b+c) + d)/6
        if (i+1)%n_step==0: Y.append(y) 

    return np.array(Y)
