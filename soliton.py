import numpy as np

def one_soliton(t,x,alpha,s,sigma=1):
    """ solution of KdV eq u_tt + sigma*uu_x + u_xx = 0
    input:
      t = time (scalar or array)
      x = space in one dim (scalar or array)
      alpha = (3*alpha^2/sigma is peak height)
      s = peak position at t=0
    return:
      u = wave amplitude at (t,x); shape of u is
       (n,) if t is scalar and shape of x is (n,)
       (m,n) if shapes of t,x are (m,) and (n,)
    """
    t = np.expand_dims(t,-1)
    f = np.exp(-alpha*(x-s) + alpha**3*t)
    return 12/sigma*alpha**2*f/(1+f)**2

def two_solitons(t,x,alpha,s,sigma=1):
    """ solution of KdV eq u_tt + sigma*uu_x + u_xx = 0
    input and return are the same as one_soliton except:
      alpha = [a_1,a_2] (3*a_i^2/sigma are peak heights)
      s = [s_1,s_2] (peak positions at t=0)
    reference:
      G.B.Whitham "Linear and Nonlinear Waves" eq(17.21)
    """
    t = np.expand_dims(t,-1)
    a1,a2 = alpha
    s1,s2 = s
    b1,b2 = a1**2,a2**2
    c1 = (a2-a1)**2
    c2 = c1/(a2+a1)**2
    f1 = np.exp(a1*(-(x-s1) + b1*t))
    f2 = np.exp(a2*(-(x-s2) + b2*t))
    f3 = f1*f2
    u = b1*f1 + b2*f2 + f3*(2*c1 + c2*(b2*f1 + b1*f2))
    u /= (1 + f1 + f2 + c2*f3)**2
    return 12/sigma*u
