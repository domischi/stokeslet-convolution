import numpy as np
from numpy.linalg import norm
import numba

def rect_forcefield_all_right(x, a=1,b=1):
    if -a<x[0] and x[0]<a and -b<x[1] and x[1]<b:
        return np.array([1.,0])
    else:
        return np.zeros(2)

@numba.jit
def rect_forcefield_to_center(x, a=1,b=1):
    x=np.array(x)
    if -a<=x[0] and x[0]<=a and -b<=x[1] and x[1]<=b:
        return -x/norm(x)
    else:
        return np.zeros(2)

@numba.jit(nopython=True)
def rect_annulus_forcefield_to_center(x, a=1,b=1, ap=0.9, bp=0.9):
    x=np.array(x)
    if (-a<=x[0] and x[0]<=-ap) or (ap<=x[0] and x[0]<=a) and (-b<=x[1] and x[1]<=-bp) or (bp<=x[1] and x[1]<=b):
        return -x/norm(x)
    else:
        return np.zeros(2)
