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
        if norm(x)>1e-10:
            return -x/norm(x)
    return np.zeros(2)

@numba.jit(nopython=True)
def rect_annulus_forcefield_to_center(x, a=1,b=1, ap=0.9, bp=0.9):
    x=np.array(x)
    if ((-a<=x[0] and x[0]<=-ap) or (ap<=x[0] and x[0]<=a)) and ((-b<=x[1] and x[1]<=-bp) or (bp<=x[1] and x[1]<=b)):
        if norm(x)>1e-10:
            return -x/norm(x)
    return np.zeros(2)

@numba.jit
def diagonal_forcefield_to_center(x, a=1,b=1, w=.3):
    x=np.array(x)
    ## is the point x within boundaries of width w of the lines going through (-a,-b) -- (a,b) or (-a,b) -- (a,-b)
    ## The equations describing the two line segments is ax+by=0 and ax-by=0 and 
    ## The shortest distance of a point (x0,y0) to a line described as such is d=abs(ax0+by0/sqrt(a^2+b^2)
    if (-a<=x[0] and x[0]<=a) and (-b<=x[1] and x[1]<=b): # Only inside rectangle
        d1=abs(a*x[0]+b*x[1])/np.sqrt(a**2+b**2)
        d2=abs(a*x[0]-b*x[1])/np.sqrt(a**2+b**2)
        if min([d1,d2])<w:
            if norm(x)>1e-10:
                return -x/norm(x)
    return np.zeros(2)
