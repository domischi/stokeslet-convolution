import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate
from tqdm import tqdm
import numba 

@numba.jit
def stokeslet(x, mu=1.):
    l=norm(x)
    if l<1e-3:
        return np.identity(2)*np.nan
    return (np.identity(2)/l+np.outer(x,x)/l**3)/(8*np.pi*mu)

@numba.jit
def compute_velocity_field_at_point(f, Xu, int_grid_x, int_grid_y):
    """
    Computes the velocity field at point x according by convoluting the force-field f (given as a function mapping R^2 to R^2) with the Stokeslet formulation, by evaluating a sum over a quadratic grid. This assumes a grid with uniform spacing to be passed
    """
    xu, yu = Xu[0], Xu[1]
    hx=int_grid_x[0,1]-int_grid_x[0,0]
    hy=int_grid_y[1,0]-int_grid_y[0,0]
    ux = 0
    uy = 0
    for si in range(len(int_grid_x)):
        for sj in range(len(int_grid_x[0])):
            x=int_grid_x[si,sj]
            y=int_grid_y[si,sj]
            stkslt=stokeslet((xu-x, yu-y))
            f_at_xy=f((x,y))
            if norm(f_at_xy)>0:
                U=stkslt.dot(f_at_xy)
                ux+=U[0]
                uy+=U[1]
    ux*=hx*hy
    uy*=hx*hy
    return ux, uy

def compute_full_velocity_field(f, xmin=-3, xmax=3, ymin=-3,ymax=3, xres=20, yres=20): 
    Y, X = np.mgrid[xmin:xmax:xres*1j, ymin:ymax:yres*1j]
    Ux=np.zeros_like(X)
    Uy=np.zeros_like(X)
    for iu in tqdm(range(len(Ux))):
        for ju in range(len(Ux[0])):
            xu=X[iu,ju]
            yu=Y[iu,ju]
            Ux[iu,ju], Uy[iu,ju]=compute_velocity_field_at_point(f,(xu,yu), X,Y)
    return X,Y,Ux,Uy


def compute_velocity_field_w_dblquad(x,y, f): 
    """
    Computes the velocity field at point x according by convoluting the force-field f (given as a function mapping R^2 to R^2) with the Stokeslet formulation numerically, using the function dblquad. This raises a few issues, by propagating singular Stokeslets which makes it a bit annoying
    """
    def ux(yp,xp):
        return np.dot(stokeslet(np.array([x-xp,y-yp])), f((xp,yp)))[0] if norm(f((xp,yp)))>0 else 0 
    def uy(yp,xp):
        return np.dot(stokeslet(np.array([x-xp,y-yp])), f((xp,yp)))[1] if norm(f((xp,yp)))>0 else 0 
    ux, _=sp.integrate.dblquad(ux , -np.inf,np.inf,-np.inf,np.inf)
    uy, _=sp.integrate.dblquad(uy , -np.inf,np.inf,-np.inf,np.inf)
    return ux,uy

def make_streamplot_with_dblquad(f, xmin=-3, xmax=3, ymin=-3,ymax=3, xres=20, yres=20, plot_shape=True):
    Y, X = np.mgrid[xmin:xmax:xres*1j, ymin:ymax:yres*1j]
    Ux=np.zeros_like(X)
    Uy=np.zeros_like(X)
    ind=np.zeros_like(X)
    for i in tqdm(numba.prange(len(X))):
        for j in tqdm(range(len(X[0]))):
            if max(abs(f((X[i,j],Y[i,j]))))>0: # Inside the geometry, mask away
                Ux[i,j]=np.nan
                Uy[i,j]=np.nan
                ind[i,j]=1
            else:
                ux,uy=compute_velocity_field(X[i,j],Y[i,j], f)
                Ux[i,j]=ux
                Uy[i,j]=uy
    if plot_shape:
        plt.pcolormesh(X,Y,ind, cmap='Greys', vmin=0, vmax=2)
    plt.streamplot(X, Y, Ux, Uy)
def make_streamplot(f, xmin=-3, xmax=3, ymin=-3,ymax=3, xres=20, yres=20, plot_shape=True):
    X,Y,Ux,Uy=compute_full_velocity_field(f, xmin, xmax, ymin, ymax, xres, yres)
    plt.streamplot(X,Y,Ux,Uy)
