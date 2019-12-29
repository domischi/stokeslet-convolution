import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate
from tqdm import tqdm
import numba
from scipy.signal import convolve

@numba.jit
def stokeslet(x, mu=1.):
    l=norm(x)
    if l<1e-3:
        l=1e-3 
    return (np.identity(2)/l+np.outer(x,x)/l**3)/(8*np.pi*mu)

@numba.jit
def compute_velocity_field_at_point(f, Xu, int_grid_x, int_grid_y):
    """
    Computes the velocity field at point x according by convoluting the force-field f (given as a function mapping R^2 to R^2) with the Stokeslet formulation, by evaluating a sum over a quadratic grid. This assumes a grid with uniform spacing to be passed
    """
    xu, yu = Xu[0], Xu[1]
    if norm(f((xu,yu)))>0:
        return np.nan, np.nan
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

def get_f_on_grid(f, xmin=-3, xmax=3, ymin=-3,ymax=3, xres=20, yres=20):
    X=np.linspace(xmin,xmax,xres)
    Y=np.linspace(ymin,ymax,yres)
    X, Y = np.meshgrid(X,Y)
    gf = []
    ## Sadly, I do need the append mechanism rather than a indexing. It seems to confuse entries of multidimensional arrays
    for i in range(len(X)):
        tmp=[]
        for j in range(len(X[0])):
            x=X[i,j]
            y=Y[i,j]
            tmp.append(f((x,y)))
        gf.append(tmp)
    return X, Y, np.array(gf)

def compute_full_velocity_field_conv(f, xmin=-3, xmax=3, ymin=-3,ymax=3, xres=20, yres=20):
    X, Y, ff = get_f_on_grid(f, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xres=xres, yres=yres)
    hx_f=X[0,1]-X[0,0]
    hy_f=Y[1,0]-Y[0,0]
    ff_X = ff[:,:,0]
    ff_Y = ff[:,:,1]
    s_X, s_Y, s = get_f_on_grid(stokeslet, xmin=2*xmin, xmax=2*xmax, ymin=2*ymin, ymax=2*ymax, xres=2*xres-1, yres=2*yres-1)
    s = np.array(s)
    hx_s=s_X[0,1]-s_X[0,0]
    hy_s=s_Y[1,0]-s_Y[0,0]
    stokeslet_XX = s[:,:,0,0]
    stokeslet_XY = s[:,:,0,1]
    stokeslet_YY = s[:,:,1,1]

    assert(np.isclose(hx_s, hx_f))
    assert(np.isclose(hy_s, hy_f))

    ind = np.zeros_like(ff_X)
    for i in range(len(ff_X)):
        for j in range(len(ff_X[0])):
            if ff_X[i,j]  != 0 or ff_Y[i,j] != 0:
                ind[i,j] = 1
    Ux = convolve(stokeslet_XX, ff_X, mode='valid') + convolve(stokeslet_XY, ff_Y, mode='valid')
    Uy = convolve(stokeslet_XY, ff_X, mode='valid') + convolve(stokeslet_YY, ff_Y, mode='valid')
    Ux *= hx_f*hy_f
    Uy *= hx_f*hy_f
    ## The velocity field does not make sense where there is a finite force field
    for i in range(len(ind)):
        for j in range(len(ind[0])):
            if ind[i,j]:
                Ux[i,j]=np.nan
                Uy[i,j]=np.nan
    return X,Y,Ux,Uy

def compute_full_velocity_field(f, xmin=-3, xmax=3, ymin=-3,ymax=3, xres=20, yres=20): 
    X=np.linspace(xmin,xmax,xres)
    Y=np.linspace(ymin,ymax,yres)
    X, Y = np.meshgrid(X,Y)
    Ux=np.zeros_like(X)
    Uy=np.zeros_like(X)
    for iu in range(len(Ux)):
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
    X=np.linspace(xmin,xmax,xres)
    Y=np.linspace(ymin,ymax,yres)
    X, Y = np.meshgrid(X,Y)
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
        plt.pcolormesh(X,Y,ind, cmap='Greys', alpha=.5, edgecolor='none')
    plt.streamplot(X, Y, Ux, Uy)

def get_domain(f,X,Y):
    ind=np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            if norm(f((X[i,j],Y[i,j])))>0:
                ind[i,j]= 1 
    return ind

def get_inflow_matrix(X,Y,Ux,Uy, normalize=True):
    """
    Given the velocity field (Ux, Uy) at positions (X,Y), compute for every x in (X,Y) if the position is contributing to the inflow or the outflow. This can be obtained by u*x (scalar product). If this quantity is positive, then it's an outflow, if it is negative it's an inflow.
    """
    io=np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            io[i,j]= Ux[i,j]*X[i,j] + Uy[i,j]*Y[i,j]
    if normalize:
        io_max=np.nanmax(abs(io))
        for i in range(len(X)):
            for j in range(len(X[0])):
                io[i,j]/=io_max
    return io

def make_streamplot(f, xmin=-3, xmax=3, ymin=-3,ymax=3, xres=20, yres=20, plot_shape=True, plot_io_pattern=False, quiver_plot=False, use_convolution_method=True): # todo implement the remove fore field shape as a flag
    """
    Do all calculations and the figure plotting for a forcefield f using the Stokesian formulation of 2D hydrodynamics.

    Parameters
    ----------
    f : function
        Force-field. Expects a tuple (x,y) and maps it to a force (f1,f2) at this point.

    xmin, xmax, ymin, ymax: scalar, optional
        Set the bounding box for the calculation. If use_convolution_method is set, then the bounding box should be symmetric around (0,0), otherwise undefined behavior occurs.

    xres, yres: int, optional
        Resolution of the grid for f. Also the resolution for the grid used for the Stokeslet if use_convolution_method is false, otherwise the second grid uses approximately twice the number of points.

    plot_shape: bool, optional
        If set, the method tries to find where f attains a finite value and indicates this by a gray background.

    plot_io_pattern: bool, optional
        If set, in the background is the inflow or outflow indicated by a density map. Here, np.dot(x,v) is used to determine if the flow is towards the center or outwards of it.

    quiver_plot: bool, optional
        If set, the method used quiver rather than streamplot. Gives another way of interpreting the data.

    use_convolution_method: bool, optional
        If set, make_streamplot relies on compute_full_velocity_field_conv rather than compute_full_velocity_field to do the Stokeslet computation. This is orders of magnitudes faster, but relies on a symmetric grid and is less intuitively implemented.

    Returns
    -------
    fig : matplotlib.pyplot.figure object
        The figure that is generated

    X, Y: ndarray, 2-dimensional
        Grid for the velocities
    Ux, Uy: ndarray, 2-dimensional
        The velocity compnents along x and y respectively, evaluated on the grid positions given by X, Y
    """
    if use_convolution_method:
        X,Y,Ux,Uy=compute_full_velocity_field_conv(f, xmin, xmax, ymin, ymax, xres, yres)
    else:
        X,Y,Ux,Uy=compute_full_velocity_field(f, xmin, xmax, ymin, ymax, xres, yres)
    if abs(Ux).sum()+abs(Uy).sum()<=0:
        print(f"Warning: There seems to be an issue with this grid. I'm not plotting anything.\n(xmin={ xmin } ,xmax={ xmax } ,ymin={ ymin } ,ymax={ ymax } ,xres={ xres } ,yres={ yres })")
    else:
        if plot_io_pattern:
            fig=plt.figure(figsize=(11,9))
        else:
            fig=plt.figure(figsize=(9,9))
        if plot_shape:
            dX=(X[0,1]-X[0,0])/2
            dY=(Y[1,0]-Y[0,0])/2
            sX = X+dX # shifted X, for plotting purposes
            sY = Y+dY # shifted Y, for plotting purposes
            ind=get_domain(f, sX,sY)
            plt.pcolormesh(X,Y,ind, cmap='Greys', alpha=.5, edgecolor='none')
        if plot_io_pattern:
            io = get_inflow_matrix(X,Y,Ux,Uy)
            im = plt.pcolormesh(X,Y,io, cmap='bwr', alpha=.6, vmin=-1, vmax=1)
            fig.colorbar(im)
        if quiver_plot:
            plt.quiver(X,Y,Ux,Uy)
        else:
            plt.streamplot(X,Y,Ux,Uy)
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        return fig, X,Y, Ux,Uy
