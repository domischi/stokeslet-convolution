from stokeslet import *
from forcefields import *
from misc import *
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt

SAVEFIG=True

ex = Experiment('Stokeslet Annulus')
if SAVEFIG:
    ex.observers.append(FileStorageObserver.create('data'))
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    ARs=np.linspace(.10,1.00,10)
    res=200
    thickness=[.1]
    L=2

@ex.capture
def one_AR(ar, res, L,t):
    a=1
    b=float(ar)
    fname=f'/tmp/AR_{int(100*float(ar))}_{int(t*10)}.png'
    ff=lambda x: rect_annulus_forcefield_to_center(x,a=a, b=b, ap=a*(1- t ),bp=b*(1-t))
    fig, _,_,_,_ = make_streamplot(ff,xmin=-L, xmax=L, ymin=-L, ymax=L, xres=res, yres=res, plot_io_pattern=True)
    plt.title(f'AR={ar}, a={a:.0f}, b={b:.2f}')
    if SAVEFIG:
        plt.savefig(fname)
        ex.add_artifact(fname)
    else:
        plt.show(block=True)
    if fig!=None:
        plt.close(fig)

@ex.automain
def main(ARs, thickness):
    for ar in ARs:
        for t in thickness:
    #for ar in tqdm(ARs):
    #    for t in tqdm(thickness):
            one_AR(ar,t=t)
