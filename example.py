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

ex = Experiment('Stokeslet Rectangular non-normalized')
ex.observers.append(MongoObserver.create())
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    ARs=np.union1d(np.linspace(.10,1.00,10),np.linspace(.90,1.00,11))
    mus=np.logspace(-2,2,5)
    res=200
    L=2

@ex.capture
def one_experiment(ar, res, L, mu):
    a=1
    b=float(ar)
    fname=f'/tmp/AR_{int(100*float(ar))}.png'
    ff=lambda x: rect_forcefield_to_center_non_normalized(x,a=a, b=b)
    fig, _,_,_,_ = make_streamplot(ff,xmin=-L, xmax=L, ymin=-L, ymax=L, xres=res, yres=res, plot_io_pattern=True)
    plt.title(f"{ ex.get_experiment_info()['name'] }, AR={ar:.2f}, a={a:.0f}, b={b:.2f}")
    if SAVEFIG:
        plt.savefig(fname)
        ex.add_artifact(fname)
    else:
        plt.show(block=True)
    if fig!=None:
        plt.close(fig)

@ex.automain
def main(ARs, mus):
    for ar in tqdm(ARs):
        for mu in mus:
            one_AR(ar, mus)
