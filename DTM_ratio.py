"""

Figure 2 in paper

"""

import numpy as np
import pandas as pd
from functools import partial
import matplotlib, sys, h5py, re, schwimmbad
from uncertainties import unumpy
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")
# plt.style.use('classic')

from helpers import get_files
import flares

def plt_median(x, y, ws, bins, ax, ii):

    df = pd.read_csv('weight_files/weights_grid.txt')
    weights = np.array(df['weights'])
    quantiles = [0.84,0.50,0.16]

    bincen = (bins[1:]+bins[:-1])/2.
    out = flares.binned_weighted_quantile(x, y, ws, bins, quantiles)
    xx, yy, yy84, yy16 = bincen, out[:,1], out[:,0], out[:,2]
    ok = np.isfinite(yy)

    # print (out, xx)


    ax.plot(xx[ok], yy[ok], marker='o', color=s_m.to_rgba(ii+0.5), alpha=0.7)
    ax.fill_between(xx[ok], yy16[ok], yy84[ok], color=s_m.to_rgba(ii+0.5), alpha=0.2)


if __name__ == "__main__":

    h = 0.6777

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    vol = (4/3)*np.pi*(14/h)**3

    quantiles = [0.84,0.50,0.16]

    df = pd.read_csv('weight_files/weights_grid.txt')
    weights = np.array(df['weights'])


    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharex=True, sharey=True, facecolor='w', edgecolor='k')

    # choose a colormap
    c_m = matplotlib.cm.viridis_r
    norm = matplotlib.colors.BoundaryNorm(np.arange(0.,len(tags)+1,1), c_m.N)
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    for ii, tag in enumerate(tags):

        z = float(tag[5:].replace('p','.'))

        mstars, SFRs, ws, DTMs = np.array([]), np.array([]), np.array([]), np.array([])

        fl = flares.flares('', sim_type='FLARES')

        for jj, mm in enumerate(fl.halos):
            try:
                with h5py.File(F'./data/flares_skirt_outputs.hdf5', 'r') as hf:
                    mstar       = np.array(hf[mm+'/'+tag+'/Galaxy'].get('Mstar'), dtype=np.float64)*1e10
                    ws          = np.append(ws, np.ones(len(mstar))*weights[int(jj)])
                    mstars      = np.append(mstars, mstar)
                    DTMs = np.append(DTMs, np.array(hf[mm+'/'+tag+'/Galaxy'].get('DTM'), dtype=np.float64))
            except:
                print (F"No data in region {mm} for tag {tag}")

        mbins = np.arange(9., 12.5, 0.5)

        plt_median(np.log10(mstars), DTMs, ws, mbins, axs, ii)


    fig.subplots_adjust(wspace=0, hspace=0, right=0.93)

    axs.set_xlabel(r'log$_{10}$(M$_{\star}$/M$_{\odot}$)', fontsize=14)
    axs.set_ylabel(r'DTM', fontsize=14)

    axs.grid(True, linestyle=(0, (0.5, 3)))
    axs.tick_params(axis="y",direction="in", which="both")
    axs.yaxis.get_ticklocs(minor=True)
    axs.minorticks_on()
    axs.tick_params(axis="x",direction="in", which="both")
    for label in (axs.get_xticklabels() + axs.get_yticklabels()):
        label.set_fontsize(13)


    cbaxes = fig.add_axes([0.95, 0.35, 0.02, 0.3])
    cbar = fig.colorbar(s_m, cax=cbaxes)
    cbar.set_label(r'$z$', fontsize = 15)
    cbar.set_ticks(np.arange(len(tags))+0.5)
    cbar.set_ticklabels(np.arange(5,11))
    cbaxes.invert_yaxis()
    for label in cbaxes.get_yticklabels():
        label.set_fontsize(13)


    plt.savefig(F"mstar_DTM.pdf", bbox_inches='tight', dpi=300)
    plt.show()
