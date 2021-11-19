"""

Figure 6 in paper

"""

import numpy as np
import pandas as pd
from functools import partial
import matplotlib, sys, h5py, schwimmbad
from uncertainties import unumpy
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
from scipy import integrate
import seaborn as sns
from FLARE.photom import lum_to_M, M_to_lum
from plot_obs import plot_UVLF
sns.set_context("paper")

from helpers import get_files
import flares


def get_lum(gal):

    try:
        data_loc = F'{gal}/flares_def1_sed.dat'
        data    = np.genfromtxt(data_loc)
        lam     = data[:,0]
        flux    = data[:,1]
        lum_int = data[:,2] * 1e-23 * 4 * np.pi * (1e6 * 3.086e18)**2
        lum     = data[:,1] * 1e-23 * 4 * np.pi * (1e6 * 3.086e18)**2
        nu      = 3e8/(lam*1e-6)


        ok      = np.where((lam>=8) & (lam<=1000))[0]
        lam_ir  = lam[ok]
        Lnu_ir  = lum[ok]
        nu_ir   = 3e8/(lam_ir*1e-6)
        Lir     = -integrate.simps(x=nu_ir, y=Lnu_ir, even='avg')/(3.826e33)


        ok = np.where(lam>8)
        lam1 = lam[ok]
        lum1 = lum[ok]
        lambda_peak = lam1[np.where(lum1==np.max(lum1))]

        return Lir, lambda_peak
    except:
        return np.array([0.]), np.array([0.])


def get_LF(data, bins, ws, weights):

    n = len(bins)-1
    LF, err, LF_tot = np.zeros(n), np.zeros(n), np.zeros(n)

    for ii in weights:
        ok = np.where(ws == ii)[0]
        if len(ok)>0:

            tmp = data[ok]
            hist, edges = np.histogram(tmp, bins = bins)

            err+=np.square(np.sqrt(hist)*ii)
            LF+=hist*ii
            LF_tot+=hist


    return LF, np.sqrt(err), LF_tot


def plot_data(axs, color, marker, Lirs, bins, bincen, binwidth, vol, num_of_regions, lw=2, label='_nolegend_', ls='solid'):

    hist, edges = np.histogram(Lirs, bins = bins)
    y = hist/(binwidth*vol*num_of_regions)
    err = np.sqrt(hist)/(binwidth*vol*num_of_regions)
    nonzero = np.where(hist>0)[0]

    ok      = np.where(y>0)[0]
    y       = y[ok]
    err     = err[ok]
    bincen  = bincen[ok]
    hist    = hist[ok]

    y_lo, y_up = np.log10(y)-np.log10(y-err), np.log10(y+err)-np.log10(y)
    mask = np.where(hist>5)[0]
    ok = np.where(hist>0)[0]

    if ls=='solid':
        axs.plot(bincen[ok], np.log10(y[ok]), color=color, ls='dashed', lw=lw)
        axs.errorbar(bincen[mask], np.log10(y[mask]), yerr=[y_lo[mask], y_up[mask]], color=color, marker=marker, lw=lw, label=label, ls=ls)
    else:
        axs.plot(bincen[ok], np.log10(y[ok]), color=color, ls=ls, lw=lw)



if __name__ == "__main__":

    fl = flares.flares('', sim_type='FLARES')

    h = 0.6777
    tags = ['010_z005p000', '009_z006p000', '008_z007p000']

    vol = (4/3)*np.pi*(14/h)**3

    quantiles = [0.84,0.50,0.16]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(13, 4), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()

    # choose a colormap
    c_m = matplotlib.cm.plasma
    dbins = np.arange(-0.3, 0.4, 0.1)
    norm = matplotlib.colors.BoundaryNorm(dbins, c_m.N)
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    df = pd.read_csv('./weight_files/weights_grid.txt')
    delta_all = np.array(df['log(1+delta)'])
    weights = np.array(df['weights'])

    bins        = np.arange(8.5, 13, 0.2)
    bincen      = (bins[1:]+bins[:-1])/2.
    binwidth    = bins[1:] - bins[:-1]

    xlabel = r'log$_{10}$(L$_{\mathrm{IR}}$/L$_{\odot}$)'
    ylabel = r'$\mathrm{log}_{10}(\Phi/(\mathrm{cMpc}^{-3}\mathrm{dex}^{-1}))$'
    title = 'LIR'
    x, y = 11.5, -5.3

    for ii, tag in enumerate(tags):

        z = float(tag[5:].replace('p','.'))

        for kk in range(len(dbins)-1):

            ok = np.where(np.logical_and(delta_all >= dbins[kk], delta_all < dbins[kk+1]))[0]
            num_regions = len(ok)

            if num_regions>0:
                this_reg = fl.halos[ok]
                Lir = np.array([])

                for jj, mm in enumerate(this_reg):

                    with h5py.File(F'./data/flares_skirt_outputs.hdf5', 'r') as hf:
                        Lir       = np.append(Lir, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('Lir'), dtype=np.float64))

                plot_data(axs[ii], s_m.to_rgba((dbins[kk]+dbins[kk+1])/2), 'o', np.log10(Lir), bins, bincen, binwidth, vol, num_regions)

        axs[ii].text(11.5, -7.3, r'$z = {}$'.format(z), fontsize = 13)


        Lirs, ws = np.array([]), np.array([])
        for jj, mm in enumerate(fl.halos):

            with h5py.File(F'./data/flares_skirt_outputs.hdf5', 'r') as hf:
                Lir    = np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('Lir'), dtype=np.float64)
                Lirs   = np.append(Lirs, Lir)

                ws        = np.append(ws, np.ones(len(Lir))*weights[int(mm)])


        LF, err, LF_tot = get_LF(np.log10(Lirs), bins, ws, weights)

        y   = LF/(binwidth*vol)
        err = err/(binwidth*vol)

        ok      = np.where(y>0)[0]
        y       = y[ok]
        err     = err[ok]
        x       = bincen[ok]
        LF_tot  = LF_tot[ok]

        y_lo, y_up = np.log10(y)-np.log10(y-err), np.log10(y+err)-np.log10(y)
        mask = np.where(LF_tot>5)[0]
        ok = np.where(LF_tot>0)[0]

        # print (y[ok], y[mask], y_lo[mask], y_up[mask])

        axs[ii].plot(x[ok], np.log10(y[ok]), color='black', ls='dashed', lw=3)
        axs[ii].errorbar(x[mask], np.log10(y[mask]), yerr=[y_lo[mask], y_up[mask]], color='black', marker='o', lw=3, label='Composite Function', ls='solid')



    axs[0].set_ylabel(ylabel, fontsize=15)


    for ax in axs:
        ax.grid(True, linestyle=(0, (0.5, 3)))
        # ax.legend(frameon=False, fontsize=11, loc=0, numpoints=1, markerscale=1)
        ax.tick_params(axis="y",direction="in")
        ax.tick_params(axis="x",direction="in")
        ax.set_xlim(10.5,13.5)
        ax.set_xticks([11,12,13])
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(14)

    axs[-1].legend(frameon=False, fontsize=11, loc=0, numpoints=1, markerscale=1)

    cbaxes = fig.add_axes([0.92, 0.35, 0.007, 0.3])
    fig.colorbar(s_m, cax=cbaxes)
    cbaxes.set_ylabel(r'$\mathrm{log}_{10}(1+\delta)$', fontsize = 14)
    for label in cbaxes.get_yticklabels():
        label.set_fontsize(13)

    fig.subplots_adjust(hspace=0, wspace=0)
    fig.text(0.46, 0.01, xlabel, va='center', fontsize=15)

    plt.savefig(F"Env_{title}.pdf", bbox_inches='tight', dpi=300)
    plt.show()
