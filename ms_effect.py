"""

Figure D1 in paper

"""

import numpy as np
import pandas as pd
import json
from functools import partial
import matplotlib, sys, h5py, re, schwimmbad
from uncertainties import unumpy
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from astropy.io import fits
from lmfit import Model
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import seaborn as sns
sns.set_context("paper")
# plt.style.use('classic')

from helpers import get_files
import flares


def piecewise_linear(x, x0, y0, m1, m2):
    """
    Fit a piecewise linear regression with two parts:

        y1 = m1*x + c1  (x <= x0)
        y2 = m2*x + c2  (x > x0)

    where,

        c1 = -m1*x0 + y0
        c2 = -m2*x0 + y0

        y0 = c2 + m2*x0 = c1 + m1*x0

    """
    return np.piecewise(x, [x < x0], [lambda x: m1*x + y0 - m1*x0, lambda x: m2*x + y0 - m2*x0])


def get_props(gal):

    data_loc = F'{gal}/flares_def1_sed.dat'
    data = np.genfromtxt(data_loc)

    lam = data[:,0]
    att = data[:,1] * 1e-23 * 4 * np.pi * (1e6 * 3.086e18)**2

    ok = np.where(lam>=8)[0]
    lam = lam[ok]
    Lnu = att[ok]

    nu = (3e8/(lam*1e-6)).astype(np.float64)

    Lir = -integrate.simps(x=nu, y=Lnu, even='avg')/(3.826e33)

    tmp_lam = np.linspace(8,1000,10000)
    tmp_Lnu = np.interp(tmp_lam, lam, Lnu)

    lambda_peak = tmp_lam[np.where(tmp_Lnu==np.max(tmp_Lnu))]

    # z, reg, num = re.findall(r'\d+', gal)
    # data_loc = F'skirt_resamp/z{z}/region_{reg}/gas_particles_{int(num)}.txt'
    # data = np.genfromtxt(data_loc)
    # ok = np.where(data[-1]<=1e6)[0]
    # Mmetal = np.sum(data[-2]*data[-3])
    # DTM = np.genfromtxt(F'skirt_resamp/z{z}/region_{reg}/DTM_values.txt')
    #
    # try:
    #     DTM = DTM[int(num)]
    # except:
    #     DTM = DTM
    #
    # Mdust = Mmetal*DTM

    return Lir, lambda_peak #, Mdust

def plt_median(out, bins, hist, ax, ii):

    bincen = (bins[1:]+bins[:-1])/2.
    xx, yy, yy84, yy16 = bincen, out[:,1], out[:,0], out[:,2]
    ok = np.isfinite(yy)
    ok1 = np.where(hist[ok]>=1)[0]

    ax.plot(xx[ok][ok1], yy[ok][ok1], marker='o', color=s_m.to_rgba(ii+0.5), alpha=0.7)
    ax.fill_between(xx[ok][ok1], yy16[ok][ok1], yy84[ok][ok1], color=s_m.to_rgba(ii+0.5), alpha=0.2)



if __name__ == "__main__":

    fl = flares.flares('', sim_type='FLARES')

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000']
    N = np.array([3844., 1772., 773., 331., 126., 44.])


    df = pd.read_csv('weight_files/weights_grid.txt')
    weights = np.array(df['weights'])
    quantiles = [0.84,0.50,0.16]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(13, 5), sharex=False, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()
    savename = 'lampeak_ms'

    # choose a colormap
    c_m = matplotlib.cm.viridis_r
    norm = matplotlib.colors.BoundaryNorm(np.arange(0.,6+1,1), c_m.N)
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    for ii, tag in enumerate(tags):

        ws, Lir, lambda_peak = np.array([]), np.array([]), np.array([])
        mstars, SFR = np.array([]), np.array([])


        for jj, mm in enumerate(fl.halos):

            with h5py.File(F'./data/flares_skirt_outputs.hdf5', 'r') as hf:
                mstar       = np.array(hf[mm+'/'+tag+'/Galaxy'].get('Mstar'), dtype=np.float64)*1e10
                SFR         = np.append(SFR, np.array(hf[mm+'/'+tag+'/Galaxy'].get('SFR'), dtype=np.float64))

                mstars    = np.append(mstars, mstar)
                ws        = np.append(ws, np.ones(len(mstar))*weights[int(mm)])

                Lir         = np.append(Lir, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('Lir'), dtype=np.float64))
                lambda_peak = np.append(lambda_peak, np.array(hf[mm+'/'+tag+'/Galaxy/Dust_Temperature'].get('Lambda_peak'), dtype=np.float64))

        mstars, Lir, SFR, sSFR = np.log10(mstars), np.log10(Lir), np.log10(SFR), np.log10(1e9*SFR/mstars)

        with open('samples/sfs_fit_%s.json'%tag) as f:
            p = json.load(f)

        x0,y0,m1,m2 = p['x0']['median'],p['y0']['median'],p['m1']['median'],p['m2']['median']
        phi = piecewise_linear(mstars-9.7,*[x0,y0,m1,m2])


        delx_low, delx_up = np.nanpercentile(SFR-phi, 16), np.nanpercentile(SFR-phi, 84)
        # print (delx)
        # print (delx_low, delx_up)
        lbins = np.arange(9, 13.5, 0.25)
        mbins = np.arange(9, 12.5, 0.25)
        sbins = np.arange(-2.5, 2., 0.25)


        x, bins = Lir, lbins

        ok = np.where(SFR-phi<=delx_low)[0]
        out = flares.binned_weighted_quantile(x[ok], lambda_peak[ok], np.ones(len(ok)), bins, quantiles)
        hist, edges = np.histogram(x[ok], bins)
        plt_median(out, bins, hist, axs[0], ii)


        ok = np.where((SFR-phi>delx_low) & (SFR-phi<delx_up))[0]
        out = flares.binned_weighted_quantile(x[ok], lambda_peak[ok], np.ones(len(ok)), bins, quantiles)
        hist, edges = np.histogram(x[ok], bins)
        plt_median(out, bins, hist, axs[1], ii)


        ok = np.where(SFR-phi>=delx_up)[0]
        out = flares.binned_weighted_quantile(x[ok], lambda_peak[ok], np.ones(len(ok)), bins, quantiles)
        hist, edges = np.histogram(x[ok], bins)
        plt_median(out, bins, hist, axs[2], ii)




    axs[1].set_xlabel(r'log$_{10}$(L$_{\mathrm{IR}}$/L$_{\odot}$)', fontsize=14)
    axs[0].set_ylabel(r'$\lambda_{\mathrm{peak}}$/$\mu$m', fontsize=14)


    twinax = axs[-1].twinx()
    twinax.set_ylim(2.898e3/30, 2.898e3/130)
    twinax.tick_params(axis="y",direction="in")
    twinax.set_ylabel(r'T$_{\mathrm{peak}}$/K', fontsize=14)
    for label in (twinax.get_yticklabels()):
        label.set_fontsize(13)

    fig.subplots_adjust(wspace=0, hspace=0)

    for ax in axs:
        ax.set_xlim(8.9, 12.9)
        ax.set_ylim(30,130)
        ax.set_yscale('log')
        ax.set_yticks([30, 40, 50, 60, 70, 80, 90, 110, 130])
        ax.set_yticklabels([30, 40, 50, 60, 70, 80, 90, 110, 130])
        ax.grid(True, linestyle=(0, (0.5, 3)))
        ax.legend(frameon=False, fontsize=10, loc=0, numpoints=1, markerscale=1, ncol=1)
        ax.tick_params(axis="y",direction="in")
        ax.tick_params(axis="x",direction="in")
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(13)

    plt.savefig(F"{savename}.pdf", bbox_inches='tight', dpi=300)
    plt.show()
