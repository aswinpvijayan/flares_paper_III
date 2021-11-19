"""

Figure C1 and E1 in paper for arguments 0 and 1 respectively.

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

def get_lum(gal):

    try:
        data_loc = F'{gal}/flares_def1_sed.dat'
        data    = np.genfromtxt(data_loc)
        lam     = data[:,0]
        flux    = data[:,1]
        lum_int = data[:,2] * 1e-23 * 4 * np.pi * (1e6 * 3.086e18)**2
        lum     = data[:,1] * 1e-23 * 4 * np.pi * (1e6 * 3.086e18)**2
        nu      = 3e8/(lam*1e-6)

        ok      = np.where((lam>=0.13) & (lam<=0.17))[0]
        filt    = np.zeros(len(lam))
        filt[ok] = 1.
        LFUV    = np.trapz(lum * filt, nu) / np.trapz(filt, nu)
        LFUVint = np.trapz(lum_int * filt, nu) / np.trapz(filt, nu)

        ok      = np.where((lam>=8) & (lam<=1000))[0]
        lam_ir  = lam[ok]
        Lnu_ir  = lum[ok]
        nu_ir   = 3e8/(lam_ir*1e-6)
        Lir     = -integrate.simps(x=nu_ir, y=Lnu_ir, even='avg')/(3.826e33)

        lam1    = np.linspace(245, 255, 1000)
        lum1    = np.interp(lam1, lam, lum)
        ok      = np.where((lam1>=249.5) & (lam1<=250.5))[0]
        filt    = np.zeros(len(lam1))
        filt[ok] = 1.
        L250    = np.trapz(lum1 * filt, lam1) / np.trapz(filt, lam1)

        ok = np.where(lam>8)
        lam1 = lam[ok]
        lum1 = lum[ok]
        lambda_peak = lam1[np.where(lum1==np.max(lum1))]

        return LFUV, Lir, L250, lambda_peak, LFUVint
    except:
        return np.array([0.]), np.array([0.]), np.array([0.]), np.array([0.]), np.array([0.])


def plot_data(axs, color, marker, LF, err, LF_tot, bincen, binwidth, vol, lw=2, label='_nolegend_', ls='solid'):

    y   = LF/(binwidth*vol)
    err = err/(binwidth*vol)

    ok      = np.where(y>0)[0]
    y       = y[ok]
    err     = err[ok]
    bincen  = bincen[ok]
    LF_tot  = LF_tot[ok]

    y_lo, y_up = np.log10(y)-np.log10(y-err), np.log10(y+err)-np.log10(y)
    mask = np.where(LF_tot>5)[0]
    ok = np.where(LF_tot>0)[0]
    # lolims = np.zeros(len(bincen))
    # lolims[mask] = True
    # y_up[mask] = 0.5
    # y_lo[mask] = 0.5

    # axs.errorbar(bincen, np.log10(y), yerr=[y_lo, y_up], lolims=lolims, color=color, marker=marker, lw=2)
    if ls=='solid':
        axs.plot(bincen[ok], np.log10(y[ok]), color=color, ls='dashed', lw=lw, label='_nolegend_')
        axs.errorbar(bincen[mask], np.log10(y[mask]), yerr=[y_lo[mask], y_up[mask]], color=color, marker=marker, lw=lw, label=label, ls=ls, ecolor=color)
    else:
        axs.plot(bincen[ok], np.log10(y[ok]), color=color, ls=ls, lw=lw, label='_nolegend_')

    return axs


def get_data(tag, loc=''):

    z = float(tag[5:].replace('p','.'))

    Lir, LFUV, mstars, SFR, LFUVlos, ws = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    reg = get_files(F'output{loc}/z_{int(z)}{loc}/*')

    for jj, mm in enumerate(reg):

        region = mm[-2:]

        with h5py.File(F'./data/flares_skirt_outputs.hdf5', 'r') as hf:
            mstar       = np.array(hf[region+'/'+tag+'/Galaxy'].get('Mstar'), dtype=np.float64)*1e10
            SFR         = np.append(SFR, np.array(hf[region+'/'+tag+'/Galaxy'].get('SFR'), dtype=np.float64))
            LFUVlos     = np.append(LFUVlos, np.array(hf[region+'/'+tag+'/Galaxy/Photometry'].get('LFUV_los'), dtype=np.float64))

            mstars    = np.append(mstars, mstar)
            ws        = np.append(ws, np.ones(len(mstar))*weights[int(region)])


        gals = get_files(mm+'/*')
        if len(gals)!=len(mstar):
            print (mm)
            sys.exit()


        pool = schwimmbad.MultiPool(processes=8)
        dat = np.array(list(pool.map(get_lum, gals)))
        pool.close()

        LFUV   = np.append(LFUV, dat[:,0].astype(np.float32))
        Lir    = np.append(Lir, dat[:,1].astype(np.float32))

    return LFUV, LFUVlos, Lir, ws



if __name__ == "__main__":

    h = 0.6777
    plt_option = int(sys.argv[1])
    vol = (4/3)*np.pi*(14/h)**3

    df = pd.read_csv('weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    if plt_option == 0:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=False, sharey=False, facecolor='w', edgecolor='k')
        tags = ['007_z008p000']
    else:
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(13, 8), sharex=True, sharey=True, facecolor='w', edgecolor='k')
        axs = axs.ravel()
        tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    norm = matplotlib.colors.Normalize(vmin=0.5, vmax=len(tags)+0.5)
    # choose a colormap
    c_m = matplotlib.cm.viridis_r
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    uvbins        = -np.arange(18, 26.5, 0.5)[::-1]
    uvbincen      = (uvbins[1:]+uvbins[:-1])/2.
    uvbinwidth    = uvbins[1:] - uvbins[:-1]


    irbins        = np.arange(8.5, 13, 0.2)
    irbincen      = (irbins[1:]+irbins[:-1])/2.
    irbinwidth    = irbins[1:] - irbins[:-1]


    for ii, tag in enumerate(tags):


        if plt_option == 0:
            title = 'noBC_compare'
            ncol=2


            LFUV, LFUVlos, Lir, ws = get_data(tag, loc='_resampNoBC')

            LF_uv, err_uv, LF_uvtot = get_LF(lum_to_M(LFUV), uvbins, ws, weights)
            plot_data(axs[0], s_m.to_rgba(3+0.5), 's', LF_uv, err_uv, LF_uvtot, uvbincen, uvbinwidth, vol, label='No BC', lw=2)

            LF_ir, err_ir, LF_irtot = get_LF(np.log10(Lir), irbins, ws, weights)
            plot_data(axs[1], s_m.to_rgba(3+0.5), 's', LF_ir, err_ir, LF_irtot, irbincen, irbinwidth, vol, label='No BC', lw=2)

            #-------------------------------------------------------------------#

            LFUV, LFUVlos, Lir, ws = get_data(tag)

            LF_uv, err_uv, LF_uvtot = get_LF(lum_to_M(LFUV), uvbins, ws, weights)
            plot_data(axs[0], s_m.to_rgba(3+0.5), 'o', LF_uv, err_uv, LF_uvtot, uvbincen, uvbinwidth, vol, lw=1, label='Fiducial')

            LF_ir, err_ir, LF_irtot = get_LF(np.log10(Lir), irbins, ws, weights)
            plot_data(axs[1], s_m.to_rgba(3+0.5), 'o', LF_ir, err_ir, LF_irtot, irbincen, irbinwidth, vol, lw=1, label='Fiducial')

            #-------------------------------------------------------------------#

            plot_UVLF(8, axs[0])


            axs[0].set_xlim([-18.5, -24.5])
            axs[0].set_ylim(-8.5, -2.5)
            axs[0].set_xlabel(r'M$_{1500}$', fontsize=14)
            axs[0].set_ylabel(r'$\mathrm{log}_{10}(\Phi/(\mathrm{cMpc}^{-3}\mathrm{Mag}^{-1}))$', fontsize=15)

            axs[1].set_xlim(10.5,13.5)
            axs[1].set_xticks([11,12,13])
            axs[1].set_ylim(-7.5, -2.5)
            axs[1].set_xlabel(r'log$_{10}$(L$_{\mathrm{IR}}$/L$_{\odot}$)', fontsize=15)
            axs[1].set_ylabel(r'$\mathrm{log}_{10}(\Phi/(\mathrm{cMpc}^{-3}\mathrm{dex}^{-1}))$', fontsize=15)


        else:
            z = float(tag[5:].replace('p','.'))
            title = 'LOS_UVLF_compare'
            xlabel = r'M$_{1500}$'
            ylabel = r'$\mathrm{log}_{10}(\Phi/(\mathrm{cMpc}^{-3}\mathrm{Mag}^{-1}))$'
            x, y = -22, -8.2
            ncol=1

            LFUV, LFUVlos, Lir, ws = get_data(tag)
            if z==10:
                label1, label2 = 'Fiducial', 'LOS'
            else:
                label1, label2 = '_nolegend_', '_nolegend_'


            LF_uv, err_uv, LF_uvtot = get_LF(lum_to_M(LFUV), uvbins, ws, weights)
            h1 = plot_data(axs[ii], s_m.to_rgba(ii+0.5), 's', LF_uv, err_uv, LF_uvtot, uvbincen, uvbinwidth, vol, lw=1, label='_nolegend_')

            LF_uv, err_uv, LF_uvtot = get_LF(lum_to_M(LFUVlos), uvbins, ws, weights)
            h2 = plot_data(axs[ii], s_m.to_rgba(ii+0.5), 's', LF_uv, err_uv, LF_uvtot, uvbincen, uvbinwidth, vol, lw=3, label='_nolegend_')


            axs[ii].text(x, y, r'$z = {}$'.format(z), fontsize = 13)

            plot_UVLF(int(z), axs[ii])

            axs[ii].set_xlim([-18.5, -24.5])
            axs[ii].set_ylim(-8.5, -2.5)

    if plt_option == 1:
        axs[-2].set_xlabel(xlabel, fontsize=15)
        axs[-1].errorbar([2,], [3.], yerr=[1], color='black', marker='s', lw=1, label='Fiducial')
        axs[-1].errorbar([2,], [3.], yerr=[1], color='black', marker='s', lw=3, label='LOS')
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.text(0.08, 0.5, ylabel, va='center', rotation='vertical', fontsize=15)


    for ax in axs:
        ax.grid(True, linestyle=(0, (0.5, 3)))
        ax.legend(frameon=False, fontsize=11, loc=0, numpoints=1, markerscale=1, ncol=ncol)
        ax.tick_params(axis="y",direction="in")
        ax.tick_params(axis="x",direction="in")
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(14)




    plt.savefig(F"app_{title}.pdf", bbox_inches='tight', dpi=300)
    plt.show()
