"""

Figure 7 and 8 in paper for arguments 0 and 1 respectively

"""

import numpy as np
import pandas as pd
from functools import partial
import matplotlib, sys, h5py, schwimmbad
import glob, os.path, re
from uncertainties import unumpy
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from astropy.io import fits
from scipy import integrate
from lmfit import Model
import seaborn as sns
sns.set_context("paper")


from FLARE.photom import lum_to_M, M_to_lum
import flares
from helpers import get_files


def IRX_beta_form(beta, dA_dbeta, beta_int):

    return 1.7 * (10**(0.4 * (dA_dbeta)*(beta - beta_int)) - 1.)

def theoretical_IRX_beta(axs):

    beta = np.arange(-4, 1.4, 0.1)
    # IRX_SMC = 1.7 * (10**(0.4*(0.91*(beta+2.3)))-1)
    # IRX_Muerer = 1.7 * (10**(0.4*(1.99*(beta+2.23)))-1)

    axs.plot(beta, np.log10(IRX_beta_form(beta, 0.91, -2.3)), label = 'Pettini+1998 (SMC)', ls = 'dashdot', color='black', alpha=0.7)
    axs.plot(beta, np.log10(IRX_beta_form(beta, 1.99, -2.23)), label = 'Meurer+1999 (Calzetti)', ls = 'dashed', color='black', alpha=0.7)

    axs.plot(beta, np.log10(IRX_beta_form(beta, 1.84, -2.43)), label = 'Reddy+2015', ls='solid', color='black', alpha=0.7)
    # axs.plot(beta, np.log10(IRX_beta_form(beta, 1.58, -1.94)), label = 'Takeuchi et al. 2012', ls='dotted', color='white')

    return axs


def plot_IRX_beta_obs(axs, z):

    #https://arxiv.org/pdf/1503.07596.pdf Capak et al. 2015
    #https://arxiv.org/pdf/1707.02980.pdf Bariˇsic et al. 2017

    #Capak2015
    Cz = np.array([5.690, 5.670, 5.546, 5.540, 5.310, 5.310, 5.310, 5.310, 5.310, 5.250, 5.148, 5.148, 5.548, 5.659, 5.659])
    Luv = np.array([11.21, 11.15, 11.08, 11.28, 11.45, 11.47, 11.11, 11.00, 10.81, 11.05, 11.04, 10.57, 10.95, 11.14, 10.23])
    Luv_err = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.07, 0.07, 0.07, 0.02, 0.02, 0.04, 0.02, 0.02, 0.05])
    LUV = unumpy.uarray(Luv, Luv_err)
    Lir = np.array([10.32, 10.30, 10.53, 11.13, 10.30, 11.13, 10.26, 10.87, 10.79, 10.35, 10.26, 10.26, 11.54, 11.94, 11.64])
    Lir_err = np.array([0.2, 0.2, 0.2, 0.54, 0.2, 0.23, 0.23, 0.23, 0.23, 0.2, 0.2, 0.2, 0.19, 0.08, 0.08])
    LIR = unumpy.uarray(Lir, Lir_err)

    IRX = LIR - LUV
    IRX_uplim = np.ones(len(Lir_err))
    IRX_uplim[Lir_err!=0] = 0

    beta = np.array([-1.92, -1.82, -1.72, -2.06, -1.01, -1.14, -0.59, -1.50, -1.30, -1.39, -1.42, -0.10, -1.59, -1.92, -1.47])
    beta_up = np.array([0.14, 0.10, 0.12, 0.13, 0.06, 0.12, 1.05, 1.05, 0.51, 0.15, 0.19, 0.29, 0.22, 0.24, 0.77])
    beta_low = np.array([0.11, 0.10, 0.15, 0.15, 0.12, 0.14, 1.12, 1.22, 0.37, 0.17, 0.18, 0.29, 0.23, 0.17, 0.44])

    zok = np.logical_and(Cz>=z-0.5, Cz<=z+0.5)
    if np.sum(zok)>0:
        ok = IRX_uplim[zok]==1
        axs.errorbar(beta[zok][ok], unumpy.nominal_values(IRX[zok])[ok], xerr=[beta_low[zok][ok], beta_up[zok][ok]], yerr=unumpy.std_devs(IRX[zok])[ok], uplims=IRX_uplim[zok][ok], ls='None', color='brown', markersize=1, marker='s')

        axs.errorbar(beta[zok][~ok], unumpy.nominal_values(IRX[zok])[~ok], xerr=[beta_low[zok][~ok], beta_up[zok][~ok]], yerr=unumpy.std_devs(IRX[zok])[~ok], ls='None', label = 'Capak+2015', color='brown', markersize=1, marker='s')

    # fud_data = np.genfromtxt('fudamoto2020_IRX_beta.txt', delimiter=',')
    # ok = (fud_data[:,-1]==0)
    # axs.errorbar(fud_data[:,1][~ok], fud_data[:,3][~ok] - fud_data[:,2][~ok], yerr = np.ones(np.sum(~ok))*0.2, uplims=np.ones(np.sum(~ok)), marker='o', color='orange', ls='None', markersize=2)
    #
    # axs.errorbar(fud_data[:,1][ok], fud_data[:,3][ok] - fud_data[:,2][ok], xerr = np.ones(np.sum(ok))*0.25, yerr = np.ones(np.sum(ok))*0.15, label=r'Fudamoto et al. (2020) ($z\sim 5$)', marker='o', color='orange', ls='None', markersize=2)

    y = np.array([1.54, 0.12, 30.26, 10.65])
    yerr_up = np.log10(y + np.array([0.95, 0.74, 1.73, 4.66])) - np.log10([0.95, 0.74, 1.73, 4.66])
    yerr_up[1] = 0
    yerr_low = np.log10(y) - np.log10(y - np.array([0.95, 0.74, 1.73, 4.66]))
    yerr_low[1] = 0.25
    y = np.log10(y)
    y[1] = 0
    yuplims = np.zeros(len(y))
    yuplims[1] = 1
    axs.errorbar([-2, -1.5, -0.8, 1.6], y, yerr = [yerr_low, yerr_low], uplims=yuplims, ls='None', label = r'Bouwens+2020 (Median, $z\geq 3.5$)', color='green', markersize=5, marker='s', alpha=0.6)

    if z<=5:
        axs.errorbar([-2., -1.47, -0.86], [-0.09, 0.20, 0.69], yerr=[[0.22, 0.29, 0.12], [0.16, 0.17, 0.11]], label=r'Fudamoto+2020 (Stacked, $z\sim 4.5$)', marker='d', color='orange', ls='None', markersize=6, alpha=0.6)

    if z<=6:
        axs.errorbar([-2.28, -1.84, -1.43], [-0.03, -0.15, 0.12], yerr=[[0.25, 0.13, 0.38], [0, 0.16, 0.17]], uplims=[1,0,0], label=r'Fudamoto+2020 (Stacked, $z\sim 5.5$)', marker='s', color='orange', ls='None', markersize=6, alpha=0.6)


    Ho_z = np.array([9.11, 8.38, 8.31, 7.5, 7.21, 7.15, 6.96, 6.90, 6.85, 6.81, 6.60])
    Ho_beta = np.array([-0.76, -1.63, -1.72, -1.10, -2.6, -1.85, -2.07, -0.88, -1.22, -1.18, -2.0])
    Ho_betaerr = np.array([0.16, 0.53, 0.50, 0.83, 0.25, 0.54, 0.26, 0.58, 0.51, 0.53, 0.48])
    Ho_betauplims = np.zeros(len(Ho_z))
    Ho_betauplims[4]=1
    Ho_IRX = np.array([0.12, 0.86, 0.61, 1.09, 0.33, 0.49, 0.49, 1.80, 0.51, 0.37, 0.16])
    Ho_IRXerr = np.array([0.25, 0.22, 0.09, 0.05, 0.25, 0.12, 0.25, 0.04, 0.25, 0.25, 0.25])
    Ho_IRXuplim = np.zeros(len(Ho_z))
    Ho_IRXuplim[[0,4,6,8,9,10]] = 1

    ok = np.logical_and(Ho_z>=z-0.5, Ho_z<=z+0.5)
    if np.sum(ok)>0:

        axs.errorbar(Ho_beta[ok], Ho_IRX[ok], xerr=Ho_betaerr[ok], yerr=Ho_IRXerr[ok], xuplims=Ho_betauplims[ok], uplims=Ho_IRXuplim[ok], label=rF'Hashimoto+2019', color='indigo', ls='None', markersize=1, marker='s')


    #Harikane2020
    H20_z = np.array([6.0293, 6.0901, 6.2037])
    H20_beta = np.array([-2., -2.6, -0.1])
    H20_betaerr = np.array([0.5, 0.6, 0.5])
    H20_IRX = np.array([11.73-11.43, 11.45-11.46, 11.65-11.63])
    H20_IRXerr = np.array([0.09, 0.25, 0.06])
    H20_IRXlolim = np.array([0,1,0])

    if z==6:

        axs.errorbar(H20_beta, H20_IRX, yerr=H20_IRXerr, lolims=H20_IRXlolim, label=rF'Harikane+2020', color='olive', ls='None', markersize=1, marker='s')


    #Bowler+2021 https://arxiv.org/abs/2110.06236
    zs = np.array([7.152, 7.0611, 6.984, 6.686, 6.633, 6.56])
    beta = np.array([-2.21, -2.32, -2.01, -1.90, -1.95, -2.35])
    beta_up = np.array([0.18, 0.21, 0.22, 0.23, 0.15, 0.23])
    beta_low = np.array([0.28, 0.07, 0.30, 0.19, 0.15, 0.47])
    IRX = np.array([0.47, -0.28, 0.29, 0.08, -0.07, -0.38])
    IRX_up = np.array([0.16, 0.17, 0.25, 0.16, 0.23, 0.])
    IRX_low = np.array([0.21, 0.25, 0.43, 0.20, 0.43, 0.25])
    uplims = np.zeros(len(IRX))
    uplims[-1] = 1

    # if z==7:
    #
    #     axs.errorbar(beta, IRX, yerr=[IRX_low, IRX_up], xerr=[beta_low, beta_up], uplims=uplims, label=rF'Bowler+2021', color='violet', ls='None', markersize=1, marker='s')


    return axs

def plot_IRX_mstar_obs(axs, z):

    #Bouwens2020
    mstar = np.array([10.8, 9.5, 9])
    IRX = np.array([19.08, 4.12, 0.41])
    IRX_up = np.log10(IRX + np.array([1.02, 3.23, 0.61])) - np.log10(IRX)
    IRX_low = np.log10(IRX) - np.log10(IRX - np.array([1.02, 2.38, 0.61]))
    IRX_low[2] = 5
    axs.errorbar(mstar, np.log10(IRX), yerr=[IRX_low, IRX_up], ls='None', label = r'Bouwens+2020 (Median, $z>3.5$)', color='green', markersize=5, marker='s', alpha=0.6)

    if z<=5:
        axs.errorbar([9.74, 10.24], [-0.09, 0.37], yerr=[[0.44, 0.14], [0.23, 0.1]], label=r'Fudamoto+2020 (Stacked, $z\sim 4.5$)', marker='d', color='orange', ls='None', markersize=6, alpha=0.6)
    if z<=6:
        axs.errorbar([9.61, 10.46], [0.04, 0.05], yerr=[[0.25, 0.21], [0, 0.19]], uplims=[1,0], label=r'Fudamoto+2020 (Stacked, $z\sim 5.5$)', marker='s', color='orange', ls='None', markersize=6, alpha=0.6)


    #ALPINE collaboration
    data = pd.read_csv('Obs_data/ALPINE_merged_catalogs.csv')
    ALPINEz = np.array(data['z_orig'])
    Lir = np.array(data['LIR'])
    Lir_err = np.array(data['unc_LIR'])

    LFUV = M_to_lum(np.array(data['M_FUV'])) * 2e15 / 3.826e33
    LFUV_lo = M_to_lum(np.array(data['M_FUV_low1sig'])) * 2e15 / 3.826e33
    LFUV_up = M_to_lum(np.array(data['M_FUV_high1sig'])) * 2e15 / 3.826e33
    LFUV, LFUV_lo, LFUV_up = np.log10(LFUV), np.log10(LFUV) - np.log10(LFUV_lo), np.log10(LFUV_lo) - np.log10(LFUV)
    LFUV_err = np.max([LFUV_lo, LFUV_up], axis=0)

    mstar = np.array(data['logMstar'])
    mstar_lo = mstar - np.array(data['logMstar_loweff1sig'])
    mstar_up = np.array(data['logMstar_higheff1sig']) - mstar

    ok = np.isnan(Lir)
    Lir_uplim = np.zeros(len(Lir))
    Lir_uplim[ok] = 1
    Lir[ok] = Lir_err[ok]
    Lir, Lir_err = np.log10(Lir), np.log10(Lir) - np.log10(Lir_err)
    Lir_err[ok] = 0.25

    IRX = Lir-LFUV
    IRX_err = np.sqrt(Lir_err**2 + LFUV_err**2)

    ok = np.logical_and(ALPINEz>z-0.5, ALPINEz<z+0.5)
    if np.sum(ok)>0:
        axs.errorbar(mstar[ok], IRX[ok], xerr=[mstar_lo[ok], mstar_up[ok]], yerr=IRX_err[ok], uplims=Lir_uplim[ok], color='grey', ls='None', markersize=1, marker='s', label='ALPINE', alpha=0.3)


    return axs


if __name__ == "__main__":

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    inp = int(sys.argv[1])

    fl = flares.flares('', sim_type='FLARES')

    df = pd.read_csv('weight_files/weights_grid.txt')
    weights = np.array(df['weights'])
    quantiles = [0.84,0.50,0.16]

    # add = ''#'_resampNoBC'#_noresamp'#'_resampNoBC'
    # reg = get_files(F'output/z_{int(z)}{add}/*')

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(13, 8), sharex=True, sharey=True, facecolor='w', edgecolor='k')
    axs = axs.ravel()

    for ii, tag in enumerate(tags):

        Lir, LFUV, mstars, SFR, ws, beta  = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        z = float(tag[5:].replace('p','.'))


        for jj, mm in enumerate(fl.halos):

            with h5py.File(F'./data/flares_skirt_outputs.hdf5', 'r') as hf:
                mstar       = np.array(hf[mm+'/'+tag+'/Galaxy'].get('Mstar'), dtype=np.float64)*1e10
                SFR         = np.append(SFR, np.array(hf[mm+'/'+tag+'/Galaxy'].get('SFR'), dtype=np.float64))

                mstars    = np.append(mstars, mstar)
                ws        = np.append(ws, np.ones(len(mstar))*weights[int(mm)])


                LFUV   = np.append(LFUV, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('LFUV'), dtype=np.float64))
                Lir    = np.append(Lir, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('Lir'), dtype=np.float64))
                beta   = np.append(beta, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('beta'), dtype=np.float64))


        LFUV = LFUV*(2E15)/(3.826e33)
        IRX = Lir/LFUV


        sSFR = SFR/mstars
        c = np.log10(sSFR*1e9) #np.log10(sSFRs/np.nanmedian(sSFRs))
        if tag == tags[0]:
            vmin, vmax=np.min(c[np.isfinite(c)])-0.1, np.max(c[np.isfinite(c)])+0.1

        if inp==0:
            hb = axs[ii].hexbin(beta, np.log10(IRX), C = c, gridsize=(40,20), cmap=plt.cm.get_cmap('coolwarm_r'), reduce_C_function=np.median, linewidths=0., mincnt=0, extent=[*[-2.3, 1.3], *[-1, 3]], vmin=vmin, vmax=vmax)

            theoretical_IRX_beta(axs[ii])
            plot_IRX_beta_obs(axs[ii], z)

            xlim = [-2.3, 1.3]
            xlabel = r'$\beta$'
            title = 'beta'
            x, y = 0.5, -0.1


        else:
            # axs[ii].scatter(np.log10(mstars), np.log10(IRX), s=1)
            hb = axs[ii].hexbin(np.log10(mstars), np.log10(IRX), C = c, gridsize=(45,20), cmap=plt.cm.get_cmap('coolwarm_r'), reduce_C_function=np.median, linewidths=0., mincnt=0, extent=[*[9., 11.5], *[-1, 3]], vmin=vmin, vmax=vmax)

            mbins = np.arange(9,12,0.3)
            mbincen = (mbins[1:]+mbins[:-1])/2.


            out = flares.binned_weighted_quantile(np.log10(mstars), np.log10(IRX), ws, mbins, quantiles)
            xx, yy, yy84, yy16 = mbincen, out[:,1], out[:,0], out[:,2]
            ok = np.isfinite(yy)

            axs[ii].plot(xx[ok], yy[ok], ls='dashed', color='black', alpha=.5, lw=2)
            axs[ii].fill_between(xx[ok], yy16[ok], yy84[ok], color='black', alpha=0.2)


            plot_IRX_mstar_obs(axs[ii], z)

            xlim = [8.9, 11.4]
            xlabel = r'log$_{10}$(M$_{\star}$/M$_{\odot}$)'
            title = 'mstar'
            x, y = 10.9, -0.1


        axs[ii].text(x, y, r'$z = {}$'.format(z), fontsize = 12)


    for ax in axs:
        ax.grid(True, linestyle=(0, (0.5, 3)))
        ax.legend(fontsize=8, frameon=False, loc=2, numpoints=1, markerscale=1)
        ax.tick_params(axis="y",direction="in")
        ax.tick_params(axis="x",direction="in")
        ax.set_ylim(-0.9, 4.5)
        ax.set_xlim(xlim)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(14)


    axs[-2].set_xlabel(xlabel, fontsize = 13)


    # axs.legend(fontsize=7, frameon=False)
    fig.text(0.08, 0.5, r'log$_{10}$(IRX)', va='center', rotation='vertical', fontsize=15)

    fig.subplots_adjust(right = 0.91, wspace=0, hspace=0)
    cbaxes = fig.add_axes([0.92, 0.25, 0.005, 0.5])
    fig.colorbar(hb, cax=cbaxes)
    cbaxes.set_ylabel(r'log$_{10}$(sSFR/Gyr$^{-1}$)', fontsize = 15)
    for label in cbaxes.get_yticklabels():
        label.set_fontsize(10)

    plt.savefig(F"IRX_{title}_sSFR.pdf", bbox_inches='tight', dpi=300)
    plt.show()
