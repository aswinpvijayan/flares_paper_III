"""

Figure 3, 4 and 5 in paper for sys.argv[1] corresponding to
0, 1 and 2 respectively.

Figure B1 for argument 1 1

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

import flares
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

def Lir_FIRE(Lir, z):
    #for fdust=0.4

    phistar     = 10**(-3.10) * ((1+z)/7)**(-1.55)
    Lstar       = 10**(10.84) * ((1+z)/7)**(-4.98)
    alpha1      = 0.53 * ((1+z)/7)**(0.43)
    alpha2      = 1.37 * ((1+z)/7)**(0.69)

    L1 = (Lir/Lstar)**alpha1
    L2 = (Lir/Lstar)**alpha2

    phi_ir = phistar/(L1 + L2)

    return phi_ir


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
        axs.plot(bincen[ok], np.log10(y[ok]), color=color, ls='dashed', lw=lw)
        axs.errorbar(bincen[mask], np.log10(y[mask]), yerr=[y_lo[mask], y_up[mask]], color=color, marker=marker, lw=lw, label=label, ls=ls)
    else:
        axs.plot(bincen[ok], np.log10(y[ok]), color=color, ls=ls, lw=lw)


def Lir_CennKimm(z, axs):

    Lir = np.array([11., 11.2, 11.4, 11.6, 11.8, 12.])
    phi = np.array([-2.5, -2.65, -3.18, -3.68, -4.01, -4.42])
    phi_err = np.array([[-2.57, -2.73, -3.35, -4.04, -4.80, -4.04], [-2.45, -2.58, -3.06, -3.47, -3.74, -5.4]])
    phi_err[0] = phi - phi_err[0]
    phi_err[1] = phi_err[1] - phi

    if z==7:
        axs.errorbar(Lir, phi, yerr=phi_err, label='Cen $\&$ Kimm 2014', color = 'olive', marker='s', markerfacecolor='None', markeredgecolor='olive', ls = 'dotted', alpha=0.6)

def logphiLir(logLir, logphistar, logLstar, exponent):

    # return (10**logphistar) * (10**((exponent+1)*logLir)) * 10**(-(exponent+1)*logLstar) * np.log(10)
    # return logphistar + (exponent+1.)*logLir - exponent*logLstar + np.log10(np.log(10))
    # return (10**logphistar) * (10**((logLir-logLstar)*exponent)) * 10**(logLir) * np.log(10)
    return logphistar + (logLir-logLstar)*exponent

def Zavala2021(z, axs):

    # Lir Fit function from Zavala+2021 (https://arxiv.org/abs/2101.04734v1)

    logLirbin = np.arange(11, 14, 0.2)
    logLir = (logLirbin[:-1]+logLirbin[1:])/2.
    binsize = (10**logLirbin[1:]-10**logLirbin[:-1])
    logphi = np.zeros(len(logLir))

    print (len(logLirbin), len(logLir), len(binsize), len(logphi))

    if z==7:
        logLstar    = 12.91
        logphistar  = -6.10
        alpha       = -0.42
        beta        = -3.0

        ok = logLir>=logLstar

        logphi[~ok]  = logphiLir(logLir[~ok], logphistar, logLstar, alpha)
        logphi[ok] = logphiLir(logLir[ok], logphistar, logLstar, beta)

        print (logphi)

        axs.errorbar(logLir, logphi-np.log10(0.2), label='Zavala 2021 IRLF fit', color = 'olive',  markeredgecolor='olive', ls = 'dashdot', alpha=0.6)


def plot_IR_obs(z, axs):

    x_W19 = np.array([12.25, 12.75, 13.25])

    # y_W19_z4 = np.array([7.1666e-05,2.5525e-05,3.6815e-07])
    # y_W19_z4err = np.array([2.9656e-06,1.7698e-06,2.1255e-07])
    # y_W19_z4_loerr = np.log10(y_W19_z4) - np.log10(y_W19_z4-y_W19_z4err)
    # y_W19_z4_uperr = np.log10(y_W19_z4+y_W19_z4err) - np.log10(y_W19_z4)
    #
    #
    # y_W19_z5 = np.array([4.7607e-05, 1.7026e-05,9.7599e-07])
    # y_W19_z5err = np.array([2.2721e-06,1.3588e-06, 3.2533e-07])
    # y_W19_z5_loerr = np.log10(y_W19_z5) - np.log10(y_W19_z5-y_W19_z5err)
    # y_W19_z5_uperr = np.log10(y_W19_z5+y_W19_z5err) - np.log10(y_W19_z5)

    y_W19_z5 = np.array([-4.20, -4.63, -6.14])
    y_W19_z5err = np.array([[-4.28, -4.71, -6.29], [-4.13, -4.55, -6.0]])
    y_W19_z5err[0] = y_W19_z5 - y_W19_z5err[0]
    y_W19_z5err[1] = y_W19_z5err[1] - y_W19_z5

    y_W19_z6 = np.array([-5.05, -5.6, -6.53])
    y_W19_z6err = np.array([[-5.13, -5.7, -6.82], [-4.98, -5.53, -6.35]])
    y_W19_z6err[0] = y_W19_z6 - y_W19_z6err[0]
    y_W19_z6err[1] = y_W19_z6err[1] - y_W19_z6


    x_Gr13 = np.array([12.75, 13.25, 13.75])
    y_Gr13 = np.array([-4.65, -5.75, -7.18])
    y_Gr13_err = np.array([0.14, 0.13, 0.43])

    if z==5:
        axs.errorbar(x_Gr13, y_Gr13, yerr=y_Gr13_err, label='Gruppioni+2013 ($3<z<4.2$)', color='brown', marker='D', alpha=0.6, ls = 'None')

        # axs.errorbar(x_W19, np.log10(y_W19_z4), yerr=np.array([y_W19_z4_loerr, y_W19_z4_uperr]), label='Wang+2019 ($4.2<z<5$)', color='black', marker='x', alpha=0.6)

        axs.errorbar(x_W19, y_W19_z5, yerr=y_W19_z5err, label='Wang+2019', color='black', marker='x', alpha=0.6, ls = 'None')

        axs.errorbar(np.array([12., 12.5]), [-3.37, -4.10], yerr=[[0.58, 0.78], [0.40, 0.52]], label=r'Gruppioni+2020 ($3.5<z<4.5$)', color='magenta', marker='D', alpha=0.6, ls = 'None')


    if ((z==5) or (z==6)):
        # axs.errorbar(x_W19, np.log10(y_W19_z5), yerr=np.array([y_W19_z5_loerr, y_W19_z5_uperr]), label='Wang+2019 ($5<z<6$)', color='grey', marker='X', alpha=0.6)

        axs.errorbar(np.array([11.5, 12., 12.5, 13.]), [-3.96, -3.60, -4.12, -3.87], yerr=[[0.78, 0.52, 0.78, 0.62], [0.55, 0.41, 0.59, 0.43]], label=r'Gruppioni+2020 ($4.5<z<6$)', color='orange', marker='D', alpha=0.6, ls = 'None')

    if z==6:
        axs.errorbar(x_W19, y_W19_z6, yerr=y_W19_z6err, label='Wang+2019', color='black', marker='x', alpha=0.6, ls = 'None')


    # Lir_CennKimm(z,axs)

    # Zavala2021(z, axs)



def plot_L250_obs(z, axs):


    if z==5:
        axs.errorbar(np.arange(25.7, 26.25, 0.1) - np.log10(3.826e26) + np.log10(1.2e12), [-4.47, -5.27, -4.93, -5.37, -5.49, -5.97], yerr=[0.12,0.18,0.11,0.14,0.15,0.23], label=r'Koprowski+2017 ($3.5<z<4.5$)', color = 'grey', marker='d', ls = 'None')

        axs.errorbar(np.array([10, 10.25, 10.5, 10.75]), [-3.52, -3.36, -3.68, -4.10], yerr=[[0.76, 0.47, 0.45, 0.76], [0.59, 0.48, 0.48, 0.68]], label=r'Gruppioni+2020 ($3.5<z<4.5$)', color='magenta', marker='D', alpha=0.6, ls = 'None')
        axs.errorbar(np.array([10.5, 10.75, 11, 11.25, 11.5, 11.75]), [-3.86, -3.59, -3.52, -3.60, -3.87, -4.18], yerr=[[0.76, 0.52, 0.48, 0.48, 0.52, 0.78], [0.59, 0.49, 0.46, 0.46, 0.49, 0.64]], label=r'Gruppioni+2020 ($4.5<z<6$)', color='orange', marker='D', alpha=0.6, ls = 'None')

    if z==6:
        axs.errorbar(np.array([10.5, 10.75, 11, 11.25, 11.5, 11.75]), [-3.86, -3.59, -3.52, -3.60, -3.87, -4.18], yerr=[[0.76, 0.52, 0.48, 0.48, 0.52, 0.78], [0.59, 0.49, 0.46, 0.46, 0.49, 0.64]], label=r'Gruppioni+2020 ($4.5<z<6$)', color='orange', marker='D', alpha=0.6, ls = 'None')


if __name__ == "__main__":

    fl = flares.flares('', sim_type='FLARES')

    h = 0.6777
    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']


    inp = int(sys.argv[1])
    if inp == 1:
        print ("Checking for SMBH bolometric option")
        try:
            BH = int(sys.argv[2])
            print ("Plotting IRLF as stellar IR + SMBH bolometric luminosity")
        except:
            BH = 0
            print ("Plotting IRLF as stellar IR")

    vol = (4/3)*np.pi*(14/h)**3


    df = pd.read_csv('weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    if inp==2:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(13, 4), sharex=True, sharey=True, facecolor='w', edgecolor='k')
        axs = axs.ravel()
        tags=tags[:3]
    else:
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(13, 8), sharex=True, sharey=True, facecolor='w', edgecolor='k')
        axs = axs.ravel()

    norm = matplotlib.colors.Normalize(vmin=0.5, vmax=len(tags)+0.5)
    # choose a colormap
    c_m = matplotlib.cm.viridis_r
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    for ii, tag in enumerate(tags):

        z = float(tag[5:].replace('p','.'))

        Lirs, LFUVs, mstars, SFRs, L250s, LFUVints = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        MBHs, MBHaccs, LBH_bol, LFUVlos = np.array([]), np.array([]), np.array([]), np.array([])

        ws = np.array([])

        for jj, mm in enumerate(fl.halos):

            try:
                with h5py.File(F'./data/flares_skirt_outputs.hdf5', 'r') as hf:
                    mstar       = np.array(hf[mm+'/'+tag+'/Galaxy'].get('Mstar'), dtype=np.float64)*1e10
                    ws          = np.append(ws, np.ones(len(mstar))*weights[int(jj)])


                    mstars      = np.append(mstars, mstar)
                    SFRs        = np.append(SFRs, np.array(hf[mm+'/'+tag+'/Galaxy'].get('SFR'), dtype=np.float64))
                    LFUVlos     = np.append(LFUVlos, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('LFUV_los'), dtype=np.float64))
                    LBH_bol     = np.append(LBH_bol, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('LBH_bol'), dtype=np.float64))


                    LFUVs       = np.append(LFUVs, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('LFUV'), dtype=np.float64))
                    Lirs        = np.append(Lirs, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('Lir'), dtype=np.float64))
                    L250s       = np.append(L250s, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('L250'), dtype=np.float64))
                    LFUVints    = np.append(LFUVints, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('LFUV_int'), dtype=np.float64))
            except:
                print ("No data in region")


        if inp == 0:
            #--------------------UV LF------------------#

            bins        = -np.arange(18, 26.5, 0.5)[::-1]
            bincen      = (bins[1:]+bins[:-1])/2.
            binwidth    = bins[1:] - bins[:-1]
            LF_uv, err_uv, LF_uvtot = get_LF(lum_to_M(LFUVs), bins, ws, weights)

            plot_data(axs[ii], s_m.to_rgba(ii+0.5), 's', LF_uv, err_uv, LF_uvtot, bincen, binwidth, vol)

            plot_UVLF(int(z), axs[ii])

            axs[ii].set_xlim([-18.5, -24.5])
            axs[ii].set_ylim(-8.5, -2.5)

            xlabel = r'M$_{1500}$'
            ylabel = r'$\mathrm{log}_{10}(\Phi/(\mathrm{cMpc}^{-3}\mathrm{Mag}^{-1}))$'
            title = 'LFUVlos'
            x, y = -22, -8.2

        if inp == 1:
            #-----------------TIR LF---------------------#

            bins        = np.arange(8.5, 13, 0.2)
            bincen      = (bins[1:]+bins[:-1])/2.
            binwidth    = bins[1:] - bins[:-1]
            LF_ir, err_ir, LF_irtot = get_LF(np.log10(Lirs), bins, ws, weights)

            if BH==0:
                plot_data(axs[ii], s_m.to_rgba(ii+0.5), '8', LF_ir, err_ir, LF_irtot, bincen, binwidth, vol)

                plot_IR_obs(z, axs[ii])

                bins        = np.arange(8.75, 12.5, 0.5)
                bincen      = (bins[1:]+bins[:-1])/2.
                axs[ii].plot(bincen,  np.log10(Lir_FIRE(10**bincen, z)), ls='dotted', label='FIRE-2 IRLF fit', lw=2, color='brown')

                title = 'LIR'

            else:
                plot_data(axs[ii], s_m.to_rgba(ii+0.5), '8', LF_ir, err_ir, LF_irtot, bincen, binwidth, vol, lw=1, label=r'Only Stellar')

                LF_ir, err_ir, LF_irtot = get_LF(np.log10(Lirs+LBH_bol), bins, ws, weights)

                plot_data(axs[ii], s_m.to_rgba(ii+0.5), '8', LF_ir, err_ir, LF_irtot, bincen, binwidth, vol, lw=2, label=r'Stars+SMBH L$_{\mathrm{Bol}}$')
                title = 'LIR+BH'

            axs[ii].set_xlim(10.5,13.5)
            axs[ii].set_xticks([11,12,13])
            axs[ii].set_ylim(-7.5, -2.5)

            xlabel = r'log$_{10}$(L$_{\mathrm{IR}}$/L$_{\odot}$)'
            ylabel = r'$\mathrm{log}_{10}(\Phi/(\mathrm{cMpc}^{-3}\mathrm{dex}^{-1}))$'

            x, y = 12.92, -4.7

        if inp == 2:
            #-----------------250micron LF---------------------#

            bins        = np.arange(9, 13.5, 0.2)
            bincen      = (bins[1:]+bins[:-1])/2.
            binwidth    = bins[1:] - bins[:-1]
            LF_250, err_250, LF_250tot = get_LF(np.log10(1.2e12*L250s/3.826e33), bins, ws, weights)

            plot_data(axs[ii], s_m.to_rgba(ii+0.5), 'o', LF_250, err_250, LF_250tot, bincen, binwidth, vol)

            plot_L250_obs(z, axs[ii])

            axs[ii].set_xlim(9, 11.4)
            axs[ii].set_ylim(-7.5, -2.5)
            axs[ii].set_yticks(np.arange(-7, -2.5, 1))

            xlabel = r'log$_{10}$(L$_{250}$/L$_{\odot}$)'
            ylabel = r'$\mathrm{log}_{10}(\Phi/(\mathrm{cMpc}^{-3}\mathrm{dex}^{-1}))$'
            title = 'L250'
            x, y = 10.9, -6.

        #--------------------------------------#

        axs[ii].text(x, y, r'$z = {}$'.format(z), fontsize = 13)


    axs[-2].set_xlabel(xlabel, fontsize=15)


    for ax in axs:
        ax.grid(True, linestyle=(0, (0.5, 3)))
        ax.legend(frameon=False, fontsize=11, loc=0, numpoints=1, markerscale=1)
        ax.tick_params(axis="y",direction="in")
        ax.tick_params(axis="x",direction="in")
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(14)


    fig.subplots_adjust(hspace=0, wspace=0)
    fig.text(0.08, 0.5, ylabel, va='center', rotation='vertical', fontsize=15)

    plt.savefig(F"LF_{title}_skirt.pdf", bbox_inches='tight', dpi=300)
    plt.show()
