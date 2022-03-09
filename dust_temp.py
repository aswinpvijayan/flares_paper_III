"""

Figure 9 and 10 in paper for arguments 0 and 1 respectively.

"""

import numpy as np
import pandas as pd
from functools import partial
import matplotlib, sys, h5py, re, schwimmbad
from uncertainties import unumpy
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['legend.fancybox'] = False
import matplotlib.pyplot as plt
from astropy.io import fits
from lmfit import Model
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import seaborn as sns
sns.set_context("paper")


from helpers import get_files
import flares

def plot_T_obs(axs, z, plt):
    #Faisst2020
    Faisst_Tpeak = np.array([52.4, 33.9, 38.9, 37.4])
    Faisst_Tpeakerr = np.array([[8.2, 5.9, 5.8, 4.9], [28.7, 9.9, 13.7, 8.]])
    Faisst_lampeak = 2.898e3/Faisst_Tpeak
    Faisst_lampeakerr = (Faisst_Tpeakerr/Faisst_Tpeak)*Faisst_lampeak
    Faisst_TMBB = np.array([57.3, 40.8, 49.4, 46.2])
    Faisst_TMBBerr = np.array([[16.6, 7.2, 10.7, 8.5], [67.1, 17.8, 29., 16.2]])
    Faisst_z = np.array([5.544, 5.293, 5.541, 5.657])
    Faisst_Lir = np.array([11.91, 11.73, 12.14, 12.49])
    Faisst_Lirerr = np.array([[0.91, 0.34, 0.45, 0.25], [0.37, 0.22, 0.21, 0.15]])

    #Strandet2016
    Strandet_z = np.array([4.2248, 4.2328, 4.2795, 4.2958, 4.304, 4.4357, 4.4771, 4.510, 4.5672, 4.757, 4.7677, 4.7993, 4.856, 5.2929, 5.576, 5.6559, 5.699, 5.811])
    Strandet_T = np.array([45.3, 31.3, 47.7, 50.2, 46.7, 37.4, 38.1, 39.9, 37.4, 57., 37.9, 38.1, 41.6, 42.1, 46.3, 50.5, 32.7, 53.5])
    Strandet_Terr = np.array([2.3, 1.4, 2.8, 2.8, 2.8, 1.6, 1.9, 2.1, 2.1, 4.2, 1.9, 1.9, 1.9, 2.1, 2.3, 2.3, 1.6, 2.8])
    Strandet_lampeak = np.array([83, 108, 80, 78, 82, 95, 94, 91, 95, 71, 94, 94, 88, 87, 82, 77, 103, 74])
    Strandet_Tpeak = 2.898e3/Strandet_lampeak

    #Jin2019
    Jin_z = np.array([5.847, 5.051, 4.440, 3.623])
    Jin_TMBB = np.array([61, 40, 42, 41])
    Jin_TMBBerr = np.array([8, 6, 6, 5])


    if plt=='z':
        axs.errorbar(Strandet_z, Strandet_Tpeak, color='violet', ls='None', marker='o', label='Strandet+2016', markersize=4, alpha=0.3)
        axs.errorbar(Strandet_z, Strandet_T, yerr=Strandet_Terr, color='violet', ls='None', marker='s', markersize=4, alpha=0.3)
        # axs.errorbar(Strandet_z, Strandet_T, yerr=Strandet_Terr, color='violet', ls='None', marker='s', markersize=4, alpha=0.3, label='Strandet+2016')

        axs.errorbar(Faisst_z, Faisst_Tpeak, yerr=Faisst_Tpeakerr, color='orange', ls='None', marker='o', label='Faisst+2020', markersize=4, alpha=0.3)
        axs.errorbar(Faisst_z, Faisst_TMBB, yerr=Faisst_TMBBerr, color='orange', ls='None', marker='D', markersize=4, alpha=0.3)
        # axs.errorbar(Faisst_z, Faisst_TMBB, yerr=Faisst_TMBBerr, color='orange', ls='None', marker='s', markersize=4, alpha=0.3, label='Faisst+2020')

        axs.errorbar(Jin_z, Jin_TMBB, yerr=Jin_TMBBerr, color='olive', ls='None', marker='D', label='Jin+2019', markersize=4, alpha=0.3)




        zs = np.array([6.02, 6.21, 7.17, 7.51])
        Tpeak = np.array([38.25, 25., 41.93, 39.43])
        yerr_low, yerr_up = np.array([26.36, 17.16, 36.48, 34.54]), np.array([72.42, 43.86, 54.77, 50.11])
        yerr_low = Tpeak - yerr_low
        yerr_up = yerr_up - Tpeak
        axs.errorbar(zs, Tpeak, yerr=[yerr_low, yerr_up], color='blue', ls='None', label='Faisst+2020 ($z>6$)', marker='o', markersize=4, alpha=0.3)

        axs.errorbar([7.152], [54.], yerr=[[6.], [7.]], color='indigo', ls='None', label='Hashimoto+2019', marker='s', markersize=4, alpha=0.3)

        axs.errorbar([6.03, 6.2], [38., 25.], yerr=[[12., 5.], [34., 19.]], color='green', ls='None', label='Harikane+2020', marker='s', markersize=4, alpha=0.3)

        axs.errorbar([8.3113], [80], yerr=[5.], lolims=[1], color='yellow', ls='None', label='Bakx+2020', marker='s', markersize=4, alpha=0.3)

        axs.errorbar([4.5, 5.5], [41., 43.], yerr=[1., 5.], color='limegreen', ls='None', label='Bethermin+2020 (Stacked)', marker='s', markersize=8, alpha=0.3)

    if plt=='Lir':
        axs.errorbar(Faisst_Lir, Faisst_lampeak, xerr=Faisst_Lirerr, yerr=Faisst_lampeakerr, color='orange', ls='None', marker='o', label='Faisst+2020 ($z \sim 5.5$)', markersize=4, alpha=0.7)

        #Casey2018 Fit
        lirbins = np.arange(8,14,0.5)
        lampeak = unumpy.uarray([102.8], [0.4]) * ((10**lirbins/1e12)**(unumpy.uarray([-0.068], [0.001])))
        axs.plot(lirbins, unumpy.nominal_values(lampeak), color='grey', label='Casey+2018 Fit', alpha=0.7, lw=2)
        axs.fill_between(lirbins, unumpy.nominal_values(lampeak)+unumpy.std_devs(lampeak), unumpy.nominal_values(lampeak)-unumpy.std_devs(lampeak), color='grey', alpha=0.2)




def plot_Tpeak_fit_z(axs):

    #Schreiber2017
    # z = np.arange(4.5, 11, 0.5)
    # T = unumpy.uarray([32.9], [2.4]) + unumpy.uarray([4.60], [0.35]) * (z - 2)
    # axs.plot(z, unumpy.nominal_values(T), ls='dashed', color='blue', label='Schreiber+2018')
    # axs.fill_between(z, unumpy.nominal_values(T)+unumpy.std_devs(T), unumpy.nominal_values(T)-unumpy.std_devs(T), color='blue', alpha=0.2)

    #Liang+2019
    z = np.arange(4., 6.5, 0.5)
    logT_25 = unumpy.uarray([-0.02], [0.06]) + unumpy.uarray([0.25], [0.09])*np.log10(1+z)
    T = 25 * 10**logT_25
    axs.plot(z, unumpy.nominal_values(T), ls='dashed', color='grey', label='Liang+2020')
    axs.fill_between(z, unumpy.nominal_values(T)+unumpy.std_devs(T), unumpy.nominal_values(T)-unumpy.std_devs(T), color='grey', alpha=0.2)

    #Bouwens2020
    z = np.arange(4.5, 8.5, 0.5)
    T = unumpy.uarray([34.6], [0.3]) + unumpy.uarray([3.94], [0.26])*(z - 2)
    axs.plot(z, unumpy.nominal_values(T), ls=(0, (1, 1)), color='violet', label='Bouwens+2020')
    axs.fill_between(z, unumpy.nominal_values(T)+unumpy.std_devs(T), unumpy.nominal_values(T)-unumpy.std_devs(T), color='violet', alpha=0.2)

    z = np.arange(8., 11.5, 0.5)
    T = unumpy.uarray([34.6], [0.3]) + unumpy.uarray([3.94], [0.26])*(z - 2)
    axs.plot(z, unumpy.nominal_values(T), ls=(0, (1, 1)), color='violet')

def Tpeak_Lir_fire(z, axs, ii):
    #Ma+2019
    LIR_fire = np.logspace(8., 12.6, 20)
    lam_peak_fire = 78.78 * (((1+z)/7)**-0.34) * (LIR_fire/1e10)**-0.084
    if z==5:
        axs.plot(np.log10(LIR_fire), lam_peak_fire, color=s_m.to_rgba(ii+0.5), label='FIRE-2', ls='dashed')
    else:
        axs.plot(np.log10(LIR_fire), lam_peak_fire, color=s_m.to_rgba(ii+0.5), ls='dashed')


def Tpeak_fit(z, alpha, beta):

    return alpha + beta*(z-5.)

def T_fit_exp(z, alpha, beta):

    return alpha*((1.+z)**beta)

def plt_median(x, y, bins, ws, ax, ii):

    df = pd.read_csv('weight_files/weights_grid.txt')
    weights = np.array(df['weights'])
    quantiles = [0.84,0.50,0.16]

    bincen = (bins[1:]+bins[:-1])/2.
    out = flares.binned_weighted_quantile(x, y, ws, bins, quantiles)
    hist, edges = np.histogram(x, bins)
    xx, yy, yy84, yy16 = bincen, out[:,1], out[:,0], out[:,2]
    ok = np.isfinite(yy)
    ok1 = np.where(hist[ok]>=1)[0]

    # print (out, xx)


    ax.plot(xx[ok][ok1], yy[ok][ok1], marker='o', color=s_m.to_rgba(ii+0.5), alpha=0.7)
    ax.fill_between(xx[ok][ok1], yy16[ok][ok1], yy84[ok][ok1], color=s_m.to_rgba(ii+0.5), alpha=0.2)


if __name__ == "__main__":

    fl = flares.flares('', sim_type='FLARES')

    tags = ['010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']
    N = np.array([3844., 1772., 773., 331., 126., 44.])

    inp = int(sys.argv[1])

    df = pd.read_csv('weight_files/weights_grid.txt')
    weights = np.array(df['weights'])
    quantiles = [0.84,0.50,0.16]
    zs = np.array([])


    if inp==0:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(13, 5), sharex=False, sharey=True, facecolor='w', edgecolor='k')
        axs = axs.ravel()
        savename = 'lampeak_properties'

        # choose a colormap
        c_m = matplotlib.cm.viridis_r
        norm = matplotlib.colors.BoundaryNorm(np.arange(0.,len(tags)+1,1), c_m.N)
        # create a ScalarMappable and initialize a data structure
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])

    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=True, sharey=False, facecolor='w', edgecolor='k')
        savename = 'Tdust_z_variation'

        norm = matplotlib.colors.Normalize(vmin=0.5, vmax=len(tags)+0.5)
        # choose a colormap
        c_m = matplotlib.cm.viridis_r
        # create a ScalarMappable and initialize a data structure
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])

        med_TSED, med_Tpeak, med_TSEDRJ = np.zeros((6,3)), np.zeros((6,3)), np.zeros((6,3))
        Tpeakerr, TSEDerr, TSEDRJerr = np.zeros(6), np.zeros(6), np.zeros(6)



    for ii, tag in enumerate(tags):

        mstars, SFR, ws, Lir, TSED, lambda_peak, TSEDRJ = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        z = float(tag[5:].replace('p','.'))

        for jj, mm in enumerate(fl.halos):

            with h5py.File(F'./data/flares_skirt_outputs.hdf5', 'r') as hf:
                mstar       = np.array(hf[mm+'/'+tag+'/Galaxy'].get('Mstar'), dtype=np.float64)*1e10
                SFR        = np.append(SFR, np.array(hf[mm+'/'+tag+'/Galaxy'].get('SFR'), dtype=np.float64))

                mstars    = np.append(mstars, mstar)
                ws        = np.append(ws, np.ones(len(mstar))*weights[int(mm)])

                Lir       = np.append(Lir, np.array(hf[mm+'/'+tag+'/Galaxy/Photometry'].get('Lir'), dtype=np.float64))

                lambda_peak= np.append(lambda_peak, np.array(hf[mm+'/'+tag+'/Galaxy/Dust_Temperature'].get('Lambda_peak'), dtype=np.float64))
                TSED      = np.append(TSED, np.array(hf[mm+'/'+tag+'/Galaxy/Dust_Temperature'].get('TSED'), dtype=np.float64))
                TSEDRJ   = np.append(TSEDRJ, np.array(hf[mm+'/'+tag+'/Galaxy/Dust_Temperature'].get('TSEDRJ'), dtype=np.float64))


            Tpeak = 2.898e3/(lambda_peak)


        if inp==0:

            bins = np.arange(9, 12.5, 0.3)
            plt_median(np.log10(mstars), lambda_peak, bins, ws, axs[0], ii)


            bins = np.arange(9, 13.5, 0.25)
            plt_median(np.log10(Lir), lambda_peak, bins, ws, axs[1], ii)

            bins = np.arange(-2.5, 2., 0.25)
            plt_median(np.log10(1e9*SFR/mstars), lambda_peak, bins, ws, axs[2], ii)

        else:

            # print (np.median(betas), np.percentile(betas,84), np.percentile(betas,16))
            Tpeakerr[ii] = 1.253*np.std(Tpeak)/np.sqrt(N[ii])
            TSEDerr[ii] = 1.253*np.std(TSED)/np.sqrt(N[ii])
            TSEDRJerr[ii] = 1.253*np.std(TSEDRJ)/np.sqrt(N[ii])


            zs = np.append(zs, z)
            print ("z:", z)
            print ("Tpeak: ", np.nanmin(Tpeak), np.nanmax(Tpeak))
            print ("TSED: ", np.nanmin(TSED), np.nanmax(TSED))
            print ("TSED,RJ: ", np.nanmin(TSEDRJ), np.nanmax(TSEDRJ))

            out = flares.binned_weighted_quantile(np.ones(len(TSED))*z, TSED, ws, [z-1,z+1], quantiles)
            med_TSED[ii][0], med_TSED[ii][1], med_TSED[ii][2] = out[1], out[1]-out[2], out[0]-out[1]

            out = flares.binned_weighted_quantile(np.ones(len(TSEDRJ))*z, TSEDRJ, ws, [z-1,z+1], quantiles)
            med_TSEDRJ[ii][0], med_TSEDRJ[ii][1], med_TSEDRJ[ii][2] = out[1], out[1]-out[2], out[0]-out[1]

            out = flares.binned_weighted_quantile(np.ones(len(Tpeak))*z, Tpeak, ws, [z-1,z+1], quantiles)
            med_Tpeak[ii][0], med_Tpeak[ii][1], med_Tpeak[ii][2] = out[1], out[1]-out[2], out[0]-out[1]

            # print (np.median(Lir/Lir_fit), np.percentile(Lir/Lir_fit, 84), np.percentile(Lir/Lir_fit, 16), np.median(Lir/Lir_fitRJ), np.percentile(Lir/Lir_fitRJ, 84), np.percentile(Lir/Lir_fitRJ, 16))




    if inp!=0:

        axs.errorbar(zs, med_TSED[:,0], yerr=TSEDerr, color='grey', alpha=0.25, marker='D', label='\\textsc{Flares}: ' + r'T$_{\mathrm{SED}}$')
        axs.fill_between(zs, med_TSED[:,0]-med_TSED[:,1], med_TSED[:,0]+med_TSED[:,2], color='grey', alpha=0.1)


        axs.errorbar(zs, med_TSEDRJ[:,0], yerr=TSEDRJerr, color='brown', marker='s', label='\\textsc{Flares}: ' + r'T$_{\mathrm{SED,RJ}}$')
        axs.fill_between(zs, med_TSEDRJ[:,0]-med_TSEDRJ[:,1], med_TSEDRJ[:,0]+med_TSEDRJ[:,2], color='brown', alpha=0.2)


        axs.errorbar(zs, med_Tpeak[:,0], yerr=Tpeakerr, color='red', marker='o', label='\\textsc{Flares}: ' + r'T$_{\mathrm{peak}}$')
        axs.fill_between(zs, med_Tpeak[:,0]-med_Tpeak[:,1], med_Tpeak[:,0]+med_Tpeak[:,2], color='red', alpha=0.2)

        popt, pcov = curve_fit(Tpeak_fit, zs, med_Tpeak[:,0], p0=[33, 4], sigma=Tpeakerr)

        print ('Normal fit to Tpeak:', popt, pcov)


        popt, pcov = curve_fit(T_fit_exp, zs, med_TSEDRJ[:,0], p0=[1.5, 0.5], sigma=TSEDRJerr)

        print ('Exponential fit to TSED,RJ:', popt, pcov)

        plot_T_obs(axs, z, 'z')
        plot_Tpeak_fit_z(axs)

        # axs.plot(zs, 2.73*(1+zs), color='black', ls='dashed', label = r'T$_{\mathrm{CMB}}$')

        axs.grid(True, linestyle=(0, (0.5, 3)))
        axs.legend(frameon=False, fontsize=9, loc=2, numpoints=1, markerscale=1, ncol=2)
        axs.set_xlabel('z', fontsize=14)
        axs.set_ylabel(r'T$_{\mathrm{dust}}$', fontsize=14)
        axs.set_xlim(4.5,10.5)
        axs.set_ylim(25,130)
        axs.set_yticks(np.arange(30,140,10))
        axs.tick_params(axis="y",direction="in")
        axs.tick_params(axis="x",direction="in")
        for label in (axs.get_xticklabels() + axs.get_yticklabels()):
            label.set_fontsize(12)

    else:
        Sch18_lampeak = 2.898e3/41.8
        Sch18_lampeaklo = Sch18_lampeak - 2.898e3/(41.8+1.6)
        Sch18_lampeakup = 2.898e3/(41.8-1.6) - Sch18_lampeak
        axs[0].errorbar([11.25], [Sch18_lampeak], xerr = [0.25], yerr=[[Sch18_lampeaklo], [Sch18_lampeakup]], marker='s', color='violet', label='Schreiber+2018 ($3.5<z<5$)', markersize=4)

        axs[1].errorbar(np.log10([357e10]), [Sch18_lampeak], xerr = [np.log10([357e10])-np.log10([(357-61) * 10**10]), np.log10([(357+37) * 10**10]) - np.log10([357e10])], yerr=[[Sch18_lampeaklo], [Sch18_lampeakup]], marker='s', color='violet', label='Schreiber+2018 ($3.5<z<5$)', markersize=4)

        plot_T_obs(axs[1], z, 'Lir')
        Tpeak_Lir_fire(5, axs[1], 0)
        Tpeak_Lir_fire(10, axs[1], 5)

        #Burnham+2021
        # B21_Lir = np.array([3.28e13, 1.64e13, 1.63e13])
        # tmp = np.array([0.68e13, 0.25e13, 0.26e13])
        # B21_Lirerr =  [np.log10(B21_Lir) - np.log10(B21_Lir-tmp), np.log10(B21_Lir+tmp) - np.log10(B21_Lir)]
        # B21_Lir = np.log10(B21_Lir)
        # B21_lampeak = np.array([71.4, 69, 71])
        # B21_lampeakerr = np.array([7.8, 9, 9])
        B21_lampeak = np.array([69, 71])
        B21_lampeakerr = np.array([9, 9])
        B21_SFR = unumpy.uarray([2460., 2430.], [380., 390.])
        B21_Mstar = unumpy.uarray([2.3e11, 5.6e11], [1.6e11, 2.7e11])
        B21_sSFR = unumpy.log10(1e9 * B21_SFR/B21_Mstar)

        axs[2].errorbar(unumpy.nominal_values(B21_sSFR), B21_lampeak, xerr=unumpy.std_devs(B21_sSFR), yerr=B21_lampeakerr, color='olive', marker = 's', ls = 'None', label = 'Burnham+2021 ($z\sim4.5$)')

        #lampeak ~ sSFR^6
        xx = np.arange(-2.6, 2., 0.25)
        axs[2].plot(xx, 10**(np.log10(75.) - (1./6.)*xx), ls='dashed', color='grey', lw=2, label=r'T$_{\mathrm{peak}}\propto$ sSFR$^{1/6}$')




        axs[0].set_xlabel(r'log$_{10}$(M$_{\star}$/M$_{\odot}$)', fontsize=14)
        axs[1].set_xlabel(r'log$_{10}$(L$_{\mathrm{IR}}$/L$_{\odot}$)', fontsize=14)
        axs[2].set_xlabel(r'log$_{10}$(sSFR/Gyr$^{-1}$)', fontsize=14)

        axs[0].set_ylabel(r'$\lambda_{\mathrm{peak}}$/$\mu$m', fontsize=14)
        axs[0].set_xlim(9.1, 11.9)

        axs[1].set_xlim(8.9,12.9)
        axs[2].set_xlim(-2.1,1.5)


        for ax in axs:
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


        twinax = axs[-1].twinx()
        twinax.set_ylim(2.898e3/30, 2.898e3/130)
        twinax.tick_params(axis="y",direction="in")
        twinax.set_ylabel(r'T$_{\mathrm{peak}}$/K', fontsize=14)
        twinax.grid(False)
        for label in (twinax.get_yticklabels()):
            label.set_fontsize(13)


        fig.subplots_adjust(wspace=0, hspace=0, right=0.89)

        cbaxes = fig.add_axes([0.94, 0.35, 0.01, 0.3])
        cbar = fig.colorbar(s_m, cax=cbaxes)
        cbar.set_label(r'$z$', fontsize = 15)
        cbar.set_ticks(np.arange(len(tags))+0.5)
        cbar.set_ticklabels(np.arange(5,11))
        cbaxes.invert_yaxis()
        for label in cbaxes.get_yticklabels():
            label.set_fontsize(13)


    # for ii in range(2):
    #     # axs[ii].set_xlabel(r'$\lambda_{\mathrm{peak}}/\mu$m')
    #     axs[ii].grid(True)
    #     # axs[ii].legend(frameon=False, fontsize=10)



    plt.savefig(F"{savename}.pdf", bbox_inches='tight', dpi=300)
    plt.show()
