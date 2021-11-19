import numpy as np
import pandas as pd
from FLARE.photom import M_to_lum


def rem_inf(low, up):

    ok = np.isfinite(low)
    low[~ok] = 10.
    ok = np.isfinite(up)
    up[~ok] = 10.

    return low, up

def plot_UVLF_Fink15(z, axs):

    data = np.genfromtxt(f"./Obs_data/uv_lum_Fink15_z{z}.txt", delimiter=',', comments='#')
    M, phi, phi_up, phi_low, uplims = data[:,0], data[:,1]*1e-3, data[:,2]*1e-3, data[:,3]*1e-3, data[:,4]
    phi_up = np.log10(phi + phi_up) - np.log10(phi)
    phi_low = np.log10(phi) - np.log10(phi - phi_low)

    phi_low, phi_up = rem_inf(phi_low, phi_up)

    axs.errorbar(M, np.log10(phi), yerr=[phi_low, phi_up], uplims=uplims, label='Finkelstein+2015', ls='None', fmt='s', color='grey', alpha=0.8)


def plot_UVLF_Bouw15(z, axs):

    data = np.genfromtxt(f"./Obs_data/uv_lum_Bouw15_z{z}.txt", delimiter=',', comments='#')
    M, phi, phi_err, uplims = data[:,0], data[:,1], data[:,2], data[:,3]
    phi_up = np.log10(phi + phi_err) - np.log10(phi)
    phi_low = np.log10(phi) - np.log10(phi - phi_err)

    phi_low, phi_up = rem_inf(phi_low, phi_up)

    axs.errorbar(M, np.log10(phi), yerr=[phi_low, phi_up], uplims=uplims, label='Bouwens+2015', ls='None', fmt='D', color='black', alpha = 0.4)

def plot_UVLF_Bouw16(z, axs):

    data = np.genfromtxt(f"./Obs_data/uv_lum_Bouw16_z{z}.txt", delimiter=',', comments='#')
    M, phi, phi_up, phi_low, uplims = data[:,0], data[:,1]*1e-3, data[:,2]*1e-3, data[:,3]*1e-3, data[:,4]
    phi_up = np.log10(phi + phi_up) - np.log10(phi)
    phi_low = np.log10(phi) - np.log10(phi - phi_low)

    phi_low, phi_up = rem_inf(phi_low, phi_up)

    axs.errorbar(M, np.log10(phi), yerr=[phi_low, phi_up], uplims=uplims, label='Bouwens+2016', ls='None', fmt='^', color='black', alpha=0.4, markersize=7)

def plot_UVLF_Bouw17(z, axs):

    data = np.genfromtxt(f"./Obs_data/uv_lum_Bouw17_z{z}.txt", delimiter=',', comments='#')
    M, phi, phi_up, phi_low, uplims = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
    phi_up = np.log10(phi + phi_up) - np.log10(phi)
    phi_low = np.log10(phi) - np.log10(phi - phi_low)

    phi_low, phi_up = rem_inf(phi_low, phi_up)

    axs.errorbar(M, np.log10(phi), yerr=[phi_low, phi_up], uplims=uplims, label='Bouwens+2017', ls='None', fmt='8', color='black', alpha=0.4)

def plot_UVLF_Stefanon19(z, axs):

    data = np.genfromtxt(f"./Obs_data/uv_lum_stefanon19_z{z}.txt", delimiter=',', comments='#')
    M, phi, phi_up, phi_low = data[:,0], data[:,1]*1e-6, data[:,2]*1e-6, data[:,3]*1e-6
    phi_up = np.log10(phi + phi_up) - np.log10(phi)
    phi_low = np.log10(phi) - np.log10(phi - phi_low)

    phi_low, phi_up = rem_inf(phi_low, phi_up)

    axs.errorbar(M, np.log10(phi), yerr=[phi_low, phi_up], label='Stefanon+2019', ls='None', fmt='p', color='black', alpha=0.4, markersize=7)

def plot_UVLF_Bowler20(z, axs):

    data = np.genfromtxt(f"./Obs_data/uv_lum_Bowler20_z{z}.txt", delimiter=',', comments='#')
    M, M_err, phi, phi_err, uplims = data[:,0], data[:,1], data[:,2]*1e-6, data[:,3]*1e-6, data[:,4]
    phi_up = np.log10(phi + phi_err) - np.log10(phi)
    phi_low = np.log10(phi) - np.log10(phi - phi_err)

    phi_low, phi_up = rem_inf(phi_low, phi_up)

    axs.errorbar(M, np.log10(phi), xerr=M_err, yerr=[phi_low, phi_up], uplims=uplims, label='Bowler+2020', ls='None', fmt='D', color='grey', alpha=0.8, markersize=7)

def plot_UVLF_Bouw21(z, axs):

    data = np.genfromtxt(f"./Obs_data/uv_lum_Bouw21_z{z}.txt", delimiter=',', comments='#')
    M, phi, phi_err, uplims = data[:,0], data[:,1], data[:,2], data[:,3]
    phi_up = np.log10(phi + phi_err) - np.log10(phi)
    phi_low = np.log10(phi) - np.log10(phi - phi_err)

    phi_low, phi_up = rem_inf(phi_low, phi_up)

    axs.errorbar(M, np.log10(phi), yerr=[phi_low, phi_up], uplims=uplims, label='Bouwens+2021', ls='None', fmt='D', color='black', alpha = 0.4)

def plot_UVLF(z, axs):


    if z == 5:

        plot_UVLF_Fink15(z, axs)
        #plot_UVLF_Bouw15(z, axs)
        plot_UVLF_Bouw21(z, axs)

    if z == 6:

        plot_UVLF_Fink15(z, axs)
        #plot_UVLF_Bouw15(z, axs)
        #plot_UVLF_Bouw17(z, axs)
        plot_UVLF_Bouw21(z, axs)

        data = np.genfromtxt("./Obs_data/uv_lum_Atek18_z6.txt", delimiter=',', comments='#')
        M, phi, phi_err = data[:,0], data[:,1], data[:,2]
        axs.errorbar(M, phi, yerr=phi_err, label='Atek+2018', ls='None', fmt='<', color='grey', alpha=0.8, markersize=7)

    if z == 7:

        plot_UVLF_Fink15(z, axs)
        #plot_UVLF_Bouw15(z, axs)
        plot_UVLF_Bouw21(z, axs)

    if z == 8:

        plot_UVLF_Fink15(z, axs)
        #plot_UVLF_Bouw15(z, axs)
        plot_UVLF_Stefanon19(z, axs)
        plot_UVLF_Bowler20(z, axs)
        plot_UVLF_Bouw21(z, axs)

    if z == 9:

        data = np.genfromtxt(f"./Obs_data/uv_lum_McLeod15_z{z}.txt", delimiter=',', comments='#')
        M, phi, phi_up, phi_low, M_up, M_low, uplims = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6]
        phi_up, phi_low = np.absolute(phi_up-phi), np.absolute(phi-phi_low)
        M_up, M_low = np.absolute(M_up-M), np.absolute(M-M_low)

        axs.errorbar(M, phi, xerr=[M_low, M_up], yerr=[phi_low, phi_up], label='McLeod+2015', ls='None', fmt='>', color='grey', alpha=0.8, markersize=7)

        plot_UVLF_Bouw16(z, axs)
        plot_UVLF_Stefanon19(z, axs)
        plot_UVLF_Bowler20(z, axs)
        plot_UVLF_Bouw21(z, axs)

    if z == 10:

        plot_UVLF_Bouw15(z, axs)
        plot_UVLF_Bouw16(z, axs)

        data = np.genfromtxt(f"./Obs_data/uv_lum_Oesch18_z{z}.txt", delimiter=',', comments='#')
        M, phi, phi_up, phi_low, uplims = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
        phi_up = np.log10(phi + phi_up) - np.log10(phi)
        phi_low = np.log10(phi) - np.log10(phi - phi_low)

        phi_low, phi_up = rem_inf(phi_low, phi_up)

        axs.errorbar(M, np.log10(phi), yerr=[phi_low, phi_up], uplims=uplims, label='Oesch+2018', ls='None', fmt='H', color = 'black', alpha = 0.5, markersize=6)



def plot_beta_Bouw12(z, axs):

    data = np.genfromtxt(f"./Obs_data/uv_beta_Bouw12_z{z}.txt", delimiter=',', skip_header=1)
    M, beta, ran_err, sys_err = data[:,0], data[:,1], data[:,2], data[:,3]
    ok = np.where(sys_err > ran_err)
    ran_err[ok] = sys_err[ok]

    axs.errorbar(M, beta, yerr=ran_err, label='Bouwens+2012', ls='None', fmt='s', color='blue', alpha=0.5)

def plot_beta_Bouw14(z, axs):

    data = np.genfromtxt(f"./Obs_data/uv_beta_Bouw14_z{z}.txt", delimiter=',', skip_header=1)
    M, beta, ran_err, sys_err = data[:,0], data[:,1], data[:,2], data[:,3]
    ok = np.where(sys_err > ran_err)
    ran_err[ok] = sys_err[ok]

    axs.errorbar(M, beta, yerr=ran_err, label='Bouwens+2014', ls='None', fmt='D', color='blue', alpha=0.5)

def plot_beta_Dunlop12(z, axs):

    data = np.genfromtxt(f"./Obs_data/uv_beta_Dunlop12_z{z}.txt", delimiter=',', skip_header=1)
    M, beta, beta_err = data[:,0], data[:,1], data[:,2]


    axs.errorbar(M, beta, yerr=beta_err, label='Dunlop+2012', ls='None', fmt='s', color='green', alpha=0.5)


def plot_beta(z, axs):


    if (z == 5) or (z == 6) or (z == 7):
        plot_beta_Bouw12(z, axs)
        plot_beta_Bouw14(z, axs)
        plot_beta_Dunlop12(z, axs)

    if z == 8:
        plot_beta_Bouw14(z, axs)
        plot_beta_Dunlop12(z, axs)


def plot_OIIIEW_Mstar(z, axs, errorbar=True):

    bins = np.arange(7.25, 11.5, 0.5)
    bincen = (bins[1:]+bins[:-1])/2.
    binwidth = bins[1:] - bins[:-1]

    if z == 7:
        df = pd.read_csv('Obs_data/Endsley_2020.csv')
        Mstar, maxMstar, minMstar = np.array(df['log10Mstar']), np.nanmedian(df['log10Mstar_up']), np.nanmedian(df['log10Mstar_low'])
        EW = np.array(df['oiiihbeta'])
        maxEW = np.log10(EW+np.array(df['oiiihbeta_up']))
        minEW = np.log10(EW-np.array(df['oiiihbeta_low']))
        logEW = np.log10(EW)
        maxEW, minEW = np.nanmedian(maxEW-logEW), np.nanmedian(logEW-minEW)

        label = 'Endsley+2020'
        c = 'orange'

    elif z == 8:
        obs = np.loadtxt('Obs_data/z8info_sw.txt').T
        EW = obs[6] + obs[10] + obs[14]
        logEW = np.log10(EW)
        uperr = np.sqrt((obs[7]-obs[6])**2 + (obs[11]-obs[10])**2 + (obs[15]-obs[14])**2)
        loerr = np.sqrt((obs[6]-obs[8])**2 + (obs[12]-obs[10])**2 + (obs[16]-obs[14])**2)
        maxEW = np.nanmedian(np.log10(EW+uperr) - np.log10(EW))
        minEW = np.nanmedian(np.log10(EW) - np.log10(EW-loerr))

        Mstar, maxMstar, minMstar = np.log10(obs[2]), np.log10(obs[4]), np.log10(obs[3])
        maxMstar, minMstar = np.nanmedian(maxMstar-Mstar), np.nanmedian(Mstar-minMstar)

        label = 'de Barros+2019'
        c = 'indianred'


    axs.scatter(Mstar, logEW, s=5, c=c, label = F'{label} individual')
    xx, yy, yy16, yy84 = np.array([]), np.array([]), np.array([]), np.array([])
    for jj in range(len(bincen)):
        thisok = np.logical_and(Mstar >= bins[jj], Mstar < bins[jj+1])
        if np.sum(thisok) > 1:

            xx = np.append(xx, bincen[jj])
            yy = np.append(yy, np.median(logEW[thisok]))
            yy84 = np.append(yy84, np.percentile(logEW[thisok], 84))
            yy16 = np.append(yy16, np.percentile(logEW[thisok], 16))

    axs.errorbar(xx, yy, yerr=[yy-yy16, yy84-yy], c=c, ls='None', fmt='o', markeredgecolor='k', label=F'{label} binned')
    if errorbar:
        axs.errorbar(10.8, 2.3, xerr=[[minMstar], [maxMstar]], yerr=[[minEW], [maxEW]], c=c, ls='None', fmt='x', markeredgecolor='k')


def plot_OIIIEW_UV(z, axs, errorbar=True):

    bins = np.arange(40, 47, 0.4)
    bincen = (bins[1:]+bins[:-1])/2.
    binwidth = bins[1:] - bins[:-1]

    if z == 7:
        df = pd.read_csv('Obs_data/Endsley_2020.csv')
        conversion_fac = 2E15  #converting from ergs/s/Hz to ergs/s at FUV
        LUV = np.log10(np.array(M_to_lum(df['MUV']))*conversion_fac)
        minLUV, maxLUV = np.array(M_to_lum(df['MUV']+df['MUV_up']))*conversion_fac,  np.array(M_to_lum(df['MUV']-df['MUV_low']))*conversion_fac

        minLUV, maxLUV = np.nanmedian(LUV-np.log10(minLUV)), np.nanmedian(np.log10(maxLUV)-LUV)

        EW = np.array(df['oiiihbeta'])
        maxEW = np.log10(EW+np.array(df['oiiihbeta_up']))
        minEW = np.log10(EW-np.array(df['oiiihbeta_low']))
        logEW = np.log10(EW)
        maxEW, minEW = np.nanmedian(maxEW-logEW), np.nanmedian(logEW-minEW)

        label = 'Endsley+2020'
        c = 'orange'


    elif z == 8:
        obs = np.loadtxt('Obs_data/z8info_sw.txt').T
        id, LUV, low, up = np.loadtxt('Obs_data/Loiii.txt', usecols = (0,5,6,7), unpack = True)
        tmp, ind1, ind2 = np.intersect1d(obs[0], id, assume_unique=True, return_indices=True)

        EW = obs[6] + obs[10] + obs[14]
        logEW = np.log10(EW)[ind1]
        uperr = np.sqrt((obs[7]-obs[6])**2 + (obs[11]-obs[10])**2 + (obs[15]-obs[14])**2)
        loerr = np.sqrt((obs[6]-obs[8])**2 + (obs[12]-obs[10])**2 + (obs[16]-obs[14])**2)
        maxEW = np.nanmedian(np.log10(EW+uperr) - np.log10(EW))
        minEW = np.nanmedian(np.log10(EW) - np.log10(EW-loerr))



        Lsol = 3.8E26 # W
        LUV = LUV[ind2] + np.log10(Lsol) + 7.
        low = low[ind2] + np.log10(Lsol) + 7.
        up = up[ind2] + np.log10(Lsol) + 7.
        maxLUV = np.nanmedian(up-LUV)
        minLUV = np.nanmedian(LUV-low)

        label = 'de Barros+2019'
        c = 'indianred'


    axs.scatter(LUV, logEW, s=5, c=c, label = F'{label} individual')
    xx, yy, yy16, yy84 = np.array([]), np.array([]), np.array([]), np.array([])
    for jj in range(len(bincen)):
        thisok = np.logical_and(LUV >= bins[jj], LUV < bins[jj+1])
        if np.sum(thisok) > 1:

            xx = np.append(xx, bincen[jj])
            yy = np.append(yy, np.median(logEW[thisok]))
            yy84 = np.append(yy84, np.percentile(logEW[thisok], 84))
            yy16 = np.append(yy16, np.percentile(logEW[thisok], 16))

    axs.errorbar(xx, yy, yerr=[yy-yy16, yy84-yy], c=c, ls='None', fmt='o', markeredgecolor='k', label=F'{label} binned')
    if errorbar:
        axs.errorbar(45.5, 2.5, xerr=[[minLUV], [maxLUV]], yerr=[[minEW], [maxEW]], c=c, ls='None', fmt='x', markeredgecolor='k')


def plot_OIIIlum_UV(z, axs, errorbar=True):

    c = 'indianred'
    bins = np.arange(40, 45.2, 0.4)
    bincen = (bins[1:]+bins[:-1])/2.
    binwidth = bins[1:] - bins[:-1]
    if z == 8:
        Llines, Llines_low, Llines_up, LUV, low, up = np.loadtxt('Obs_data/Loiii.txt', usecols = (2, 3, 4, 5, 6, 7), unpack = True)
        Lsol = 3.8E26 # W
        LUV = LUV + np.log10(Lsol) + 7.
        low = low + np.log10(Lsol) + 7.
        up = up + np.log10(Lsol) + 7.
        maxLUV = np.nanmedian(up-LUV)
        minLUV = np.nanmedian(LUV-low)

        Llines = Llines + np.log10(Lsol) + 7.
        Llines_low = Llines_low + np.log10(Lsol) + 7.
        Llines_up = Llines_up + np.log10(Lsol) + 7.
        R = Llines - LUV
        maxR = np.nanmedian(np.sqrt((Llines_up-Llines)**2 + (up-LUV)**2))
        minR = np.nanmedian(np.sqrt((Llines-Llines_low)**2 + (LUV-low)**2))

        axs.scatter(LUV, R, s=5, c=c, label = 'de Barros+2019 individual')
        xx, yy, yy16, yy84 = np.array([]), np.array([]), np.array([]), np.array([])
        for jj in range(len(bincen)):
            thisok = np.logical_and(LUV >= bins[jj], LUV < bins[jj+1])
            if np.sum(thisok) > 1:

                xx = np.append(xx, bincen[jj])
                yy = np.append(yy, np.median(R[thisok]))
                yy84 = np.append(yy84, np.percentile(R[thisok], 84))
                yy16 = np.append(yy16, np.percentile(R[thisok], 16))

        axs.errorbar(xx, yy, yerr=[yy-yy16, yy84-yy], c=c, ls='None', fmt='o', markeredgecolor='k', label='de Barros 2019+binned')
        if errorbar:
            axs.errorbar(45.3, -2, xerr=[[minLUV], [maxLUV]], yerr=[[minR], [maxR]], c=c, ls='None', fmt='x', markeredgecolor='k')
