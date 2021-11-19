import sys, h5py, re, schwimmbad
import numpy as np
import pandas as pd
from functools import partial
from lmfit import Model
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def Planck_fn(nu, T):

    kb      = 1.3807e-23 #Boltzmann constant in SI units
    h       = 6.6261e-34 #Plancks constant in SI units
    c       = 3e8 #light speed m/s

    return (nu**3) / (np.exp(h*nu / (kb*T)) - 1)

def to_micron(nu):

    return 3e8/(nu*1e-6)

def GP_MBB_function(nu, beta, alpha, lambda0, T, A):

    # A, N, T = theta
    # beta    = 1.6
    kb      = 1.3807e-23 #Boltzmann constant in SI units
    h       = 6.6261e-34 #Plancks constant in SI units
    c       = 3e8
    nu0     = c/(lambda0*1e-6)
    tau     = (nu/nu0)**beta
    # alpha   = 2.
    b1      = 26.68
    b2      = 6.246
    b3      = 1.905e-4
    b4      = 7.243e-5
    L_alpha_T   = 1/((b1+b2*alpha)**(-2) + (b3+b4*alpha)*T)
    lambda_c    = (3/4)*L_alpha_T
    nu_c    = c/(lambda_c*1e-6)

    tau_c   = (nu_c/nu0)**beta

    return A + np.log10((1-np.exp(-tau)) * Planck_fn(nu, T) + (1-np.exp(-tau_c)) * ((nu_c)**3) * ((nu_c/nu)**(alpha)) * np.exp(-(nu_c/nu)**2) / (np.exp(h*nu_c/(kb*T))-1.))


def OT_MBB_function(nu, beta, T, A):

    return A + beta*np.log10(nu) + np.log10(Planck_fn(nu, T))

#Calulate the MBB temperature
def get_MBB_temp(nu, Lnu, z, type=0):


    if type==0:
        MBB_model = Model(GP_MBB_function)
        # MBB_model.set_param_hint('N', min=1e-10)
        A = np.mean(np.log10(Lnu) - GP_MBB_function(nu, 1.6, 2., 100, 50, 0))
        MBB_model.set_param_hint('alpha', min=1., max=4.5)


        MBB_model.set_param_hint('T', min=2.73*(1+z), max=150)
        MBB_model.set_param_hint('beta', min=0.5, max=3)
        MBB_model.set_param_hint('lambda0', min=50., max=300.001)



        result = MBB_model.fit(np.log10(Lnu), nu=nu, beta=2., alpha=2., lambda0=100., T=45, A=A)

    elif type==1:
        MBB_model = Model(OT_MBB_function)
        # MBB_model.set_param_hint('N', min=1e-10)
        A = np.log10(Lnu) - OT_MBB_function(nu, 1.6, 2., 0)
        A = np.mean(A[np.isfinite(A)])

        MBB_model.set_param_hint('T', min=2.73*(1+z), max=100)
        MBB_model.set_param_hint('beta', min=1.5, max=2.5)


        result = MBB_model.fit(np.log10(Lnu), nu=nu, beta=2., T=45, A=A)

        # print (result.conf_interval())


    return result.best_values['T'], result.params['T'].stderr, result


def get_MBB_fit(best_values, T, inp):

    if inp==0:
        MBB_model = partial(GP_MBB_function, beta=best_fit.best_values['beta'], A=best_fit.best_values['A'], N=best_fit.best_values['N'], T=T)
    else:
        MBB_model = partial(MBB_function, A=best_fit.best_values['A'], T=T)

    return MBB_model


def get_Mdust(Lnu, TMBB, nu, Knu, z):

    return Lnu/(4 * np.pi * Knu * (10**logPlanck_fn(nu, TMBB)))

def get_Tcorr(Tdust, beta, z):

    exp = 4.+beta
    Tcmb = 2.73
    Tcorr = (Tdust**exp - (Tcmb**exp)*((1+z)**exp - 1))**(1/exp)
    return Tcorr


def get_Lir_lpeak(data_loc):

    data_loc = F'{gal}/flares_def1_sed.dat'
    data = np.genfromtxt(data_loc)

    lam = data[:,0]
    att = data[:,1] * 1e-23 * 4 * np.pi * (1e6 * 3.086e18)**2

    ok = np.where(lam>=8)[0]
    lam = lam[ok]
    Lnu = att[ok]

    tmp_lam = np.linspace(8,1000,10000)
    tmp_Lnu = np.interp(tmp_lam, lam, data[:,1][ok])

    lambda_peak = tmp_lam[np.where(tmp_Lnu==np.max(tmp_Lnu))]

    nu = (3e8/(lam*1e-6)).astype(np.float64)

    Lir = -integrate.simps(x=nu, y=Lnu, even='avg')/(3.826e33)

    return Lir, lambda_peak


def get_props(gal, z, region, inp):

    data_loc = F'{gal}/flares_def1_sed.dat'
    data = np.genfromtxt(data_loc)

    lam = data[:,0]
    att = data[:,1] * 1e-23 * 4 * np.pi * (1e6 * 3.086e18)**2

    ok = np.where(lam>=8)[0]
    lam = lam[ok]
    Lnu = att[ok]

    tmp_lam = np.linspace(8,1000,10000)
    tmp_Lnu = np.interp(tmp_lam, lam, data[:,1][ok])

    lambda_peak = tmp_lam[np.where(tmp_Lnu==np.max(tmp_Lnu))]

    nu = (3e8/(lam*1e-6)).astype(np.float64)

    Lir = -integrate.simps(x=nu, y=Lnu, even='avg')/(3.826e33)

    # ok = np.where(lam>=20)[0]
    # lam = lam[ok]
    # Lnu = att[ok]
    # nu = nu[ok]


    if inp!=0:
        x = np.arange(20,1000,10)
        f = interp1d(lam, Lnu, kind='cubic')
        y = f(x)
        x_nu = (3e8/(1e-6*x)).astype(np.float64)

        T_MBB, T_MBBerr, best_fit = get_MBB_temp(x_nu, y, z, type=0)
        Lir_fit = -integrate.simps(x=nu, y=10**GP_MBB_function(nu, best_fit.best_values['beta'], best_fit.best_values['alpha'], best_fit.best_values['lambda0'], T_MBB, best_fit.best_values['A']), even='avg')/(3.826e33)


        # x = np.arange(20,1000,10)
        x = np.arange(20,1000,10)
        f = interp1d(lam, Lnu, kind='cubic')
        y = f(x)
        x_nu = (3e8/(1e-6*x)).astype(np.float64)

        T_MBB1, T_MBBerr1, best_fit1 = get_MBB_temp(x_nu, y, z, type=1)
        Lir_fit1 = -integrate.simps(x=nu, y=10**OT_MBB_function(nu, best_fit1.best_values['beta'], T_MBB1, best_fit1.best_values['A']), even='avg')/(3.826e33)


    # lam1    = np.linspace(845, 855, 1000)
    # lum1    = np.interp(lam1, lam, Lnu)
    # ok      = np.where((lam1>849.57) & (lam1<850.53))[0]
    # filt    = np.zeros(len(lam1))+4553321305
    # filt[ok] = 1.
    # L850    = np.trapz(lum1 * filt, lam1) / np.trapz(filt, lam1)
    #
    # loc = F'{gal}/flares_cnv_convergence.dat'
    # with open(loc, 'r') as file:
    #     data = file.readlines()
    # Mdust_inp = float(re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[+\-]?\ *[0-9]+)?', data[7])[0])
    #
    #
    # lambda_k = 850 #micron
    # k850 = 1.5
    # Mdust = get_Mdust(L850, T_MBB, 3e8/(lambda_k*1e-6), k850, z)/(1.988435e33)


    # return Mdust_inp, Mdust, Lir, L850, T_MBB, T_MBBerr, lambda_peak, best_fit.best_values['beta'], best_fit.best_values['A']

    if inp!=0:
        return Lir, lambda_peak, T_MBB, T_MBBerr, best_fit.best_values['beta'], best_fit.best_values['lambda0'], T_MBB1, T_MBBerr1, Lir_fit, Lir_fit1
    else:
        return Lir, lambda_peak
