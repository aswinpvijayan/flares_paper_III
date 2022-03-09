"""

Figure A1 and A2 in paper for arguments 0 and 1 respectively.
Requires the SEDs of galaxies which is not provided.

"""

import numpy as np
import matplotlib, sys, h5py
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['legend.fancybox'] = False
import matplotlib.pyplot as plt
from scipy import integrate
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns
sns.set_context("paper")
plt.style.use('seaborn-whitegrid')


def get_data(loc):

    data = np.genfromtxt(loc)

    lam = data[:,0]
    att = data[:,1] * 1e-23 * 4 * np.pi * (1e6 * 3.086e18)**2
    unatt = data[:,2] * 1e-23 * 4 * np.pi * (1e6 * 3.086e18)**2

    nu = 3e8/(lam*1e-6)

    ok = np.where((lam>=3) & (lam<=1100))[0]
    lam_ir = lam[ok]
    Lnu_ir = att[ok]
    nu_ir = 3e8/(lam_ir*1e-6)
    Lir     = -integrate.simps(x=nu_ir, y=Lnu_ir, even='avg')/(3.826e33)

    ok      = np.where((lam>0.13) & (lam<0.17))[0]
    filt    = np.zeros(len(lam))
    filt[ok] = 1.
    LFUV    = np.trapz(att * filt, lam) / np.trapz(filt, lam)

    print (np.log10(Lir), np.log10(LFUV))

    return lam, nu, att



# SED from the FLARES master file
tag = '007_z008p000'
region = '00'
z=8

# with h5py.File(F'../../flares_pipeline/data/FLARES_{region}_sp_info.hdf5', 'r') as hf:
#     mstar       = np.array(hf[tag+'/Galaxy'].get('Mstar_30'), dtype=np.float64)*1e10
#     SED_att     = np.array(hf[tag+'/Galaxy/BPASS_2.2.1/Chabrier300/SED'].get('DustModelI'), dtype = np.float64)
#     SED_intr    = np.array(hf[tag+'/Galaxy/BPASS_2.2.1/Chabrier300/SED'].get('No_ISM'), dtype = np.float64)
#     lam_master  = np.array(hf[tag+'/Galaxy/BPASS_2.2.1/Chabrier300/SED'].get('Wavelength'), dtype = np.float64)
#
# req_ind = np.where(mstar>10**9.5)[0]

# nu_master = 3e8/(1e-10*lam_master)

# tmp1 = SED_att[req_ind[int(gal)]] # / (1e-23 * 4 * np.pi * (1e6 * 3.086e18)**2)
# tmp2 = SED_intr[req_ind[int(gal)]] # / (1e-23 * 4 * np.pi * (1e6 * 3.086e18)**2)

plt_option=int(sys.argv[1])

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=True, sharey=True, facecolor='w', edgecolor='k')

if plt_option==0:
    gal = '000'
    axins = zoomed_inset_axes(axs, 2, loc=3)

    loc = F'output_app/photons_1e6_default/z_{int(z)}/flares_{region}/gal_{gal}/flares_def1_sed.dat'
    x, xx, y = get_data(loc)
    axs.plot(np.log10(x), np.log10(xx*y), ls='solid', color='black', label='N=1e6, Fiducial', alpha=0.6)
    axins.plot(np.log10(x), np.log10(xx*y), ls='solid', color='black', alpha=0.6)

    loc = F'output_app/photons_1e6_NLTE/z_{int(z)}/flares_{region}/gal_{gal}/flares_def1_sed.dat'
    x, xx, y = get_data(loc)
    axs.plot(np.log10(x), np.log10(xx*y), ls='solid', color='red', label='N=1e6, NLTE', alpha=0.6)
    axins.plot(np.log10(x), np.log10(xx*y), ls='solid', color='red', alpha=0.6)

    loc = F'output_app/photons_1e7_NLTE/z_{int(z)}/flares_{region}/gal_{gal}/flares_def1_sed.dat'
    x, xx, y = get_data(loc)
    axs.plot(np.log10(x), np.log10(xx*y), ls='solid', color='orange', label='N=1e7, NLTE', alpha=0.6)
    axins.plot(np.log10(x), np.log10(xx*y), ls='solid', color='orange', alpha=0.6)

    loc = F'output_app/photons_1e7_high_NLTE/z_{int(z)}/flares_{region}/gal_{gal}/flares_def1_sed.dat'
    x, xx, y = get_data(loc)
    axs.plot(np.log10(x), np.log10(xx*y), ls='solid', color='blue', label=r'N=1e7, NLTE, Higher $\lambda$ resolution', alpha=0.6)
    axins.plot(np.log10(x), np.log10(xx*y), ls='solid', color='blue', alpha=0.6)

    x1, x2, y1, y2 = 0.5, 1.3, 43, 44
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    axins.grid(False)
    mark_inset(axs, axins, loc1=2, loc2=4, fc="none", ec="0.5")

else:
    gal = '001'
    loc = F'output_app/photons_1e6_default/z_{int(z)}/flares_{region}/gal_{gal}/flares_def1_sed.dat'
    x, xx, y = get_data(loc)
    axs.plot(np.log10(x), np.log10(xx*y), ls='solid', color='black', label='SMC, Fiducial', alpha=0.6)
    # axins.plot(np.log10(x), np.log10(xx*y), ls='solid', color='black', alpha=0.6)
    # print (x[np.argmax(y)], x[np.argmax(y*xx)])

    loc = F'output_app/photons_1e6_MW/z_{int(z)}/flares_{region}/gal_{gal}/flares_def1_sed.dat'
    x, xx, y = get_data(loc)
    axs.plot(np.log10(x), np.log10(xx*y), ls='solid', color='red', label='MW', alpha=0.6)
    # axins.plot(np.log10(x), np.log10(xx*y), ls='solid', color='brown', alpha=0.6)




axs.grid(True, ls='dotted')
axs.set_xlim(-1, 3)
axs.set_ylim(41, 45.6)
axs.legend(loc=2)

# axs.set_xlabel(r'log$_{10}$($\nu$(Hz))', fontsize=10)
axs.set_xlabel(r'log$_{10}$($\lambda$/$\mu$m)', fontsize=14)
axs.set_ylabel(r'log$_{10}$($\nu$L$_{\nu}$(erg/s))', fontsize=14)
axs.tick_params(axis="y",direction="in")
axs.tick_params(axis="x",direction="in")
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(13)

plt.savefig(F"sed_Lnu_app_conf_{plt_option+1}.pdf", bbox_inches='tight', dpi=300)
# plt.savefig(F"test.png", bbox_inches='tight', dpi=300)
plt.show()
