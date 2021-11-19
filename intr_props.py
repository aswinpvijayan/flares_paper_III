"""

Figure 1 in paper

"""

import numpy as np
from functools import partial
import schwimmbad, h5py, matplotlib, timeit, sys
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")
# plt.style.use("classic")

def get_data(ii, tag, limit=1000):

    region = str(ii)
    if len(region) == 1:
        region = '0'+region

    try:
        with h5py.File(F'./data/flares_skirt_outputs.hdf5', 'r') as hf:
            mstar       = np.array(hf[region+'/'+tag+'/Galaxy'].get('Mstar'), dtype=np.float64)*1e10
            SFR        = np.array(hf[region+'/'+tag+'/Galaxy'].get('SFR'), dtype=np.float64)

    except:
        mstar, SFR = np.array([]), np.array([])
        print (F"No data in region {mm} for tag {tag}")

    return mstar, SFR

def get_hist(x, bins):

    hist, edges = np.histogram(x, bins = bins)
    left,right = edges[:-1],edges[1:]
    X = np.array([left,right]).T.flatten()
    Y = np.log10(np.array([hist,hist]).T.flatten())
    Y[Y<0] = -1

    return X, Y



if __name__ == "__main__":

    # inp = int(sys.argv[1])

    # choose a colormap
    c_m = matplotlib.cm.viridis_r
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5,6,1), c_m.N)
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    tags = ['010_z005p000','009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000']

    # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), sharex=False, sharey=False, facecolor='w', edgecolor='k')

    fig     = plt.figure(figsize = (5, 4))
    ax      = fig.add_axes((0.1, 0.1, 0.6, 0.6))
    ax_R    = fig.add_axes((0.7, 0.1, 0.3, 0.6))
    ax_T    = fig.add_axes((0.1, 0.7, 0.6, 0.35))


    for ii, tag in enumerate(tags):

        z = float(tag[5:].replace('p','.'))

        func = partial(get_data, tag=tag, limit=1000)
        pool = schwimmbad.MultiPool(processes=8)
        dat = np.array(list(pool.map(func, np.arange(0,40))))
        pool.close()

        mstars = np.concatenate(dat[:,0])
        SFRs = np.concatenate(dat[:,1])


        x = np.log10(mstars)
        y = np.log10(SFRs)
        xlabel = r'log$_{10}$(M$_{\star}$/M$_{\odot}$)'
        ylabel = r'log$_{10}$(SFR/(M$_{\odot}$yr$^{-1}$))'
        savename = 'Mstar_sfr.pdf'

        ax.scatter(x, y, s=2, alpha=0.6, color=s_m.to_rgba(ii), label = rF'z={z} ({len(x)})')

        bins = np.arange(9., 12.5, 0.5)
        X, Y = get_hist(x, bins = bins)
        ax_T.plot(X, Y, lw = 3, color=s_m.to_rgba(ii))

        bins = np.arange(-1, 4, 0.5)
        X, Y = get_hist(y, bins = bins)
        ax_R.plot(Y, X, lw = 3, color=s_m.to_rgba(ii))


    for axs in [ax, ax_R, ax_T]:
        axs.grid(True, ls='dotted')
        for label in (axs.get_xticklabels() + axs.get_yticklabels()):
            label.set_fontsize(11)


    ax.set_xlim([9., 11.9])
    ax_T.set_xlim([9., 11.9])
    ax_T.set_xticklabels([])

    ax.set_ylim([-1,3.4])
    ax_R.set_ylim([-1,3.4])
    ax_R.set_yticklabels([])

    ax_R.set_xlim(left=0)
    ax_T.set_ylim(bottom=0)

    ax_R.set_xticks(np.arange(0,4,1))
    ax_T.set_yticks(np.arange(0,4,1))

    ax_R.set_xlabel(r'log$_{10}$(N)')
    ax_T.set_ylabel(r'log$_{10}$(N)')


    ax.set_xlabel(xlabel, fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)
    ax.legend(frameon=False, fontsize=9, scatterpoints=1, loc=4, markerscale=3)

    plt.savefig(savename, bbox_inches='tight', dpi=300)

    plt.show()
