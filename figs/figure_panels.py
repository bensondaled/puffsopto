import matplotlib.pyplot as pl
import numpy as np
import pyfluo as pf
import pandas as pd
import matplotlib as mpl
import h5py, datetime
from matplotlib.patches import Arc, Wedge, Ellipse
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, ttest_ind, ttest_1samp, linregress
from scipy.stats import norm as scinorm
from scipy.ndimage import label
from skimage.io import imread
from scipy.misc import imrotate
from scipy.ndimage import zoom
from scipy.misc import imresize
from scipy.signal import resample
from scipy.signal import medfilt
from scipy.stats import t as tdistribution
from scipy.stats import sem as scisem
sem = lambda *args,**kwargs: scisem(*args, ddof=1, **kwargs)
from skimage.exposure import equalize_adapthist as clahe
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import proportions_ztest
from statsmodels import robust
mad = robust.mad
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D, blended_transform_factory as blend
from matplotlib.text import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Path

jit = lambda sig=.1: np.random.normal(0,sig)

## Params
data_file = 'fig_data.h5'
FS = 7

mcols = {
            0: 'k',
            2: [.255, .541, .851], #'cornflowerblue',
            3: [.424, .251, .690], #'mediumpurple',
            4: [.243, .612, .549], #'lightseagreen',
            5: 'darkblue',#[.43, .0, .0],
            6: 'darkcyan',#[.62, .4, .22],
            7: 'steelblue',#[1., .71, .41],
            8: 'lightslategrey',
        }

mlabs = {
        0 : 'Light off',
        2 : 'Bilateral',
        3 : 'Left',
        4 : 'Right',
        5 : 'CP1',
        6 : 'CP2',
        7 : 'CP3',
        8 : 'Delay',
        }

pcols = {
            0: 'k',
            1: 'darkgrey',
            2: 'darkgrey',
            3: 'darkgrey',
            4: 'darkgrey',
        }

def psy_bsl(axs, panel_id):
    # baseline psychometrics
    ax = axs[panel_id]

    dats = {}
    with h5py.File(data_file) as h:
        grps = [grp for grp in h if grp.startswith('psy_bsl_')]
        for grp in grps:
            dats[grp] = h5_to_df(h, grp)
    
    ntrials = 0 
    for s,d in dats.items():
        if 'meta' in s:
            ax.plot(d.index, d['mean'], lw=.75, color='k', zorder=10)
        else:
            ax.plot(d.index, d['mean'], lw=.25, color='grey')
            ntrials += d['len'].sum()

    ax.set_xticks([-12,0,12])
    ax.set_yticks([0,.5,1])
    ax.set_yticklabels(['0','.5','1'])
    ax.set_xlabel('#R-#L puffs', fontsize=FS, labelpad=0)
    ax.set_ylabel('Fraction R choices', fontsize=FS, labelpad=0)
    ax.tick_params(labelsize=FS, pad=.01)

    ax.text(.25, .8, f'{len(dats)-1} mice', fontsize=FS-1, ha='center', va='center', transform=ax.transAxes, color='grey')
    ax.text(.25, .7, f'{ntrials:,.0f} trials', fontsize=FS-1, ha='center', va='center', transform=ax.transAxes, color='k')

    return axs

def impairment_simulation(axs, panel_id, agent=0, man=0):
    ax = axs[panel_id]

    titles = {
            0 : 'No impairment',
            1 : 'Sensation/attention\nimpairment',
            2 : 'Retention\nimpairment',
            3 : 'Action\nimpairment',
            }

    with h5py.File(data_file) as h:
        regr = h5_to_df(h, f'impairment_simulation_agent{agent}_man{man}')

    ax.errorbar(regr.index, regr['weight'], yerr=regr['yerr'], lw=.75, color=mcols[man])
    w567 = 3.8/3
    if man in [5,6,7]:
        ax.axvspan((man-5)*w567, (man-5+1)*w567, color=mcols[man], alpha=.3, lw=0)

    if man == 0:
        ax.text(-.95, .5, titles[agent], ha='center', va='center', transform=ax.transAxes, fontsize=FS)
        ax.set_ylabel('Weight on evidence\n(normalized)', fontsize=FS-1, labelpad=-2)

    if man == 5 and agent == 3:
        ax.text(1.2, -.3, 'Time in trial (s)', fontsize=FS, ha='center', va='center', transform=ax.transAxes)
    
    ax.set_ylim([0, .25])
    ax.set_yticks([0,.25])
    ax.set_yticklabels(['0','1'])
    ax.set_xticks([1,2,3])
    ax.set_xlim([0,3.8])


    if agent != 3:
        ax.set_xticklabels([])

    if man != 0:
        ax.set_yticklabels([])

    ax.tick_params(labelsize=FS)

    return axs

def regr_bsl(axs, panel_id):
    # baseline regressions
    ax = axs[panel_id]

    dats = {}
    with h5py.File(data_file) as h:
        grps = [grp for grp in h if grp.startswith('regr_bsl_')]
        for grp in grps:
            dats[grp] = h5_to_df(h, grp)
    
    for s,d in dats.items():
        if 'meta' in s:
            ax.errorbar(d.index, d['weight'], yerr=d['yerr'], lw=.75, color='k', zorder=10)
        elif 'shuf' in s:
            m = d['weight']
            err = d['yerr'] # 95-ci
            ptch = ax.fill_between(d.index, m-err, m+err, lw=.25, color='lightgrey', alpha=.6)
            ptch.set_edgecolor('dimgrey')
            ptch.set_linestyle((0,(1,1)))
        else:
            ax.plot(d.index, d['weight'], lw=.25, color='grey')

    ax.text(1.07, .09, 'Shuffle', fontsize=FS, color='grey', transform=ax.transAxes, ha='center', va='center')
    
    ax.set_ylim([-.05, None])
    ax.set_xlim([0.3, 3.5])
    ax.set_yticks([0, .5])
    ax.set_xlabel('Cue period (s)', fontsize=FS, labelpad=0)
    ax.set_ylabel('Weight on evidence', fontsize=FS, labelpad=2)
    ax.tick_params(labelsize=FS, pad=.01)

    return axs

def heatmap(axs, panel_id):
    ax = axs[panel_id]
    x,y,w,h = ax.get_position().bounds
    cax = ax.figure.add_axes([x+w+.05, y+.08, w/15, h-.16])

    with h5py.File(data_file) as h:
        hm = np.array(h['meta_heatmap']).T

    ims = ax.imshow(hm, cmap=pl.cm.PRGn, origin='lower', vmin=0, vmax=1)
    cb = pl.colorbar(ims, cax=cax)
    cb.set_ticks([])
    cb.outline.set_linewidth(0)
    cb.ax.tick_params(labelsize=FS)
    cax.text(.5, -.3, '100% L\nchoices', fontsize=FS-2, ha='center', va='center', transform=cax.transAxes)
    cax.text(.5, 1.3, '100% R\nchoices', fontsize=FS-2, ha='center', va='center', transform=cax.transAxes)

    ax.set_xlim([-.5, 13.5])
    ax.set_ylim([-.5, 13.5])

    ax.set_xticks([0,12])
    ax.set_yticks([0,12])

    ax.set_ylabel('#R puffs', fontsize=FS, labelpad=0)
    ax.set_xlabel('#L puffs', fontsize=FS, labelpad=0)

    ax.tick_params(labelsize=FS, pad=.01)
    
    axs[panel_id] = [ax,cax]
    return axs

def psys(axs, panel_id, manips=[0,2], ylab=True, grp='exp', easy=False):
    """Cue-period bilateral stimulation psychometrics
    """
    ax = axs[panel_id]

    if grp=='exp':
        pref = ''
    elif grp=='ctl':
        pref='ctl_'

    if all([i in [0,2,3,4] for i in manips]):
        reqstr = '_reqbil'
    elif all([i in [0,5,6,7,8] for i in manips]):
        reqstr = '_reqsub'
    else:
        reqstr = ''

    dats = {}
    with h5py.File(data_file) as h:
        for i in manips:
            rstr = reqstr if i==0 else ''
            dats[i] = h5_to_df(h, f'{pref}psy_manip{i}{rstr}')
        easy_means = h5_to_df(h, 'easy_means')

    easy_x = np.concatenate([d.index for _,d in dats.items()])
    easy_x = np.array([np.min(easy_x), np.max(easy_x)]) + np.array([-5,5])
    
    for mani,man in enumerate(manips):
        dat = dats[man]
        mean = dat['mean'].values
        yerr = dat[['yerr0','yerr1']].values.T
        lw = .75# if man==0 else 1
        ax.errorbar(dat.index, mean, yerr=yerr, color=mcols[man], lw=lw)

        # easy ctrl
        if easy:
            e = easy_means[easy_means.manip==man]
            for side in [0,1]:
                ei = e[e.side==side]
                ncor = (ei['mean'] * ei['n']).sum()
                ntot = ei['n'].sum()
                frac = ncor/ntot
                frac = 1-frac if side==0 else frac
                conf = confidence(frac, ntot)
                ofs = 1.5 + 2.*mani
                ofs = -ofs if side==0 else ofs
                ax.errorbar(easy_x[side]+ofs, frac, yerr=conf, lw=1., marker='o', markersize=2., mfc=mcols[man], mec=mcols[man], ecolor=mcols[man], mew=0)

    ax.set_ylim([0,1])
    if easy:
        ax.set_xlim([-23,23])
        ax.spines['bottom'].set_visible(False)
        ax.plot([-13,13], [0,0], transform=blend(ax.transData,ax.transAxes), lw=.25, color='k', clip_on=False)
        ax.plot([-23,-18], [0,0], transform=blend(ax.transData,ax.transAxes), lw=.25, color='k', clip_on=False)
        ax.plot([18,23], [0,0], transform=blend(ax.transData,ax.transAxes), lw=.25, color='k', clip_on=False)

        ax.text(21.5, -.15, 'R guided', fontsize=FS-2, rotation=90, transform=blend(ax.transData,ax.transAxes), ha='center', va='center')
        ax.text(-20.5, -.15, 'L guided', fontsize=FS-2, rotation=90, transform=blend(ax.transData,ax.transAxes), ha='center', va='center')
    else:
        ax.set_xlim([-16,16])
    ax.set_xticks([-12, 0, 12])
    ax.set_yticks([0,.5,1])
    ax.set_yticklabels(['0','','1'])
    ax.tick_params(labelsize=FS, pad=2)
    
    if ylab:
        ax.set_ylabel('Fraction R choices', fontsize=FS, labelpad=0)
    ax.set_xlabel('#R-#L puffs', fontsize=FS, labelpad=3)

    if grp == 'ctl':
        pass
        #ttl = ax.set_title(r'ChR2$^–$ controls', fontsize=FS-1)
        #ttl.set_position([.5, .93])

    # manip-specific labels
    shorts = dict(Bilateral='Bil', Left='L', Right='R')
    if manips == [0,2]:
        lab = mlabs[2]
        #lab = shorts.get(lab, lab)
        ax.text(.68, .32, lab, color=mcols[2], fontsize=FS-1, ha='center', va='center', transform=ax.transAxes)
        ax.text(.44, .8, 'Light\noff', color=mcols[0], fontsize=FS-1, ha='center', va='center', transform=ax.transAxes)
    elif manips == [0,3,4] or manips == [0,3] or manips == [0,4]:
        labr = 'Right'#shorts.get('Right')
        labl = 'Left'#shorts.get('Left')
        if 3 in manips:
            ax.text(.65, .3, labl, color=mcols[3], fontsize=FS-1, ha='center', va='center', transform=ax.transAxes)
        if 4 in manips:
            ax.text(.33, .67, labr, color=mcols[4], fontsize=FS-1, ha='center', va='center', transform=ax.transAxes)
    return axs

def fracs_simple(axs, panel_id, manips=[0,5,6,7], hard=False, dp_title=False, xlabs=False):
    ax = axs[panel_id]
    
    if hard:
        suf = '_hard'
    else:
        suf = ''
    
    if all([i in [0,2,3,4] for i in manips]):
        reqstr = '_reqbil'
    elif all([i in [0,5,6,7,8] for i in manips]):
        reqstr = '_reqsub'
    else:
        reqstr = ''
    
    dats = {}
    with h5py.File(data_file) as h:
        for i in manips:
            rstr = reqstr if i==0 else ''
            dats[i] = h5_to_df(h, f'fracs_manip{i}{rstr}{suf}')

    for idx,m in enumerate(manips):
        y = dats[m]
        y = y[y.subj==-1] # meta subj
        f = float(y.frac)
        n = float(y.n)
        err = confidence(f, n)
        ax.errorbar(idx, f*100, yerr=err*100, marker='o', color=mcols[m], markersize=2, lw=.5)

    if dp_title:
        ax.text(1.75, 1.25, 'Delay period perturbation', fontsize=FS, ha='center', va='center', clip_on=False, transform=ax.transAxes)

    ax.tick_params(labelsize=FS)
    ax.set_ylabel('% correct', fontsize=FS, labelpad=4)
    ax.set_xlim([-.25, len(manips)-.75])
    ax.set_xticks([])
    if xlabs:
        for mi,m in enumerate(manips):
            ax.text(mi, -.05, mlabs[m], color=mcols[m], fontsize=FS, rotation=90, transform=blend(ax.transData,ax.transAxes), ha='center', va='top')

    # signif

    return axs

def curly_brace(x, y, width=1/8, height=1., curliness=1/np.e, pointing='left', **patch_kw):
    '''Create patch of a curly brace

    Parameters
    ----------
    x : origin in x
    y : origin in y
    width : horizontal span of patch
    curliness : 1/e tends to look nice
    pointing : direction it points (only supports left and right currently)
    
    Notes
    -----
    To add to Axes:
    cb = curly_brace()
    ax.add_artist(cb)
    
    Thanks to:
    https://graphicdesign.stackexchange.com/questions/86334/inkscape-easy-way-to-create-curly-brace-bracket
    http://www.inkscapeforum.com/viewtopic.php?t=11228
    https://css-tricks.com/svg-path-syntax-illustrated-guide/
    https://matplotlib.org/users/path_tutorial.html
    '''

    verts = np.array([
           [width,0],
           [0,0],
           [width, curliness],
           [0,.5],
           [width, 1-curliness],
           [0,1],
           [width,1]
           ])
    
    if pointing == 'left':
        pass
    elif pointing == 'right':
        verts[:,0] = width - verts[:,0]

    verts[:,1] *= height
    
    verts[:,0] += x
    verts[:,1] += y

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             ]

    path = Path(verts, codes)

    pp = PathPatch(path, facecolor='none', **patch_kw) 
    return pp

def fracs(axs, panel_id, manips=[2,3,4], grp='exp', hard=False, labelmode=0, dispmode='pts', show_ctl=True, show_signif=True):
    """Cue-period bilateral stimulation fraction correct
    """
    ax = axs[panel_id]
    
    # preferences
    if grp=='exp':
        pref = ''
    elif grp=='ctl':
        pref='ctl_'

    if hard:
        suf = '_hard'
    else:
        suf = ''
    
    if all([i in [0,2,3,4] for i in manips]):
        reqstr = '_reqbil'
    elif all([i in [0,5,6,7,8] for i in manips]):
        reqstr = '_reqsub'
    else:
        reqstr = ''
    
    # load data
    dats = {}
    dats_ctl = {}
    with h5py.File(data_file) as h:
        for i in [0]+manips:
            rstr = reqstr if i==0 else ''

            df = h5_to_df(h, f'{pref}fracs_manip{i}{rstr}{suf}')
            if df is not None:
                df = df[df.subj!=-1]
                dats[i] = df

            df = h5_to_df(h, f'ctl_fracs_manip{i}{rstr}{suf}')
            if df is not None:
                df = df[df.subj!=-1]
                dats_ctl[i] = df
        #lapse_ignoring_cp1 = float(np.array(h['agent_statistics'][f'lapse_ignoring_cp1{suf}']))
        #lapse_ignoring_cp12 = float(np.array(h['agent_statistics'][f'lapse_ignoring_cp12{suf}']))
   
    assert np.all(dats[0].subj.values == dats[manips[0]].subj.values) # maintain order

    frs = {dn:dat.frac.values - dats[0].frac.values for dn,dat in dats.items()}
    frs_ctl = {dn:dat.frac.values - dats_ctl[0].frac.values for dn,dat in dats_ctl.items() if dat is not None}
    
    # plot individual data points
    if dispmode == 'pts':
        for row_idx in range(len(dats[0])):
            frs_i = np.array([frs[m][row_idx] for m in manips])
            ys = frs_i*100

            cols = [mcols[i] for i in manips]
            xs = [i+jit(.04) for i in range(len(manips))]
            #ax.plot(xs, ys, color='grey', lw=.25, zorder=0)
            ax.scatter(xs, ys, c=cols, s=4, edgecolor='none')
        
        # ctl
        if show_ctl:
            for row_idx in range(len(dats_ctl[0])):
                frs_i = np.array([frs_ctl[m][row_idx] for m in manips])
                ys = frs_i*100

                cols = ['lightgrey' for i in manips]
                xs = [i+jit(.1) for i in range(len(manips))]
                #ax.plot(xs, ys, color='grey', lw=.25, zorder=0)
                ax.scatter(xs, ys, c=cols, s=1, edgecolor='none')

    # plot me[di]an lines
    for idx,man in enumerate(manips):
        
        median = np.median(frs[man])*100
        mean = np.mean(frs[man])*100
        xs = [idx-.1, idx+.1]
        ys = [mean]*2
        if dispmode in ['pts']:
            ax.plot(xs, ys, lw=2, color=mcols[man], alpha=.5)
    
            # signif lines
            if show_signif:
                x0 = idx+.2
                col = 'k'#mcols[man]
                # curly brace:
                cb = curly_brace(x=x0, y=0, width=.2, height=mean, pointing='right', transform=ax.transData, lw=.5, edgecolor='dimgrey')
                ax.add_patch(cb)
                # or lines:
                #ax.plot([x0]*2, [0,mean], color=col, lw=.25, clip_on=False)
                #ax.plot([x0-.05, x0], [0,0], color=col, lw=.25, clip_on=False)
                #ax.plot([x0-.05, x0], [mean,mean], color=col, lw=.25, clip_on=False)
                ax.text(x0+.3, mean/2-.4, r'$\ast$', clip_on=False, ha='center', va='center', fontsize=FS-3, color='dimgrey')

        if dispmode in ['agg']:
            ax.plot(np.mean(xs), ys[0], lw=.5, color=mcols[man], marker='o', markersize=2)

        _sem = 100*sem(frs[man])
        iqr = 100*np.percentile(frs[man], [25,75])[:,None]
        iqr = np.abs(median-iqr)
        if dispmode == 'agg':
            ax.errorbar(idx, mean, yerr=_sem, color=mcols[man], markersize=0, lw=1)

        if man in frs_ctl and grp!='ctl' and show_ctl:
            ys_ctl = [np.median(frs_ctl[man])*100]*2
            ax.plot(xs, ys_ctl, lw=1, color='grey', alpha=.5)
     
    #minn = min([np.median(v) for _,v in frs.items()]) * 100
    #maxx = max([np.median(v) for _,v in frs.items()]) * 100
    if dispmode == 'pts':
        minn = min([v.min() for _,v in frs.items()]) * 100
        maxx = max([v.max() for _,v in frs.items()]) * 100
    elif dispmode == 'agg':
        minn = min([v.mean() for _,v in frs.items()]) * 100
        maxx = max([v.mean() for _,v in frs.items()]) * 100

    
    # annotation lines
    '''
    trans = blend(ax.transAxes, ax.transData)
    yvals = [0, -lapse_ignoring_cp1*100, -lapse_ignoring_cp12*100]
    labs = ['', 'agent that forgets\n1/3 of evidence', 'agent that forgets\n2/3 of evidence']
    lws = [.5, .25, .25]
    if labelmode == 0:
        skip = [.5, 1, 1] # 1 skip all, .5 skip text, 0 skip nothing
    elif labelmode == 1:
        skip = [0, 0, 0]
    for yval,lab,sk,lw in zip(yvals,labs,skip,lws):
        if sk < 1:
            ax.plot([0,1], [yval]*2, lw=lw, color='k', ls=':', dashes=[1,1], transform=trans)
        if sk == 0:
            ax.text(1.02, yval, lab, fontsize=FS-3, transform=trans, ha='left', va='center')
    '''
    
    # aesthetics
    ax.set_ylim([minn-3, maxx+5])
    if grp == 'ctl':
        ax.set_ylim([minn-30, maxx+2])
    ax.set_xlim([-.5, len(manips)-.5])
    ax.set_xticks(range(len(manips)))
    ax.set_xticklabels([])
    #ax.set_yticks([0, -30])
    ax.tick_params(labelsize=FS, pad=2)
    
    # chance line
    if labelmode == 0:
        xlim = ax.get_xlim()
        ax.plot(xlim, [0,0], lw=.5, ls=':', dashes=[1,1], color='dimgrey')
    
    # labels
    shorts = dict(Bilateral='Bilateral', Left='Left', Right='Right')
    for mi,man in enumerate(manips):
        rotation = 0
        y = -.06
        lab = mlabs[man]
        lab = shorts.get(lab, lab)
        ax.text(mi, y, lab, transform=blend(ax.transData,ax.transAxes), color=mcols[man], fontsize=FS, ha='center', rotation=rotation, va='top')
    ax.set_ylabel('∆% correct\n(light on - light off)', fontsize=FS, labelpad=5)
    #ax.set_xlabel('Trial type', fontsize=FS, labelpad=5)
    #if labelmode==1:
    #    ax.set_title('Impaired choice accuracy', fontsize=FS)



    # STATS
    print(f'Fraction correct in manips {manips}')
    test = ttest_rel
    for man in manips:
        d0 = dats[0]
        dx = dats[man]
        assert np.all(d0.subj == dx.subj)
        res = test(d0.frac.values, dx.frac.values)
        print(f'\tControl vs Man{man}: p={res.pvalue:0.4f}')
        
        # Ctl
        d0 = dats_ctl[0]
        dx = dats_ctl[man]
        assert np.all(d0.subj == dx.subj)
        res = test(d0.frac.values, dx.frac.values)
        print(f'\tChR-: Control vs Man{man}: p={res.pvalue:0.4f}')

    return axs

def reg_by_subj(axs, panel_id, manip=7):

    # CURRENTLY NOT IN USE

    ax = axs[panel_id]
    
    if all([i in [0,2,3,4] for i in manips]):
        reqstr = '_reqbil'
    elif all([i in [0,5,6,7,8] for i in manips]):
        reqstr = '_reqsub'
    else:
        reqstr = ''
    
    dats = {}
    with h5py.File(data_file) as h:
        for i in [manip]+[0]:
            rstr = reqstr if i==0 else ''
            dats[i] = np.array(h[f'regr_subj_manip{i}{rstr}'])
    
    regs = {dn: (dat - dats[0])[:,0] for dn,dat in dats.items()} # [:,0] for first bin
    print(regs)

    return axs

def reg_difs(axs, panel_id, manips=[5,6,7,8]):
    ax = axs[panel_id]
    
    if all([i in [0,2,3,4] for i in manips]):
        reqstr = '_reqbil'
    elif all([i in [0,5,6,7,8] for i in manips]):
        reqstr = '_reqsub'
    else:
        reqstr = ''
    
    dats = {}
    with h5py.File(data_file) as h:
        for i in manips+[0]:
            rstr = reqstr if i==0 else ''
            dats[i] = np.array(h[f'regr_subj_manip{i}{rstr}'])
    
    regs = {dn: (dat - dats[0])[:,0] for dn,dat in dats.items()} # [:,0] for first bin

    for row_idx in range(len(dats[0])):
        reg_i = np.array([regs[m][row_idx] for m in manips])
        ys = reg_i

        cols = [mcols[i] for i in manips]
        xs = [i+jit(.1) for i in range(len(manips))]
        #ax.plot(xs, ys, color='grey', lw=.25, zorder=0)
        #ax.scatter(xs, ys, c=cols, s=2, edgecolor='none')

    # median lines
    for idx,man in enumerate(manips):
        xs = [idx-.2, idx+.2]
        ys = [np.median(regs[man])]*2
        mean = np.mean(regs[man])
        err = sem(regs[man])

        #ax.plot(xs, ys, lw=2, color=mcols[man], alpha=.5)
        ax.errorbar(idx, mean, yerr=err, color=mcols[man], marker='o', markersize=2, lw=.5)

    #minn = min([v.min() for _,v in regs.items()]) 
    #maxx = max([v.max() for _,v in regs.items()]) 
    minn = min([np.mean(v) for _,v in regs.items()]) 
    maxx = max([np.mean(v) for _,v in regs.items()]) 
    
    trans = blend(ax.transAxes, ax.transData)
    ax.plot([0,1], [0,0], lw=.5, color='k', ls=':', dashes=[1,1], transform=trans)
    
    ax.set_ylim([minn-.09, maxx+.09])
    ax.set_xlim([-.5, len(manips)-.5])
    ax.set_xticks([])
    
    #for mi,man in enumerate(manips):
    #    ax.text(mi, -.15, mlabs[man], transform=blend(ax.transData,ax.transAxes), color=mcols[man], fontsize=FS-1, ha='center', rotation=0)

    ax.tick_params(labelsize=FS, pad=2)

    ax.set_ylabel('∆ weight on early evidence\n(light on - light off)', fontsize=FS, labelpad=5)
    #ax.set_xlabel('Trial type', fontsize=FS, labelpad=5)
    #ax.set_title('Reweighting of early evidence', fontsize=FS)
    
    return axs

def regs(axs, panel_id, manips=[0,2], ylab=True, hard=False, xlab=True, ylim=(-.02,.4), shade=True, annotate=False, main_title=False):
    """Cue-period bilateral stimulation regression
    """
    ax = axs[panel_id]

    if hard is False:
        suf = ''
    elif hard is True:
        suf = '_hard'
    
    if all([i in [0,2,3,4] for i in manips]):
        reqstr = '_reqbil'
    elif all([i in [0,5,6,7,8] for i in manips]):
        reqstr = '_reqsub'
    else:
        reqstr = ''

    dats = {}
    dats_bysub = {}
    with h5py.File(data_file) as h:
        for i in manips:
            rstr = reqstr if i==0 else ''
            dats[i] = h5_to_df(h, f'regr_manip{i}{rstr}{suf}')
            #dats_bysub[i] = np.array(h[f'regr_subj_manip{i}{reqstr}'])
    
    #regs = {dn: (dat) for dn,dat in dats_bysub.items()}

    w567 = 3.8/3
    #w567 = 1.5/3
    
    ofss = np.linspace(-.05,.05,len(manips))
    for man,ofs in zip(manips, ofss):
        dat = dats[man]
        mean = dat['weight'].values
        yerr = dat['yerr'].values
        #lw = .75 if man==0 else 1
        ax.errorbar(dat.index+ofs, mean, yerr=yerr, color=mcols[man], lw=.75, marker=None, elinewidth=.75)
        #ax.plot(dat.index+ofs, regs[man].T, color='grey', lw=.5)
        if man in [5,6,7] and shade:
            ax.axvspan((man-5)*w567, (man-5+1)*w567, color=mcols[man], alpha=.4, lw=0)
        elif man == 8 and shade:
            ax.axvspan(3.8, 4.3, color=mcols[8], alpha=.4, lw=0)

    if annotate and manips==[0,8]:
        ax.text(.5, .65, 'Light off', fontsize=FS-1, transform=ax.transAxes, ha='center', va='center')
        ax.text(.4, .1, 'Delay-period light', fontsize=FS-1, transform=ax.transAxes, ha='center', va='center', color=mcols[8])

    ax.set_ylim(ylim)
    ax.set_xlim([-.2,4.])
    if True:#ylim[1] == .4:
        ax.set_yticks([0,.2,.4])
        ax.set_yticklabels(['0','','0.4'])
    ax.tick_params(labelsize=FS, pad=2)
    
    if xlab:
        ax.set_xlabel('Cue period (s)', fontsize=FS, labelpad=3)
    if ylab:
        ax.set_ylabel('Weight on evidence', fontsize=FS, labelpad=0)
    if main_title:
        ax.text(.5, 1.2, 'Sub-cue-period light delivery', fontsize=FS, transform=ax.transAxes, ha='center', va='center')

    # signif
    if manips in ([0,7],[0,6]):
        d0 = dats[0]['weight'].values[0]
        dx = dats[manips[1]]['weight'].values[0]
        idx = np.array(dats[0].index)[0] - 0.5
        cb = curly_brace(idx, dx, height=d0-dx, width=.3, transform=ax.transData, edgecolor='dimgrey', lw=.5, pointing='left')
        ax.add_artist(cb)
        ax.text(idx-.1, (d0+dx)/2-.003, '$\\ast$', color='grey', fontsize=FS-3, ha='center', va='center')
    elif manips in ([0,8],[0,2,3,4]):
        dx = dats[manips[-1]]['weight'].values
        ex = dats[manips[-1]]['yerr'].values
        if manips==[0,2,3,4]:
            ys = [-.03]*len(dx)
        else:
            ys = [dx[i]-ex[i]-.02 for i in range(len(dx))]
        for i,idx in enumerate(dats[0].index):
            ax.text(idx+.08, ys[i], r'$\ast$', color='grey', fontsize=FS-3, ha='center', va='center', transform=ax.transData)

    return axs

def light_triggered_regression(axs, panel_id):
    ax = axs[panel_id]

    light_colour = [.424,.541,.620]

    with h5py.File(data_file) as h:
        ltr = h5_to_df(h, 'light_triggered_regression')
    
    dur = 3.8
    ax.errorbar(ltr.index, ltr.weights.values, yerr=ltr.err_bootstrap, lw=.75, color='k', elinewidth=.75) 

    smean = ltr.shuffle_mean.values
    serr = ltr.shuffle_err.values
    ax.fill_between(ltr.index, smean-serr, smean+serr, alpha=.2, color='grey', lw=0)
    ax.axvspan(0, dur/3, color=light_colour, alpha=.35, lw=0)

    trans = blend(ax.transAxes, ax.transData)
    ax.text(1., smean[-1], 'Shuffle', fontsize=FS, ha='left', va='center', transform=trans, color='grey')
    
    trans = blend(ax.transData, ax.transAxes)
    ax.text(dur/6, 1.03, 'Light delivery', fontsize=FS, ha='center', va='bottom', transform=trans, color=light_colour)

    ax.set_xlabel('Time from light onset (s)', fontsize=FS, labelpad=4)
    ax.set_ylabel('Weight on evidence', fontsize=FS, labelpad=6)

    ax.tick_params(labelsize=FS)

    return axs

def light_delivery_schematic(axs, panel_id, manips=[2,3,4], exclude_phases=[], labelmode=0):

    ax = axs[panel_id]
    
    starts = np.array([0, 3.8, 4.6])
    ends = np.array([3.8, 4.6, 7.])
    phase_ids = np.array([1,2,3])
    isex = np.array([pi in exclude_phases for pi in phase_ids])
    starts = starts[~isex]
    ends = ends[~isex]
    phase_ids = phase_ids[~isex]

    manip_bounds = {
            2 : [0, 3.8],
            3 : [0, 3.8],
            4 : [0, 3.8],
            5 : [0, 3.8/3],
            6 : [3.8/3, 2*3.8/3],
            7 : [2*3.8/3, 3.8],
            8 : [3.8, 4.3],
            }

    delta = .002
    emax = np.max(ends)
    starts = starts/emax + delta
    ends = ends/emax - delta
    manip_bounds = {k:(np.array(v))/emax + np.array([delta,-delta]) for k,v in manip_bounds.items()}
    txts = ['Cue period', 'Delay', 'Lick']

    for idx,(s,e,txt,pid) in enumerate(zip(starts,ends,txts,phase_ids)):
        if pid in exclude_phases:
            continue
        col = pcols[pid]
        ax.fill_between([s,e], 0, 1, color=col, alpha=.5, lw=0, transform=ax.transAxes)
        if labelmode in [0,1,2]:
            ax.text((s+e)/2, 1.2, txt, fontsize=FS-1, ha='center', va='center', color=col, transform=ax.transAxes)
    
    hpad = .1
    h = (1-hpad*(len(manips)+1))/len(manips)
    y0 = 1-h-hpad
    for mi,manip in enumerate(manips):
        s,e = manip_bounds[manip]
        y = y0 - mi*(h+hpad)
        rect = pl.Rectangle((s,y), e-s, h, transform=ax.transAxes, clip_on=False, facecolor=mcols[manip], edgecolor='none')
        ax.add_patch(rect)
        if labelmode in [0,2]:
            if labelmode == 2:
                tx = -.18
            elif labelmode == 0:
                tx = -.15
            ax.text(tx, y+h/2, mlabs[manip], color=mcols[manip], ha='center', va='center', fontsize=FS-1)
    
    if labelmode==0:
        lx = -.8
    elif labelmode in [1,3]:
        lx = -.4
    
    if labelmode in [1,3]:
        ax.text(lx, .5, 'Light\ndelivery', ha='center', va='center', fontsize=FS-1)
    elif labelmode in [0]:
        ax.text(lx, 1.2, 'Light\ndelivery', ha='center', va='center', fontsize=FS-1)
    
    if labelmode in [0,1,2,4]:
        ax.text(.5, -.4, 'Time in trial →', fontsize=FS-1, ha='center', va='center')
    elif labelmode in [3]:
        ax.text(.5, -.4, 'Cue period →', fontsize=FS-1, ha='center', va='center', color=pcols[1])
    
    ax.axis('off')

    return axs

def easy_control(axs, panel_id):
    ax = axs[panel_id]

    with h5py.File(data_file) as h:
        means = h5_to_df(h, 'easy_means')

    for idx,man in enumerate(means.manip.unique().astype(int)):
        dati = means[means.manip==man]
        mean = dati['mean'].mean() * 100
        err = 100*sem(dati['mean'])
        ax.errorbar(idx, mean, yerr=err, color=mcols[man], lw=.5, marker='o', markersize=2, markeredgewidth=0)
        
        # manip frac for comparison
        with h5py.File(data_file) as h:
            exp = h5_to_df(h, f'fracs_manip{man}')
            mean = exp['frac'].mean() * 100
            err = 100*sem(exp['frac'])
            ax.errorbar(idx, mean, yerr=err, color=mcols[man], lw=.5, marker='x', markersize=2, alpha=.5, markeredgewidth=.5)

    ax.tick_params(labelsize=FS)
    ax.set_xticks([])
    ax.set_ylim([50, 100])
    ax.set_yticks([50, 75, 100])

    return axs

def prettify_axes(axs):
    for ax in flatten(axs):
        for _,spine in ax.spines.items():
            spine.set_linewidth(.25)
        ax.tick_params(axis='both', width=.25, length=2)

def flatten(x):
    for i in x:
        if isinstance(i, (list,tuple,np.ndarray)):
            for j in flatten(i):
                yield j
        else:
            yield i

def h5_to_df(handle, name):
    if name not in handle:
        return None
    grp = handle[name]
    columns = sorted(list(grp.keys()))
    dat = np.array([grp[c] for c in columns]).T
    dat = np.atleast_2d(dat)
    df = pd.DataFrame(dat, columns=columns)
    df = df.set_index('index', drop=True)
    return df

def confidence(p,n):
    """Compute 95% confidence intervals

    Parameters
    ----------
    p : array-like
        probabilities (fractions) successes
    n : array-like
        trial counts
    """

    k = np.asarray(p*n) # number of successes
    n = np.asarray(n)
    
    # confidence interval bounds
    lo,hi = proportion_confint(k, n, method='jeffrey', alpha=0.05)

    # error values (difference from measured value)
    e0,e1 = p-lo,hi-p
    return np.array([e0,e1])[:,None]


def task_structure(axs, panel_id):
    PLACEHOLDERS = True

    ax = axs[panel_id]
    fig = ax.figure
    ax.axis('off')

    puff_r = imread('drawings/items/puff.png')
    puff_l = puff_r[:,::-1,:]
    water = imread('drawings/items/water.png')
    tongue_l = imread('drawings/items/tongue.png')
    tongue_r = tongue_l[:,::-1,:]
    mouse = imread('drawings/items/mouse.png')

    def draw_im(img, x, y, w, h):
        # x and y in ax coords
        # w and h in fig coords
        disp_coords = ax.transAxes.transform([x,y])
        fig_coords = fig.transFigure.inverted().transform(disp_coords)
        axim = fig.add_axes([fig_coords[0], fig_coords[1], w, h])
        axim.axis('off')
        kw = {}
        if PLACEHOLDERS is True:
            axim.imshow(img, **kw)
        return axim
    
    ax_x,ax_y,ax_w,ax_h = ax.get_position().bounds
    # lines
    z = .01
    end = 1.
    dur = 10 # seconds # in truth 12, but cutting ITI
    fps = (end-z)/dur # fraction per second
    yposs = [i/6 + .02 for i in range(6)]
    stick = .1
   
    # draw mouse
    axim = draw_im(mouse, .2, 1.5, ax_w/2, ax_h/2)

    tyo = .0033333333 # tick y offset due to line thickness differences
    # phase boundaries
    ts = [z+fps*i for i in [1., 4.8, 5.6, 8.6]]
    #for t in ts:
    #    ax.plot([t,t], [yposs[0],yposs[-1]+stick], transform=ax.transAxes, lw=1, color='grey', linestyle='--')
    evidence_boundaries = [z+fps*i for i in [1.,4.8]]
    x0,x1 = evidence_boundaries
    w = x1-x0
    y = yposs[0] + tyo
    h = yposs[-1]-y+1/6 - tyo
    rect = pl.Rectangle((x0,y), w, h, color='lightgray', transform=ax.transAxes)
    ax.add_patch(rect)

    ax.text(z+fps*2.9, yposs[-1]+.18, 'Cue period', transform=ax.transAxes, ha='center', fontsize=FS, color='dimgrey')

    # delay period shading
    x0,x1 = [z+fps*i for i in [4.87,5.67]]
    w = x1-x0
    y = yposs[0] + tyo
    h = yposs[-1]-y+1/6 - tyo
    rect = pl.Rectangle((x0,y), w, h, color='darkgray', transform=ax.transAxes)
    ax.add_patch(rect)

    # iti shading
    x0,x1 = [z+fps*i for i in [8.63,12]]
    w = x1-x0
    y = yposs[0] + tyo
    h = yposs[-1]-y+1/6 - tyo
    rect = pl.Rectangle((x0,y), w, h, color='whitesmoke', transform=ax.transAxes)
    ax.add_patch(rect)

    # events
    dlick_time = 6.150
    elw = .75
    # start sound
    ax.plot([z,z], [yposs[-1]+tyo,yposs[-1]+stick+tyo], transform=ax.transAxes, lw=elw, color='k')
    # error sound
    t = z+fps*(dlick_time+.090)
    ax.plot([t,t], [yposs[-1],yposs[-1]+stick], transform=ax.transAxes, color='k', ls=':', dashes=[.6, .2], lw=elw)
    # l puffs
    t = [z+fps*ti for ti in [1,1.2,1.6,1.8,2.5,3.5,3.8,4.4,4.8]]
    for ti in t:
        ax.plot([ti,ti], [yposs[-2]+tyo,yposs[-2]+stick+tyo], transform=ax.transAxes, lw=elw, color='dimgrey')
    # r puffs
    t = [z+fps*ti for ti in [1,2.2,3.2,4.8]]
    for ti in t:
        ax.plot([ti,ti], [yposs[-3]+tyo,yposs[-3]+stick+tyo], transform=ax.transAxes, lw=elw, color='dimgrey')
    # l licks
    tongue_colour = np.array([212,123,121])/255
    t = [z+fps*ti for ti in [dlick_time+i for i in np.arange(0,12)/8]]
    t += [z+fps*ti for ti in [8.0+i for i in np.arange(0,3)/8]]
    for ti in t:
        ax.plot([ti,ti], [yposs[-4]+tyo,yposs[-4]+stick+tyo], transform=ax.transAxes, lw=elw, color=tongue_colour)
    # decision lick
    ax.plot([z+fps*i for i in [dlick_time,dlick_time]], [yposs[0]+tyo,1.08+tyo], lw=.5, transform=ax.transAxes, color=tongue_colour, linestyle=':', clip_on=False, dashes=[1,1])
    ax.plot([z+fps*i for i in [dlick_time,6.5]], [1.08+tyo,1.08+tyo], lw=.5, transform=ax.transAxes, color=tongue_colour, linestyle=':', clip_on=False, dashes=[1,1])
    ax.text(6.6*fps+z, yposs[-1]+.18, 'Decision lick', ha='left', transform=ax.transAxes, fontsize=FS, color=tongue_colour)
    # r licks
    t = [z+fps*ti for ti in [7.6+i for i in np.arange(0,3)/8]]
    for ti in t:
        ax.plot([ti,ti], [yposs[-5]+tyo,yposs[-5]+stick+tyo], transform=ax.transAxes, lw=elw, color=tongue_colour)
    # water
    water_colour = np.array([71,192,242])/255
    t = z+fps*(dlick_time+.090)
    ax.plot([t,t], [yposs[0]+.01,yposs[0]+stick], transform=ax.transAxes, lw=elw, color=water_colour)
    # main lines 
    for ypos in yposs[1:]:
        ax.plot([z,end], [ypos,ypos], transform=ax.transAxes, lw=.5, color='k')

    # time line
    ax.plot([z, z+8*fps], 2*[yposs[0]], transform=ax.transAxes, lw=.5, color='k')
    h0,h1 = 8.9,9.1
    ax.plot([z+8*fps, z+h0*fps], 2*[yposs[0]], transform=ax.transAxes, lw=.5, color='k')
    ax.plot([z+h1*fps, 1], 2*[yposs[0]], transform=ax.transAxes, lw=.5, color='k')
    # hashes
    yp0 = yposs[0]
    dy = .04
    dx = .007
    ax.plot([z+h0*fps-dx, z+h0*fps+dx], [yp0-dy, yp0+dy], transform=ax.transAxes, lw=.5, color='k', clip_on=False)
    ax.plot([z+h1*fps-dx, z+h1*fps+dx], [yp0-dy, yp0+dy], transform=ax.transAxes, lw=.5, color='k', clip_on=False)

    # time ticks
    tt = np.arange(0,9,2)
    t = [z+fps*ti for ti in [i for i in tt]]
    for tidx,(ti,tti) in enumerate(zip(t,tt)):
        ax.plot([ti,ti], [yposs[0],yposs[0]-stick/2], transform=ax.transAxes, lw=.5, color='k', clip_on=False)
    
   
    times_to_mark = [0,8]
    for ttm in times_to_mark:
        ax.text(z+ttm*fps, yposs[0]-stick/2-.09, str(ttm), ha='center', va='center', fontsize=FS, clip_on=False, transform=ax.transAxes)

    ax.text(z+(end-z)/2, -.2, 'Time (s)', ha='center', transform=ax.transAxes, fontsize=FS, clip_on=False)

    # images
    # speaker
    pad = .05
    cx,cy = z-pad, yposs[-1]+stick/2
    w = .006
    h = w * ax_w/ax_h 
    rect = pl.Rectangle((cx-w/2, cy-h/2), w, h, transform=ax.transAxes, color='k')
    ax.add_patch(rect)

    pg = pl.Polygon([  [cx+w/2,cy+h/2],
                       [cx+w/2,cy-h/2],
                       [cx+w/2+w, cy-h/2-h],
                       [cx+w/2+w, cy+h/2+h],
        ], color='k', transform=ax.transAxes)
    ax.add_patch(pg)
    
    arc = Arc((cx+w/2+.012,cy), .01, .01*4, 0, -90, 90, transform=ax.transAxes, facecolor='none', edgecolor='k', lw=.5)
    ax.add_patch(arc)
    arc = Arc((cx+w/2+.019,cy), .015, .015*5, 0, -90, 90, transform=ax.transAxes, facecolor='none', edgecolor='k', lw=.5)
    ax.add_patch(arc)
    arc = Arc((cx+w/2+.026,cy), .02, .02*6, 0, -90, 90, transform=ax.transAxes, facecolor='none', edgecolor='k', lw=.5)
    ax.add_patch(arc)
   
    ps = .04
    draw_im(puff_l, z-2.3*ps/2, yposs[-2]-1.*ps/2, ps, ps)
    draw_im(puff_r, z-2.3*ps/2, yposs[-3]-1.*ps/2, ps, ps)
    ws = .03
    draw_im(water, z-3*ws/2, yposs[0]-ws/2, ws, ws)
    ls = .035
    draw_im(tongue_l, z-2.9*ls/2, yposs[-4]-1.5*ls/2, ls, ls)
    draw_im(tongue_r, z-2.9*ls/2, yposs[-5]-1.5*ls/2, ls, ls)
    
    """
    strs = ['Sound cue','Left puff','Right puff','Left lick','Right lick','Water reward']
    cols = ['k','grey','grey','salmon','salmon','cadetblue']
    for s,yp,c in zip(strs, yposs[::-1], cols):
        ax.text(z-.04, yp+ls, s, fontsize=FS-3, ha='right', va='center', transform=ax.transAxes, color=c)
    """

    return axs

def ephys(axs, panel_id, cell_type='pc'):
    # cell types: pc / dcn
    ax = axs[panel_id]

    fig = ax.figure
    x,y,w,h = ax.get_position().bounds
    ax.remove()
    
    hpad = h/1.5
    wpad = .03
    w2 = w/2 - wpad

    ax0 = fig.add_axes([x,y+hpad,w,h/6])
    ax1 = fig.add_axes([x,y,w2,h/2])
    ax2 = fig.add_axes([x+w2+wpad*2,y,w2,h/2])

    axs[panel_id] = [ax0,ax1,ax2]

    with h5py.File(data_file) as h:
        grp = h[f'ephys_rawtrace_{cell_type}']
        time = np.array(grp['time'])
        tr = np.array(grp['trace'])
        ss = np.array(grp['ss'])
        cs = np.array(grp['cs'])
    
    # raw trace
    ax0.plot(time, tr, color='k', lw=.25)
    ax0.plot(ss, tr.max()*np.ones_like(ss), marker='o', color='dimgrey', lw=0, markersize=.75, markeredgewidth=0, clip_on=False) 
    ax0.plot(cs, tr.max()*np.ones_like(cs), marker='o', color='dimgrey', lw=0, markersize=.75, markeredgewidth=.25, markerfacecolor='none', clip_on=False)
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.set_xlim([time[0],time[-1]])
    scalebar_x = 0.050 # seconds
    ax0.plot([time[-1]-scalebar_x,time[-1]], [-.1]*2, lw=.5, color='k', clip_on=False, transform=blend(ax0.transData, ax0.transAxes))
    ax0.axis('off')
    
    # means
    for pro,ax,xmax in zip([6,3],[ax1,ax2],[2.24,5]):
        with h5py.File(data_file) as h:
            grp = h[f'ephys_pro{pro}_celltype-{cell_type}']
            lstart = grp.attrs['light_start_time']
            lstop = grp.attrs['light_stop_time']
            time = np.array(grp['time'])
            mean = np.array(grp['mean'])
            sem = np.array(grp['sem'])
        bin_width = np.mean(np.diff(time))
        ax.bar(time, mean, yerr=sem, width=bin_width-.01, ecolor='dimgrey', color='k', error_kw=dict(lw=.001))

        bary = mean.max()+sem.max()+6
        bardur = lstop-lstart
        ax.plot([lstart,lstop], 2*[bary], color='dodgerblue', lw=2, zorder=0)
        ax.text((lstart+lstop)/2, bary+20, f'{bardur} s', fontsize=FS-1, color='dodgerblue', ha='center', va='center')

        ax.set_xlim([-1,xmax])
        ax.set_xticks([])
        ax.set_yticks([60, 120])
        if ax is ax2:
            ax.set_yticklabels([])

    for ax in axs[panel_id]:
        ax.tick_params(labelsize=FS, pad=1)
    ax1.set_ylabel('Firing rate\n(Hz)', fontsize=FS, labelpad=2)

    return axs

def latency(axs, panel_id, manips=[0,2,3,4]):
    ax = axs[panel_id]
    
    if all([i in [0,2,3,4] for i in manips]):
        reqstr = '_reqbil'
    elif all([i in [0,5,6,7,8] for i in manips]):
        reqstr = '_reqsub'
    else:
        reqstr = ''
       
    dats = {}
    with h5py.File(data_file) as h:
        grp = h['latency']
        for i in manips:
            rstr = reqstr if i==0 else ''
            dats[i] = np.array(grp[f'manip{i}{rstr}'])

    for i,man in enumerate(manips):
        dat = dats[man]

        #print(np.mean(dat>2500))
        #dat = dat[dat<2500]

        col = mcols[man]
        bplot = ax.boxplot(dat, positions=[i], showfliers=False, notch=True, widths=[.6], whis=[10,90], bootstrap=1000)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            pl.setp(bplot[element], color=col, linewidth=.5)
        pl.setp(bplot['caps'], linewidth=0)
        '''
        vio = ax.violinplot(dat, positions=[i], showmedians=True)
        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vio[partname].set_edgecolor(col)
            vio[partname].set_linewidth(.5)
        for vp in vio['bodies']:
            vp.set_facecolor(col)
            vp.set_edgecolor(col)
            vp.set_linewidth(.5)
        '''
    
    ax.set_ylim([350,850])
    ax.set_yticks([400,600,800])
    ax.set_xlim([-.5, len(manips)-.5])
    ax.set_xticks(np.arange(len(manips)))
    ax.set_xticklabels([])
    shorts = dict(Bilateral='Bil', Left='L', Right='R')
    shorts['None']='Off'
    for idx,m in enumerate(manips):
        lab = shorts[mlabs[m]]
        rot = 90 if len(lab)>1 else 0
        ax.text(idx, -.12, lab, color=mcols[m], fontsize=FS, rotation=rot, transform=blend(ax.transData,ax.transAxes), ha='center', va='center')
    ax.tick_params(labelsize=FS, pad=2) 
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('Decision latency (ms)', fontsize=FS, labelpad=3)

    # FOR STATS: see main.py paired within subjects

    return axs

def likelihood_landscape_julia(axs, panel_id, manip=234, param_idxs=[1,2], xlab=False, ylab=False, yticklabs=True, cbar=False):
    ax = axs[panel_id]

    names = {
            0:'$\\sigma^2_a$', 
            1:'$\\sigma^2_s$', 
            2:'↑\nUnstable\n\n\n$\\lambda$\n\n\nLeaky\n↓', 
            3:'bias',
            4:'Lapse', 
            }
    mnames = {0:'Ctrl',234:'Full CP',67:'Mid-late CP',8:'Delay'}

    with h5py.File(data_file) as h:
        grp = h[f'ddm_julia_man{manip}']
        params = np.array(grp['fit_params'])
        hess = np.array(grp['hessian'])
        bootp = np.array(grp['fit_params_boot'])
        
        grp0 = h[f'ddm_julia_man0']
        params0 = np.array(grp0['fit_params'])
        hess0 = np.array(grp0['hessian'])

    cmap_x = pl.cm.gist_heat
    cmap_0 = pl.cm.Greys_r

    #sandbox for colormap
    vals = cmap_x(np.linspace(0,1,256))
    vals_old = vals.copy()
    vals[:,2] = vals_old[:,0]
    vals[:,0] = vals_old[:,2]
    cmap_x = mcolors.LinearSegmentedColormap.from_list('new',vals,N=256)
    
    #nbins = 20
    #ws = np.linspace(.001,1.96,nbins)[::-1] # number of se's from mean
    
    stepsize = .09
    ws = np.arange(.001,3.+stepsize,stepsize)[::-1] # number of se's from mean

    for i,w in enumerate(ws):
        col = cmap_x( scinorm.pdf(w)/scinorm.pdf(0) )
        xs,ys = error_ellipse(params, hess, param_idxs, w)
        fillx = ax.fill(xs,ys,color=col, lw=0, alpha=.6)
        
        col = cmap_0(i/len(ws))
        xs,ys = error_ellipse(params0, hess0, param_idxs, w)
        fill0 = ax.fill(xs,ys,color=col, lw=0, alpha=.6)
        se0 = np.sqrt(np.diag(np.linalg.inv(hess0)))
        #ax.plot(params0[param_idxs[0]], params0[param_idxs[1]], color='grey', marker='o', markersize=3, lw=0, markeredgewidth=.5)

        se_x = 0#se0[param_idxs[0]]
        se_y = 0#se0[param_idxs[1]]
        #ax.errorbar(params0[param_idxs[0]], params0[param_idxs[1]], color='grey', xerr=se_x, yerr=se_y, lw=.5, marker='o', markersize=2)

    # 95ci lines
    '''
    xs,ys = error_ellipse(params, hess, param_idxs, 1.96)
    ax.plot(xs,ys,color='red', lw=.15)
    xs,ys = error_ellipse(params0, hess0, param_idxs, 1.96)
    ax.plot(xs,ys,color='white', lw=.15)
    '''

    # datapoints from bootstrap
    #ax.scatter(*bootp.T[param_idxs], color='indigo', marker='.', s=1, alpha=.2, zorder=100)

    # bestfit marks
    ax.plot(params[param_idxs[0]], params[param_idxs[1]], marker='x', color=cmap_x(.5), markersize=2, markeredgewidth=.5, alpha=.4)
    
    lims = {0:[-1,5000], 1:[-1,None], 2:[-2, .6], 3:[None,None], 4:[-.01, .7]}
    ax.set_xlim(lims[param_idxs[0]])
    ax.set_ylim(lims[param_idxs[1]])

    # lambda vertical line
    if param_idxs[1]==2:
        ax.axhline(0, color='white', ls=':', lw=.5)

    # specific tick markings
    if param_idxs[0]==4:
        ax.set_xticks([0,.3,.6])
    if param_idxs[1]==2:
        ax.set_yticks([0,-1,-2])

    if xlab:
        ax.set_xlabel(names[param_idxs[0]], fontsize=FS, labelpad=5)
    if ylab and param_idxs[1]==2:
        ax.text(-.4, .5, names[param_idxs[1]], fontsize=FS, ha='center', va='center', transform=ax.transAxes)
    elif ylab:
        ax.set_ylabel(names[param_idxs[1]], fontsize=FS, labelpad=3)

    if yticklabs is False:
        ax.set_yticklabels([])

    # title as schematics
    mini_light_delivery_schematic(x0=.2, y0=1.04, w=.6, h=.1, m=manip, transform=ax.transAxes, ax=ax)

    ax.set_facecolor('k')
    ax.tick_params(labelsize=FS)

    # colorbar
    if cbar:
        fig = ax.figure
        x,y,w,h = ax.get_position().bounds
        hcb = h/2
        wcb = w/15
        xpad = w/4
        cax1 = fig.add_axes([x+w+xpad, y+(h-hcb)/2, wcb, hcb])
        cax2 = fig.add_axes([x+w+xpad+wcb*4, y+(h-hcb)/2, wcb, hcb])

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=cmap_x, norm=norm,orientation='vertical')
        cb2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap_0, norm=norm,orientation='vertical')
        
        for cax in [cax1,cax2]:
            cax.tick_params(labelsize=FS-1, width=.2, pad=1, length=1)
        for cb in [cb1,cb2]:
            cb.outline.set_linewidth(.2)
        cb2.set_ticks([0,.5,1])
        cb2.set_ticklabels(['0','','1'])
        cb1.set_ticks([0,.5,1])
        cb1.set_ticklabels([])
        
        cax2.text(-1.3, 1.3, 'Normalized\nlikelihood', fontsize=FS-1, ha='center', va='center', transform=cax2.transAxes)
        
        cax1.text(.5, -.25, 'Light\non', fontsize=FS-1, ha='center', va='center', transform=cax1.transAxes)
        cax2.text(.5, -.25, 'Light\noff', fontsize=FS-1, ha='center', va='center', transform=cax2.transAxes)
    
    return axs

def likelihood_landscape(axs, panel_id, manip=234, param_idxs=[1,2]):
    ax = axs[panel_id]

    names = ['$\\lambda$', '$\\sigma^2_a$', '$\\sigma^2_s$', 'lapse', 'bias']
    mnames = {0:'Ctrl',234:'Full CP',67:'Mid-late CP',8:'Delay'}

    pistr = '-'.join([str(i) for i in param_idxs])
    clips = {
            '1-2': {234:(.999, 1), 67:(.995,1), 8:(.85,1)},
            '3-0': {234:(.5, 1), 67:(.6,1), 7:(.5,1)},
            }
    
    lims = {
            '1-2': {8:(0,30), 67:(0,30)},
            }

    minmaxs = {0:(-2,2), 1: (0,None), 2: (0,None), 4:(0,1)}

    p0,p1 = param_idxs
    p0,p1 = int(p0),int(p1)

    with h5py.File(data_file) as h:
        grp = h[f'ddm_man{manip}']
        grp0 = h[f'ddm_man0']
        params,se = np.array(grp['fit_params'])
        params0,se0 = np.array(grp0['fit_params'])
        mapp_ds = grp[f'map{p0}-{p1}']
        i0,i1 = np.array(mapp_ds.attrs['i0']),np.array(mapp_ds.attrs['i1'])
        mapp = np.array(mapp_ds)

    mapp = mapp/mapp.max() # NLL
    l_map = np.exp(-mapp) # likelihood
    l_map = (l_map-l_map.min())/(l_map.max()-l_map.min())
    
    vmin,vmax = clips.get(pistr, {}).get(manip, (0,1))

    c = ax.pcolormesh(i1, i0, l_map, cmap=pl.cm.gist_heat, vmin=vmin, vmax=vmax)

    # pts
    x0,y0 = params0[p1],params0[p0]
    xX,yX = params[p1],params[p0]
    ex0,ey0 = se0[p1],params0[p0]
    exX,eyX = se[p1],params[p0]

    ax.errorbar(x0, y0, marker='o', color='lightgrey', markersize=2, lw=0)
    #ax.errorbar(x0, y0, yerr=ey0, xerr=ex0, color='lightgrey', lw=.5)
    #ax.errorbar(xX, yX, yerr=eyX, xerr=exX, color='lightblue', lw=.5)
    
    ax.tick_params(labelsize=FS, pad=2)
    ax.set_xlim([i1[0],i1[-1]])
    ax.set_ylim([i0[0],i0[-1]])
    ax.set_xticks([i1[0],i1[-1]])
    ax.set_yticks([i0[0],i0[-1]])
    lim = lims.get(pistr, {}).get(manip, None) 
    if lim is not None:
        ax.set_xlim([lim[0],lim[1]])
        ax.set_ylim([lim[0],lim[1]])
        ax.set_xticks([lim[0],lim[-1]])
        ax.set_yticks([lim[0],lim[-1]])
    ax.set_xlabel(names[p1], fontsize=FS, labelpad=-5)
    ax.set_ylabel(names[p0], fontsize=FS, labelpad=-5)

    #ax.set_title(mnames[manip], fontsize=FS)
    # title as schematics
    mini_light_delivery_schematic(x0=.2, y0=1.04, w=.6, h=.1, m=manip, transform=ax.transAxes, ax=ax)
    
    '''
    cb = pl.colorbar(c)
    cb.set_ticks(cb.get_clim())
    cb.set_ticklabels(['0','1'])
    cb.set_label('Normalized likelihood')
    '''

    return axs

def remove_leading_zero(s):
    s = s.get_text()
    print(s)
    if len(s) == 0:
        return s
    if s[0] != '0':
        return s
    s = s[1:]
    return s

def bound_error(e, p, bounds=None):
    # given error bar e on point p, keep range within bounds by adjusting error values

    if bounds is None:
        return e

    b0,b1 = bounds

    etemp = np.array([-e,e])
    lims = p + etemp

    e = np.array([e,e])

    if (b0 is not None) and lims[0]<b0:
        e[0] = p
    if (b1 is not None) and lims[1]>b1:
        e[1] = b1-p

    return np.array(e)[:,None]

def ddm_params_julia_bootstrap(axs, panel_id, param_idx, manips=[8,234,0], yticklabels=False):
    '''Display best-fit parameters from DDM fits
    '''
    ax = axs[panel_id]

    names = [
            '$\\sigma^2_a$\nAccumulator\nnoise', 
            '$\\sigma^2_s$\nSensory\nnoise', 
            '$\\lambda$\nMemory\n← leak', 
            'Bias\n',
            'Lapse\n',
            ]
    units = [
            '(puffs$^2$ / s)',
            '(puffs$^2$ / s)',
            '(s$^{-1}$)',
            '',
            '',
            ]
    mnames = {0:'Light off',234:'Full CP',67:'Mid-late CP',8:'Delay','0_sub6000flip25':'Simulated\nimpairment'}
    xlim = {2:(-3.1,.3), 0:(-5,80), 1:(-5,420), 4:(-.01,.70), 3:(-2.5,2.5)}
    minmaxs = {0:(0,None), 1: (0,None), 2: None, 3:None, 4:(0,1)}
    
    params = {}
    with h5py.File(data_file) as h:
        for manip in manips:
            grp = h[f'ddm_julia_man{manip}']
            p = np.array(grp['fit_params_boot'])
            params[manip] = p
   
    print(f'DDM params, bootstrap fits, param {param_idx}')
    for mi,manip in enumerate(manips):
        p = params[manip][:,param_idx]
        median = np.median(p, axis=0)
        ci_lo, ci_hi = np.percentile(p, [2.5,97.5], axis=0)
         
        p0 = params[0][:,param_idx]
        ci0_lo, ci0_hi = np.percentile(p0, [2.5,97.5], axis=0)
        
        print(f'\tManip{manip}\t:\t{ci_lo:0.3f} – {ci_hi:0.3f}')

        #std = p.std(axis=0, ddof=1)     
        #err = bound_error(std, median, minmaxs[param_idx])
        iqr_lo,iqr_hi = np.percentile(p, [25,75], axis=0)
        
        h = .1 # thickness of violins

        #ax.fill_between([ci_lo,ci_hi], mi-h, mi+h, color='dodgerblue', alpha=.1, lw=0)
        
        # display distribution with kernel density estimate
        vio = ax.violinplot(p, positions=[mi], widths=h*2, vert=False)
        for partname in ('cbars','cmins','cmaxes'):
            vio[partname].set_edgecolor('grey')
            vio[partname].set_linewidth(0)
            vio[partname].set_alpha(0.15)
        #vio['cmedians'].set_linewidth(1)
        #vio['cmedians'].set_color('k')
        #vio['cmedians'].set_alpha(1)
        #vio['cmedians'].set_length(2)
        for vp in vio['bodies']:
            vp.set_facecolor('grey')
            vp.set_edgecolor('grey')
            vp.set_linewidth(0)
            vp.set_alpha(0.15)
       
        # show median line
        ax.plot([median]*2, [mi-h,mi+h], lw=1, color='k')
        
        # signifiance markers
        #if ci_lo>ci0_hi or ci_hi<ci0_lo:
        #    ax.plot([median]*2, [mi-h,mi+h], lw=.5, color='red')

        # 95ci lines
        #ax.plot([ci_lo]*2, [mi-h,mi+h], lw=.5, color='dimgrey')
        #ax.plot([ci_hi]*2, [mi-h,mi+h], lw=.5, color='dimgrey')

        #ax.errorbar(median, mi, xerr=err, marker=None, lw=.5, color='k')
        #ax.plot([iqr_lo, iqr_hi], [mi]*2, marker=None, lw=.5, color='k')
        
        if manip == 0:
            ax.axvline(median, linestyle=':', dashes=[2,3], color='k', lw=.5, zorder=0)

    if param_idx == 2:
        pass
        #ax.axvline(0, linestyle=':', dashes=[1,1], color='k', lw=.1)

    ax.set_yticks([])
    if yticklabels is not False:
        if yticklabels == 'text':
            for i,m in enumerate(manips):
                mn = mnames[m]
                if m==0 and '0_sub6000flip25' in manips:
                    mn = 'Data\n(light off)'
                ax.text(-.4, i, mn, fontsize=FS, ha='center', va='center', transform=blend(ax.transAxes, ax.transData))
        else:
            ax.set_yticklabels([])

            for mi,m in enumerate(manips):
                h = .3
                wfull = .7
                wdel = .3
                xpad = .1
                
                mini_light_delivery_schematic(ax, x0=-wfull-wdel-xpad, y0=mi-h/2, w=wfull, h=h, m=m, transform=blend(ax.transAxes,ax.transData), titles=True)
            
    else:
        ax.set_yticklabels([])
    
    ax.tick_params(labelsize=FS)
    ax.set_ylim([-.4, len(manips)-1+.4])
    ax.set_xlim(xlim[param_idx])
    ax.set_title(names[param_idx], fontsize=FS)
    ax.set_xlabel(units[param_idx], fontsize=FS, labelpad=3)

    ax.spines['left'].set_visible(False)

    return axs

def ddm_julia_params(axs, panel_id, param_idx, manips=[8,67,234,0], yticklabels=False):
    '''Display best-fit parameters from DDM fits
    '''
    ax = axs[panel_id]

    names = [
            '$\\sigma^2_a$\nAccumulator\nnoise', 
            '$\\sigma^2_s$\nSensory\nnoise', 
            '$\\lambda$\nMemory\n← leak', 
            'Bias\n',
            'Lapse\n',
            ]
    mnames = {0:'Ctrl',234:'Full CP',67:'Mid-late CP',8:'Delay'}
    xlim = {2:(-2.5,.5), 0:(-5,200), 1:(-5,200), 4:(-.1,1), 3:(-3,3)}
    minmaxs = {0:(0,None), 1: (0,None), 2: None, 3:(0,1), 4:(0,1)}
    
    params = {}
    hessians = {}
    
    with h5py.File(data_file) as h:
        for manip in manips:
            grp = h[f'ddm_julia_man{manip}']
            p = np.array(grp['fit_params'])
            hess = np.array(grp['hessian'])
            params[manip] = p
            hessians[manip] = hess
    
    print(names[param_idx])
    for mi,manip in enumerate(manips):
        p = params[manip][param_idx]
        hess = hessians[manip]
        s = np.sqrt(np.diag(np.linalg.inv(hess)))[param_idx]

        lo = p-1.96*s
        hi = p+1.96*s
        print(f'\t {mnames[manip]} : {lo:0.3f} – {hi:0.2f}')
        #s = bound_error(s, p, minmaxs[param_idx])

        #if param_idx in [1,2] and manip==234:
        #    color = 'grey'
        #elif param_idx == 0 and manip==67:
        #    color = 'grey'
        #elif param_idx == 3 and manip==8:
        #    color = 'grey'
        #else:
        color = 'k'
        ax.errorbar(p, mi, xerr=s, marker='|', markersize=3.5, lw=.5, color=color)

    if param_idx == 2:
        ax.axvline(0, linestyle=':', dashes=[1,1], color='k', lw=.5)

    ax.tick_params(labelsize=FS)
    ax.set_ylim([-.4, len(manips)-1+.4])
    #ax.set_xlim(xlim[param_idx])
    ax.set_yticks(np.arange(len(manips)))
    ax.set_title(names[param_idx], fontsize=FS)
    
    ax.set_yticks([])
    if yticklabels:
        #ax.set_yticklabels([mnames[i] for i in manips])
        ax.set_yticklabels([])

        for mi,m in enumerate(manips):
            h = .3
            wfull = .7
            wdel = .3
            xpad = .1
            
            mini_light_delivery_schematic(ax, x0=-wfull-wdel-xpad, y0=mi-h/2, w=wfull, h=h, m=m, transform=blend(ax.transAxes,ax.transData))
            
    else:
        ax.set_yticklabels([])

    return axs

def ddm_params(axs, panel_id, param_idx, manips=[8,67,0], yticklabels=False):
    '''Display best-fit parameters from DDM fits
    '''
    ax = axs[panel_id]

    names = ['$\\lambda$\nMemory\n← leak', '$\\sigma^2_a$\nAccumulator\nnoise', '$\\sigma^2_s$\nSensory\nnoise', 'Lapse\n', 'Bias\n']
    mnames = {0:'Ctrl',234:'Full CP',67:'Mid-late CP',8:'Delay'}
    xlim = {0:(-.65,.65), 1:(-5,50), 2:(-5,50), 3:(-.1,1), 4:(-3,3)}
    minmaxs = {0:None, 1: (0,None), 2: (0,None), 3:(0,1), 4:None}
    
    params = {}
    se = {}
    
    with h5py.File(data_file) as h:
        for manip in manips:
            grp = h[f'ddm_man{manip}']
            p,s = np.array(grp['fit_params'])
            params[manip] = p
            se[manip] = s

    for mi,manip in enumerate(manips):
        p = params[manip][param_idx]
        s = se[manip][param_idx]
        s = bound_error(s, p, minmaxs[param_idx])
        if param_idx in [1,2] and manip==234:
            color = 'grey'
        elif param_idx == 0 and manip==67:
            color = 'grey'
        elif param_idx == 3 and manip==8:
            color = 'grey'
        else:
            color = 'k'
        ax.errorbar(p, mi, xerr=s, marker='o', markersize=2.5, lw=.5, color=color)

    if param_idx == 0:
        ax.axvline(0, linestyle=':', dashes=[1,1], color='k', lw=.5)

    ax.tick_params(labelsize=FS)
    ax.set_ylim([-.4, len(manips)-1+.4])
    ax.set_xlim(xlim[param_idx])
    ax.set_yticks(np.arange(len(manips)))
    ax.set_title(names[param_idx], fontsize=FS)
    
    ax.set_yticks([])
    if yticklabels:
        #ax.set_yticklabels([mnames[i] for i in manips])
        ax.set_yticklabels([])

        for mi,m in enumerate(manips):
            h = .3
            wfull = .7
            wdel = .3
            xpad = .1
            
            mini_light_delivery_schematic(ax, x0=-wfull-wdel-xpad, y0=mi-h/2, w=wfull, h=h, m=m, transform=blend(ax.transAxes,ax.transData))
            
    else:
        ax.set_yticklabels([])

    # matplotlib bug workaround:
    xt = ax.get_xticks().tolist()
    ax.set_xticklabels(xt)
    def format_num(s):
        s = s.get_text()
        f = float(s)
        if f%1==0:
            return str(int(f))
        else:
            if np.abs(f)<1:
                s = str(float(f))
                s = s.replace('0.','.')
                return s
    ax.set_xticklabels([format_num(i) for i in ax.get_xticklabels()])

    # signif
    print('DDM params')
    for mi,manip in enumerate(manips):
        p0 = params[0][param_idx]
        s0 = se[0][param_idx]
        p = params[manip][param_idx]
        s = se[manip][param_idx]
        print(f'\tParam {param_idx}')
        low,hi = p-1.95*s, p+1.95*s
        print(f'\t\t{mnames[manip]}, {low:0.3f} – {hi:0.3f}')

    return axs

def mini_light_delivery_schematic(ax, x0, y0, w, h, m=0, transform=None, titles=False):
    if m == 0:
        fc = 'none'
    elif m == 234:
        fc = 'dodgerblue'
    else:
        fc = 'none'

    titles = {
            0 : 'Light off',
            234 : 'Cue-period light',
            8 : 'Delay-period light',
            }

    rect = pl.Rectangle((x0,y0), w, h, transform=transform, clip_on=False, edgecolor='k', facecolor=fc, lw=.3)
    ax.add_patch(rect)
    
    if titles:
        ax.text(x0+w/2, y0+h+h/2, titles[m], fontsize=FS-1, ha='center', va='center', transform=transform)

    if m==67:
        wi = w/3
        rect1 = pl.Rectangle((x0+wi, y0+h/2), wi, h/2, transform=transform, clip_on=False, facecolor='dodgerblue', lw=0, zorder=0)
        rect2 = pl.Rectangle((x0+wi*2, y0), wi, h/2, transform=transform, clip_on=False, facecolor='dodgerblue', lw=0, zorder=0)
        ax.add_patch(rect1)
        ax.add_patch(rect2)
    
    if m==567:
        wi = w/3
        rect1 = pl.Rectangle((x0, y0+2*h/3), wi, h/3, transform=transform, clip_on=False, facecolor='dodgerblue', lw=0, zorder=0)
        rect2 = pl.Rectangle((x0+wi, y0+h/3), wi, h/3, transform=transform, clip_on=False, facecolor='dodgerblue', lw=0, zorder=0)
        rect3 = pl.Rectangle((x0+wi*2, y0), wi, h/3, transform=transform, clip_on=False, facecolor='dodgerblue', lw=0, zorder=0)
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
            
    if m == 8:
        wi = 0.21*w
        rect = pl.Rectangle((x0+w, y0), wi, h, transform=transform, clip_on=False, facecolor='dodgerblue', lw=0, zorder=0)
        ax.add_patch(rect)


def easy_trials(axs, panel_id, manips=[0,2,3,4]):
    ax = axs[panel_id]
    
    if all([i in [0,2,3,4] for i in manips]):
        reqstr = '_reqbil'
    elif all([i in [0,5,6,7,8] for i in manips]):
        reqstr = '_reqsub'
    else:
        reqstr = ''

    with h5py.File(data_file) as h:
        easy_means = h5_to_df(h, 'easy_means')
    print(easy_means[easy_means.manip==3])
    
    for mi,man in enumerate(manips):
        e = easy_means[easy_means.manip==man]
        for side in [None]:#[0,1]:
            if side is not None:
                ei = e[e.side==side]
            else:
                ei = e
            ncor = (ei['mean'] * ei['n']).sum()
            ntot = ei['n'].sum()
            frac = ncor/ntot
            #frac = 1-frac if side==0 else frac
            conf = confidence(frac, ntot)
            ofs = -.01 if side==0 else +.01
            ax.errorbar(mi+ofs, frac, yerr=conf, lw=.5, marker='o', markersize=1, mfc='none', mec=mcols[man], ecolor=mcols[man], mew=.5)

    return axs

def ai27d_histology(axs, panel_id):
    ax = axs[panel_id]
    
    '''
    fig = ax.figure
    x,y,w,h = ax.get_position().bounds
    ax.remove()
    
    ax0 = fig.add_axes([x,y,w/2,h])
    ax1 = fig.add_axes([x+w/2,y,w/2,h])

    axs[panel_id] = [ax0,ax1]
    '''
    
    with h5py.File(data_file) as h:
        img0 = np.array(h['ai27d_histology']['img0'])
        img1 = np.array(h['ai27d_histology']['img1'])
    
    ax.imshow(img0)

    ax.axis('off')

    #ax1.imshow(img1)
    #ax1.axis('off')
    #ax.text(.5, .5, 'TO\nREPLACE', color='white', weight='bold', ha='center', va='center', transform=ax.transAxes, fontsize=FS)

    return axs

def error_ellipse(x_bf, hessian, idx, width):
    
    # x_bf : best-fit parameter for the offset of ellipse
    # hessian : Hessian matrix for covariance matrix
    # idx : id of parameters to get the covrariance error ellipse

    covariance = np.linalg.inv(hessian)
    covariance = covariance[idx][:,idx]

    eigenval, eigenvec = np.linalg.eig(covariance)

    # Get the largest eigenvalue
    # Get the index of the largest eigenvector
    largest_eigenvec_ind_c = np.argmax(eigenval)
    largest_eigenval = eigenval[largest_eigenvec_ind_c]
    largest_eigenvec = eigenvec[:, largest_eigenvec_ind_c]

    # Get the smallest eigenvector and eigenvalue
    if largest_eigenvec_ind_c == 0:
        smallest_eigenval = eigenval[1]
        smallest_eigenvec = eigenvec[:,1]
    else:
        smallest_eigenval = eigenval[0]
        smallest_eigenvec = eigenvec[0,:]

    # Calculate the angle between the x-axis and the largest eigenvector
    angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])

    # This angle is between -pi and pi.
    # Let's shift it such that the angle is between 0 and 2pi
    if angle < 0:
        angle = angle + 2*np.pi

    # % Get the 95% confidence interval error ellipse
    #chisquare_val_95 = 2.1459

    theta_grid = np.linspace(0,2*np.pi,1000)
    phi = angle

    # % x0,y0 ellipse centre coordinates
    X0=x_bf[idx[0]]
    Y0=x_bf[idx[1]]
    a=np.sqrt(largest_eigenval)
    b=np.sqrt(smallest_eigenval)

    # % the ellipse in x and y coordinates
    ellipse_x_r  = width*a*np.cos( theta_grid )
    ellipse_y_r  = width*b*np.sin( theta_grid )

    # %Define a rotation matrix
    R = np.array([ [np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)] ])

    # let's rotate the ellipse to some angle phi
    r_ellipse = np.array([ellipse_x_r, ellipse_y_r]).T @ R

    Xs = r_ellipse[:,0] + X0
    Ys = r_ellipse[:,1] + Y0
    return Xs,Ys

